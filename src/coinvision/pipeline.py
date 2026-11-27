"""The date-localisation pipeline: candidates from the ring sweep and from the detector, one score.

Two sources propose where the year might be. The ring sweep is label-free and covers the annulus
blindly; the detector predicts the block's rectangle outright. Their candidate sets overlap but each
one covers photos the other misses - union coverage on the annotated set is 0.995 against 0.917 for
the sweep alone - so both run and a single ranker scores the merged list. One ranker, not two,
because two separately trained models have no reason to agree on what a score means.

Every candidate is reduced to four corners, so one crop routine serves both and the demo can draw
either without knowing where it came from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from PIL.Image import Resampling, Transform

from coinvision.localize import Circle, DateCandidate, detect_coins, ring_windows, to_gray

# The ranker reads 160x80 crops; the aspect matches a year block, four digits side by side.
CROP_SIZE = (160, 80)

# The ranker saw each source under that source's own filter, because the two caches were built
# separately: arc windows went through crop_upright (bicubic) and detector quads through quad_crop
# (bilinear). Serving has to keep that split - cropping a detector quad bicubically moved the
# end-to-end number from 0.928 to 0.910 on the first 1200 photos of a re-run, which is the size of a
# train/serve mismatch that hides behind the fold spread.
CROP_FILTER = {"ring": Resampling.BICUBIC, "detector": Resampling.BILINEAR}
CROP_FILTER_DEFAULT = Resampling.BICUBIC


@dataclass
class Candidate:
    """One place the year could be: four corners, in reading order, plus where it came from."""

    corners: np.ndarray
    source: str
    score: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        return float(self.corners[:, 0].mean()), float(self.corners[:, 1].mean())


@dataclass
class Answer:
    """What the pipeline decided, with enough detail to draw it and to argue about it."""

    has_date: bool
    score: float
    threshold: float
    coins: list[Circle] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    best: Candidate | None = None
    crop: Image.Image | None = None

    @property
    def top(self) -> list[Candidate]:
        return sorted(self.candidates, key=lambda item: item.score, reverse=True)[:5]


def window_corners(candidate: DateCandidate) -> np.ndarray:
    """Corners of a ring window's rotated rectangle, in the order the crop expects."""
    half_arc = 0.5 * float(candidate.arc_span)
    half_radial = 0.5 * float(candidate.radial_span)
    angle = np.radians(float(candidate.angle_deg))
    # the crop rotates by -angle_deg, so the reading direction is (cos, sin) of angle_deg
    ux, uy = float(np.cos(angle)), float(np.sin(angle))
    vx, vy = -uy, ux
    cx, cy = candidate.center
    corners = [
        (
            cx + sx * half_arc * ux + sy * half_radial * vx,
            cy + sx * half_arc * uy + sy * half_radial * vy,
        )
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return np.asarray(corners, dtype=np.float64)


def crop_corners(
    image: Image.Image,
    corners: np.ndarray,
    size: tuple[int, int] = CROP_SIZE,
    resample: Resampling = CROP_FILTER_DEFAULT,
):
    """Resample the quadrilateral into an upright crop, x1->x2 running along the reading direction.

    This is the one crop routine of the stage, and the filter is an argument because the ranker was
    trained on two different ones - see CROP_FILTER.
    """
    width, height = size
    origin = corners[0]
    a = (corners[1] - origin) / width
    b = (corners[3] - origin) / height
    return image.convert("RGB").transform(
        size,
        Transform.AFFINE,
        (a[0], b[0], origin[0], a[1], b[1], origin[1]),
        resample=resample,
    )


# The reader is a second model with a different appetite: it was trained on tight year blocks at
# 448x224. The localiser's windows are wider than the year by design - the metric is containment, so
# a window only has to hold the year, not frame it. Feeding a ring window straight to the reader
# turned a confident 1832 into a shaky 1837, because the digits shrink threefold and the framing is
# wrong. reading_crop hands the reader a box the size of the year instead.
#
# The two sources need different treatment, measured over 200 annotated photos: the annotated year
# fills 0.34 x 0.23 of a winning ring window but 0.73 x 0.68 of a winning detector quad, so the ring
# one is shrunk by the first ratio and the detector one is used as it comes.
READ_SIZE = (448, 224)
READ_SHRINK = {"ring": (0.34, 0.23), "detector": (1.0, 1.0)}


def _axes(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Centre, unit vector along the reading direction, and the two full side lengths."""
    width = float(np.linalg.norm(corners[1] - corners[0]))
    height = float(np.linalg.norm(corners[3] - corners[0]))
    along = (corners[1] - corners[0]) / max(width, 1e-9)
    return corners.mean(axis=0), along, width, height


def _rect(center: np.ndarray, along: np.ndarray, width: float, height: float) -> np.ndarray:
    """The four corners of a rectangle of that size, turned with the reading direction."""
    across = np.array([-along[1], along[0]])
    half_width, half_height = width / 2.0, height / 2.0
    return np.array(
        [
            center - half_width * along - half_height * across,
            center + half_width * along - half_height * across,
            center + half_width * along + half_height * across,
            center - half_width * along + half_height * across,
        ]
    )


def reading_box(candidates: Sequence[Candidate], best: Candidate) -> np.ndarray:
    """The tight quadrilateral the reader should see, built from whichever candidate won."""
    shrink_x, shrink_y = READ_SHRINK.get(best.source, (1.0, 1.0))
    center, along, width, height = _axes(np.asarray(best.corners, dtype=np.float64))
    return _rect(center, along, width * shrink_x, height * shrink_y)


def _fit_aspect(corners: np.ndarray, aspect: float) -> np.ndarray:
    """Grow the shorter side so the crop is not stretched: the reader learned one aspect only."""
    center, along, width, height = _axes(np.asarray(corners, dtype=np.float64))
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    return _rect(center, along, width, height)


def reading_crop(
    image: Image.Image,
    candidates: Sequence[Candidate],
    best: Candidate,
    size: tuple[int, int] = READ_SIZE,
):
    """The crop that goes to the reader: the year's size, at the reader's aspect."""
    box = _fit_aspect(reading_box(candidates, best), size[0] / size[1])
    return crop_corners(image, box, size, CROP_FILTER.get(best.source, CROP_FILTER_DEFAULT))


def crop_candidate(image: Image.Image, candidate: Candidate, size: tuple[int, int] = CROP_SIZE):
    """The crop for one candidate, under the filter its source was trained with."""
    return crop_corners(
        image, candidate.corners, size, CROP_FILTER.get(candidate.source, CROP_FILTER_DEFAULT)
    )


def score_candidates(scorer, image: Image.Image, candidates: Sequence[Candidate]) -> None:
    """Fill in every candidate's score in one batched call; the scorer wants 160x80 RGB floats."""
    if not candidates:
        return
    crops = np.stack(
        [
            np.asarray(crop_candidate(image, candidate), dtype=np.float32) / 255.0
            for candidate in candidates
        ]
    )
    logits = scorer.run(None, {"windows": crops.transpose(0, 3, 1, 2)})[0].reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.score = float(score)


def run(
    image: Image.Image,
    scorer,
    detector=None,
    meta: dict | None = None,
    threshold: float = 0.1,
    max_coins: int = 2,
    top_k_detector: int = 3,
) -> Answer:
    """Locate the year: ring sweep plus detector, one ranker, best crop out.

    `detector` is a `coinvision.date_detector.DateDetector`; without it the pipeline degrades to
    the ring sweep alone, which is the shipped behaviour before the detector existed.
    """
    meta = meta or {}
    sweep = meta.get("sweep", {})
    bands = tuple(tuple(band) for band in sweep.get("bands", ((0.30, 1.10), (0.60, 1.25))))
    arc = float(sweep.get("arc_degrees", 46.0))
    step = float(sweep.get("step_degrees", 11.5))

    gray = to_gray(image)
    coins = detect_coins(gray, max_coins=max_coins)

    candidates: list[Candidate] = []
    for circle in coins:
        for window in ring_windows(circle, arc, step, bands):
            candidates.append(Candidate(window_corners(window), "ring"))
    if detector is not None:
        for quad in detector.quads(image, top_k=top_k_detector):
            candidates.append(Candidate(np.asarray(quad, dtype=np.float64), "detector"))

    score_candidates(scorer, image, candidates)
    best = max(candidates, key=lambda item: item.score) if candidates else None
    score = best.score if best else 0.0
    return Answer(
        # nothing to crop is the only refusal the pipeline can justify without labelled negatives
        has_date=bool(best) and score >= threshold,
        score=score,
        threshold=threshold,
        coins=coins,
        candidates=candidates,
        best=best,
        crop=None if best is None else crop_candidate(image, best),
    )
