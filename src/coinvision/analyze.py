from __future__ import annotations

import csv
import hashlib
import statistics
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from .utils import human_bytes, parse_year

GRAYSCALE_MODES = {"1", "L", "LA", "I", "I;16", "I;16B", "I;16L", "F", "I;16N"}

EXTENSION_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".jpe": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}

PROBLEM_REASONS = ("empty", "not_an_image", "unreadable", "corrupted", "truncated")

ANALYZE_COLUMNS = (
    "image",
    "year",
    "format",
    "mode",
    "color",
    "width",
    "height",
    "bytes",
    "status",
    "reason",
    "detail",
    "file_sha256",
    "pixel_sha256",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def check_image_file(path: str | Path, deep: bool = True) -> dict:
    path = Path(path)
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return {"status": "fail", "reason": "unreadable", "detail": str(exc), "bytes": None}

    if size_bytes == 0:
        return {"status": "fail", "reason": "empty", "detail": "0-byte file", "bytes": 0}

    try:
        with Image.open(path) as img:
            fmt = img.format
            width, height = img.size
            mode = img.mode
            if deep:
                img.verify()
    except UnidentifiedImageError as exc:
        return {"status": "fail", "reason": "not_an_image", "detail": str(exc), "bytes": size_bytes}
    except (OSError, SyntaxError, ValueError) as exc:
        return {"status": "fail", "reason": "corrupted", "detail": str(exc), "bytes": size_bytes}

    if deep:
        try:
            with Image.open(path) as img:
                img.load()
                pixel_sha256 = hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()
        except (OSError, SyntaxError, ValueError) as exc:
            detail = str(exc)
            reason = "truncated" if "truncat" in detail.lower() else "corrupted"
            return {"status": "fail", "reason": reason, "detail": detail, "bytes": size_bytes}
    else:
        pixel_sha256 = None

    return {
        "status": "ok",
        "bytes": size_bytes,
        "width": width,
        "height": height,
        "mode": mode,
        "format": fmt,
        "color": "grayscale" if mode in GRAYSCALE_MODES else "color",
        "pixel_sha256": pixel_sha256,
    }


def _default_csv_path(directory: Path) -> Path:
    return directory.parent / f"{directory.name}_analyze.csv"


def analyze_to_csv(
        directory: str | Path,
        recursive: bool = True,
        deep_check: bool = True,
        progress: bool = True,
) -> Path:
    directory = Path(directory)
    bars_off = not progress or not sys.stderr.isatty()

    out_csv = _default_csv_path(directory)

    iterator = directory.rglob("*") if recursive else directory.glob("*")
    image_paths = sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in EXTENSION_FORMATS
    )

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(ANALYZE_COLUMNS)
        inspect = tqdm(
            image_paths, desc="analyze images", unit=" img", disable=bars_off
        )
        for path in inspect:
            file_digest = ""
            try:
                file_digest = sha256_file(path)
            except OSError as exc:
                probe = {
                    "status": "fail",
                    "reason": "unreadable",
                    "detail": str(exc),
                    "bytes": "",
                }
            else:
                probe = check_image_file(path, deep=deep_check)

            row: dict[str, object] = {
                "image": path.as_posix(),
                "year": parse_year(path) or "",
                "format": "",
                "mode": "",
                "color": "",
                "width": "",
                "height": "",
                "bytes": "",
                "status": "",
                "reason": "",
                "detail": "",
                "file_sha256": file_digest,
                "pixel_sha256": "",
            }

            row.update(probe)
            writer.writerow([row[column] for column in ANALYZE_COLUMNS])
        inspect.close()
    return out_csv


def _counter_to_dict(counter: Counter) -> dict:
    return {str(key): counter[key] for key in sorted(counter, key=lambda k: str(k))}


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolution_stats(sizes: list[tuple[int, int]]) -> dict:
    if not sizes:
        return {"unique": 0, "min": None, "max": None}
    areas = [w * h for w, h in sizes]
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    return {
        "unique": len(set(sizes)),
        "min": [int(value) for value in sizes[areas.index(min(areas))]],
        "max": [int(value) for value in sizes[areas.index(max(areas))]],
        "width_mean": round(statistics.fmean(widths), 1),
        "height_mean": round(statistics.fmean(heights), 1),
    }


def _size_stats(byte_sizes: pd.Series) -> dict:
    if not byte_sizes:
        return {"total": 0, "human_total": "0 B", "median": 0.0}
    return {
        "total": sum(byte_sizes),
        "human_total": human_bytes(sum(byte_sizes)),
        "median": round(statistics.median(byte_sizes), 1),
    }


def summarize(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["ext"] = (
        df["image"].astype("string")
        .str.extract(r".*(\.[^./\\]+)$", expand=False)
        .str.lower()
    )
    for col in ("width", "height", "bytes"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ok = df["status"].eq("ok")
    ok_df = df[ok]

    years = df["year"].astype("string").replace("", pd.NA).dropna()
    year_counts = years.value_counts()

    reasons = (
        df["reason"].astype("string").replace("", pd.NA)
        .fillna(df["status"].astype("string").replace("", pd.NA))
        .fillna("unknown")
    )[~ok].value_counts()

    resolutions = list(map(tuple, ok_df[["width", "height"]].dropna().astype(int).to_numpy()))
    sizes = ok_df["bytes"].dropna().astype(int).tolist()
    res = _resolution_stats(resolutions)
    size = _size_stats(sizes)

    expected = ok_df["ext"].map(EXTENSION_FORMATS).astype("string")
    reported = ok_df["format"].astype("string").replace("", pd.NA)
    mismatch = int((reported.notna() & expected.notna() & (reported != expected)).sum())

    return pd.Series({
        "Files": len(df),
        "Extensions": df["ext"].fillna("").value_counts().to_dict(),
        "Formats": ok_df["format"].value_counts(dropna=False).to_dict(),
        "Modes": ok_df["mode"].value_counts(dropna=False).to_dict(),
        "Color": ok_df["color"].value_counts(dropna=False).to_dict(),
        "Resolution": f"{res['unique']} unique, min {res['min']}, max {res['max']}",
        "File size": f"{size['human_total']} total, median {human_bytes(size['median'])}",
        "Year in name": f"{years.size} with / {len(df) - years.size} without",
        "Years": f"{year_counts.size} unique, {years.min()}..{years.max()}",
        "Year counts": year_counts.to_dict(),
        "Integrity": f"{int(ok.sum())}/{len(df)} ok, {int((~ok).sum())} problems",
        "Problem reasons": reasons.to_dict() if len(reasons) else "none",
        "Format mismatch": mismatch,
    }, dtype=object)


def format_report(stats: pd.Series) -> str:
    return stats.to_string()


RESOLUTION_PERCENTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def resolution_stats(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].eq("ok")].copy()
    for col in ("width", "height"):
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    ok = ok.dropna(subset=["width", "height"])
    ok = ok[(ok["width"] > 0) & (ok["height"] > 0)]

    metrics = {
        "width": ok["width"].astype(float),
        "height": ok["height"].astype(float),
        "short_side": ok[["width", "height"]].min(axis=1).astype(float),
        "aspect_ratio": (ok["width"] / ok["height"]).astype(float),
        "megapixels": (ok["width"] * ok["height"] / 1_000_000).astype(float),
    }

    rows = []
    for name, series in metrics.items():
        if series.empty:
            continue
        desc = series.describe(percentiles=RESOLUTION_PERCENTILES)
        rows.append(
            {
                "metric": name,
                "count": int(desc["count"]),
                "min": round(float(desc["min"]), 2),
                "p1": round(float(desc["1%"]), 2),
                "p5": round(float(desc["5%"]), 2),
                "p25": round(float(desc["25%"]), 2),
                "p50": round(float(desc["50%"]), 2),
                "p75": round(float(desc["75%"]), 2),
                "p95": round(float(desc["95%"]), 2),
                "p99": round(float(desc["99%"]), 2),
                "max": round(float(desc["max"]), 2),
                "mean": round(float(desc["mean"]), 2),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def resolution_stats_from_csv(path: str | Path) -> pd.DataFrame:
    return resolution_stats(pd.read_csv(path))


def format_resolution_report(table: pd.DataFrame) -> str:
    return table.to_string()


def summary_from_csv(path: str | Path) -> dict:
    return summarize(pd.read_csv(path))

def main(directory: str, deep_check: bool = True, ) -> Path:
    path = analyze_to_csv(directory, deep_check=deep_check, progress=True)
    print(format_report(summary_from_csv(path)))
    print(f"\nCSV -> {path}")
    return path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a coin image dataset into a CSV")
    parser.add_argument("directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--deep",
        dest="deep_check",
        action="store_true",
        default=True,
        help="full decode + pixel hashes (default)",
    )
    mode.add_argument(
        "--shallow",
        dest="deep_check",
        action="store_false",
        help="header-only check, no pixel hashes",
    )
    args = parser.parse_args()
    main(args.directory, deep_check=args.deep_check)
