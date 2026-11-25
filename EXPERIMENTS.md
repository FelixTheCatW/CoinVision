# Experiment log

Every number below was produced by a real run on the GTX 1050 Ti in this repository and lives in
`runs/` (git-ignored). "Last-digit accuracy" is the metric that matters: years 1830-1837 share the
prefix `183`, so the task is a 7-way classification of the final digit.

## Dataset

| year | source | manifest | train | val |
|---|---|---|---|---|
| 1830 | 26 | 26 | 21 | 5 |
| 1831 | 240 | 240 | 192 | 48 |
| 1832 | 599 | 587 | 470 | 117 |
| 1833 | 181 | 178 | 142 | 36 |
| 1835 | 142 | 140 | 112 | 28 |
| 1836 | 277 | 274 | 219 | 55 |
| 1837 | 318 | 311 | 249 | 62 |
| total | 1783 | 1756 | 1405 | 351 |

27 images were dropped as pixel duplicates; there is no 1834. The photos are a mixture of NGC slabs
(the year is printed on the label) and raw obverse/reverse pairs (the year is engraved, a few dozen
pixels high in the original 1600x800 frame).

## Protocol

- `coinvision split data/manifests/all.csv --folds 5` writes five folds that are stratified by year
  and never split the shots of one listing across train/val (42 of 351 val images used to share a
  listing with train).
- All comparisons run all five folds, `--deterministic`, with the in-memory pixel cache.
- Determinism is verified: two runs of the same config reproduce the loss to nine decimal places.

## Results

| # | change | last-digit acc | macro F1 | scope |
|---|---|---|---|---|
| 1 | from scratch, 30 ep | 0.516 | 0.380 | single split |
| 2 | + ImageNet init, 30 ep | 0.8376 | 0.7973 | single split |
| 3 | + grouping, deterministic, 30 ep | 0.8519 +- 0.0178 | 0.8115 | 5 folds, pooled 1496/1756 |
| 4 | balanced class weights | 0.7927 +- 0.0116 | 0.7285 | 5 folds |
| 5 | 60 epochs | 0.8588 +- 0.0154 | 0.8203 | 5 folds |
| 6 | input 896x448, batch 8, 30 ep | 0.8079 | 0.6810 | fold0 |
| 7 | crops of the native photo, 30 ep | 0.6300 | 0.4070 | fold0 |
| 8 | stacked views (frame+label+date, 9 ch) | 0.7542 | 0.6264 | fold0 |
| 9 | test-time views (3 flip-free) | 0.8475 / 0.8409 / 0.8262 | - | folds 0-2 |
| 10 | **augment + 60 ep + smoothing + cosine** | **0.8758 +- 0.0199** | **0.8149** | 5 folds, pooled 1538/1756 |
| 11 | + bag of two seeds | 0.8833 | 0.8386 | 5 folds, pooled 1551/1756 |

Reference points for rows 6-9: the plain fold0 model scores 0.8616 there.

### Row 10, the recipe now in `configs/default.toml`

Screen on fold0 first: baseline 0.8616 / macro F1 0.7999, augment+60 ep 0.8898 / 0.8609, label
smoothing alone 0.8362 / 0.7535, cosine alone 0.8616 / 0.8420, combination 0.9068 / 0.8721.
Full cross-validation of the combination: +2.39 pp paired over the folds (sd 1.68, t = 3.17).
Best epochs land at 39-57 of 60, so the old 30-epoch budget was cutting training short.

Per digit, pooled over the folds (30-epoch baseline -> recipe):

| digit | n | recall | F1 |
|---|---|---|---|
| 0 | 26 | 0.615 -> 0.462 | 0.762 -> 0.600 |
| 1 | 240 | 0.942 -> 0.950 | 0.917 -> 0.929 |
| 2 | 587 | 0.917 -> 0.934 | 0.895 -> 0.904 |
| 3 | 178 | 0.697 -> 0.725 | 0.740 -> 0.777 |
| 5 | 140 | 0.650 -> 0.700 | 0.674 -> 0.745 |
| 6 | 274 | 0.807 -> 0.869 | 0.807 -> 0.847 |
| 7 | 311 | 0.900 -> 0.916 | 0.900 -> 0.931 |

Everything improved except 1830, which label smoothing and augmentation squeeze further; bagging two
seeds brings it back to 0.577 recall (F1 0.698) and is the only thing that measurably helps it.

## Finding the date before classifying it

`src/coinvision/localize.py` is the label-free pre-step: coin circle, ring unwrap, upright crops. It
exists because the classifier cannot rescue a frame in which the year is a few pixels high, and it
answers confidently on frames it has never seen (a blank white canvas scores 0.66 on 1836).

What works, measured:

| step | evidence |
|---|---|
| circle detection (gradient voting + rim profile) | synthetic disc: radius within 12%, centre within 25 px; a two-coin frame: both circles found, radius within 15% |
| ring unwrap and upright crop | a candidate at -90 deg turns vertical stripes horizontal and the reverse; crops are resampled with an explicit affine, so the orientation is testable |
| coverage | 31 windows per band, two bands, 46 deg wide at an 11.5 deg step: any run of four digits spans far less than 46 deg, so it lies inside at least one window by construction |

The rim radius is now the outermost strong ring of the radial gradient profile, not the tallest one:
a shield or a portrait closes its own contour inside the rim and used to win the peak, which put the
year outside the circle on two photos out of five. Measured over the 271 `engraved` annotations (the
quad is assigned to the circle whose rim is nearest to it):

| rim radius | p50 | p75 | p90 | year outside the circle | a window holds the year |
|---|---|---|---|---|---|
| tallest peak (before) | 0.99 | 1.04 | 1.80 | 117/271 = 43% | 83% |
| outermost strong peak (now) | 0.97 | 1.01 | 1.18 | 76/271 = 28% | 91% |

The last column is the demo's geometry - 110 deg arc, 12 deg step, one band 0.32-1.02 R - counted as
containment of the whole quad at >= 0.99. Both columns are the share of photos, not of quads.

A circle whose rim is not visible round the whole arc is not a coin, and the accumulator does find
such circles: on `6a5efac8ba5568d05aa7c8d6-1-1837.webp` it returned (848, 549, r=162) on the caption
beside the coin. Candidates are now scored by their rim - the 10th percentile of the per-angle
gradient peak in a +-10% band, over its median - and skipped below 0.12, with the next peak tried
instead: a real rim scores 0.50-0.85, a caption 0.00. The filter drops 36 of 271 circles, 32 of them
junk and 4 that happened to hold the year, so "year outside the circle" stays at 28% while "a circle
smaller than the coin" disappears from the previews. Two circles are found on 251 of 270 photos
instead of 268; the rest keep the one circle that passed.

The sweep step is 24 deg, not 12: neighbouring windows used to overlap by 89%, nine copies of one arc.
At 15 windows per coin (30 per frame instead of 60) the share of years inside a single window is
90.8% against 91.1% at step 12, and a request drops from ~1.2 s to ~0.8 s.

The ranker was retrained on that grid: recall@1@0.9 0.912 +- 0.052 over five folds, up from 0.821,
recall@5@0.9 0.936, median top-1 containment 1.000, ceiling 94% (15 photos have no window holding the
whole year). It takes three negatives per positive window now, not six: at 1:6 sigmoid sat at 0.43 on
true windows, so the demo's 0.5 threshold rejected half of the photos whose top window did hold the
year. At 1:3 positives score 0.98 and 98% of photos pass, 94% of them with the year in the top window.
That grid is superseded by the two-band one below: recall@1@0.9 0.907 +- 0.056 at a 97.2% ceiling.

What does not work yet:

- **Choosing the window by the classifier's own confidence.** Two runs, both ranked by the product of
  the digit probabilities:
  - glyph runs only (five candidates per coin, 16 photos): 2/16 correct, while the same model on the
    same 16 photos scores 16/16 on the full frame, and some candidate was right in 6/16;
  - the full ring sweep (124 candidates per photo, 12 photos): 1/12 correct. The fraction of runs
    where some candidate is right is not usable evidence here - with 124 guesses per photo, a model
    biased towards the common classes hits the right one by chance.
  In both cases the confidence peaks on windows holding shield detail or legend letters, so it is an
  anti-selector, not a selector.
- **Glyph grouping.** After unwrapping, dark blobs are grouped into runs of four glyphs of similar
  height at one radius. On a real photo the filter keeps 131 glyphs per coin (legend letters plus the
  beaded border, which starts around 0.95R) and the top-scoring run lands on "REX" of the legend -
  confirmed by eye on the six best crops of one photo: none contained the year.

Consequence: the localiser can generate candidates and upright crops, but deciding which one is the
year needs supervision - either boxes for a detector, or a classifier trained on crop labels. Until
one of those exists, the whole-frame model stays the thing that answers.

## The grid was the ceiling: 26 misses and a second band

Measured over the 270 engraved annotations, with the year counted as covered when at least 90% of
the annotated quad falls inside a window - through the same affine that builds the crop, sampling a
64x64 grid inside the quad, not through the training cache:

| grid | photos | covered | windows per frame |
|---|---|---|---|
| one band 0.32-1.02, step 24 (before) | 270 | 0.904 | 29 |
| one band to 1.10 | 270 | 0.907 | 29 |
| one band to 1.14 | 270 | 0.915 | 29 |
| step 12 instead of 24 | 270 | 0.904 | 58 |
| arc 120 + band to 1.12 | 270 | 0.915 | 29 |
| two bands 0.30-1.10 + 0.60-1.25 (now) | 270 | 0.933 | 58 |

Every one of the 26 photos the single band misses has the year's centre outside the detected circle.
Eight of them are at 1.12-1.32 R with the year reaching 1.23-1.50 R: just past the outer edge of the
band, on circles whose rim score is a real 0.39-0.78, so the circle is right and only the band is
too narrow. The other 18 are 1.6-3.8 R away, i.e. the detected circle is not the coin at all - 19 of
those photos also carry a slab annotation, and a contact sheet of the 18 quads showed 17 of them with
the year engraved in metal: the coin is simply small in the frame and the detector found something
else.

Rejected: raising `max_coins` to 3, which "recovered" 24 of the 26. The rim-score probe shows why that
number is empty - of the 34 candidate circles that do cover a year, the median rim score is 0.006 and
23 of them score below the 0.12 filter, i.e. they are exactly the peaks the filter throws away. A
circle whose centre is off by two radii makes windows all over the frame, and one of them lands on
the year by accident. Proposals got sprayed; the coin was not found.

Adopted: the second band. Same 269 photos, same code, only the grid changed:

| | one band | two bands |
|---|---|---|
| windows per frame | 30 | 58 |
| crops in the cache | 7815 | 14880 |
| windows positive at containment 0.9 | 500 | 924 |
| ceiling, cache metric | 94% (15 photos) | 97.2% (7 photos) |
| ceiling, crop-affine metric | 0.904 | 0.937 |
| recall@1@0.9, five folds | 0.912 +- 0.052 | 0.907 +- 0.056 |
| recall@5@0.9, five folds | 0.936 | 0.964 |
| recall@1@0.9 on photos that have a good window | ~0.97 | 1.000 +- 0.000 |
| verdict passes, end to end | 0.956 | 0.996 |
| of passes, top-1 holds the year | 0.903 | 0.877 |
| seconds per photo, CPU | 0.47 | 0.44 |

The read: the ceiling moved 3.3 pp and the verdict now passes nearly every photo that has a date,
while recall@1 did not move at all - the ranker was already at the ceiling of the old grid, and is
now perfect on every photo whose proposals contain a good window. What got worse is precision at the
0.5 threshold (0.903 -> 0.877): the wider band lets borderline windows through, so the threshold is
the next thing to look at, not the model.

Two data notes from the same pass:

- One photo (`6708f0b96213d00cabaee8c4-2-1832.webp`, a quad at 0.9 deg with the digits on white stock)
  was reclassified `engraved` -> `slab`: its crop shows printed text, not metal.
- 22 photos carry both an `engraved` and a `slab` row (a slab photo that shows the coin as well as the
  label). `annotation_kinds` keeps the kind of the first filled row, so 19 of them leave the ranker's
  training set: 250 of the 277 boxed photos train. Fixing that is worth ~19 photos of engraved data,
  and it is the reason the "270 annotated years" and "250 training photos" do not match.

## What 1600 annotated photos bought

The ranker was the last thing that could be improved by looking at data instead of at code, so the
whole annotation file was filled in: 1770 boxes over 1747 photos, up from 300 boxes over 277. The
pass also tagged the domain - 114 rows are printed labels on NGC stock and are excluded, so 1637
photos train. Five photos carry no box and no kind: they are the first real "no legible date"
examples and they are not enough to train a negative class.

The cache over that set: 97170 windows, 5793 positives at containment 0.9 (6.0%, the same rate as
before), and 56 photos where no window holds the whole year, i.e. a ceiling of 96.6%.

| | 250 photos | 1637 photos |
|---|---|---|
| folds, recall@1@0.9 | 0.816 0.880 0.980 0.918 0.942 | 0.932 0.924 0.958 0.933 0.910 |
| pooled recall@1@0.9 | 0.907 +- 0.056 | 0.931 +- 0.015 |
| recall@5@0.9 | 0.964 +- 0.027 | 0.963 +- 0.004 |
| recall@1@0.9 given a good window | 1.000 | 1.000 |
| ceiling, cache metric | 97.2% | 96.6% |

The mean moved 2.4 pp, which is the smaller half of the result: the fold spread fell from +-0.056 to
+-0.015. One fold used to give 0.816 and another 0.980, so any model comparison was noise; now the
number is reproducible and A/B changes can be read off it.

End to end on those same 1747 photos, the two models through the identical pipeline:

| | 250-photo model | 1637-photo model |
|---|---|---|
| recall@1@0.9 | 0.842 | 0.872 |
| ceiling | 0.917 | 0.917 |
| verdict passes at 0.5 | 0.978 | 0.888 |
| of passes, top-1 right | 0.859 | 0.943 |
| seconds per photo, CPU | 0.52 | 0.42 |

Ranking improved and the score scale moved. On the reference photo every window holding the year
scores 0.105 / 0.066 / 0.028 and every window that does not is below 0.018; the 250-photo model put
the same three at 0.956 / 0.906 / 0.833 but also lifted a window with no year at all to 0.869. The
new model separates the classes better in rank and worse in level, so the 0.5 cut - calibrated on the
old level - rejected photos it used to accept.

Sweeping the cut over the 1747 photos:

| threshold | passes | of passes, top-1 right |
|---|---|---|
| 0.02 | 0.983 | 0.886 |
| 0.05 | 0.977 | 0.892 |
| **0.10** | **0.971** | **0.897** |
| 0.15 | 0.961 | 0.904 |
| 0.20 | 0.949 | 0.911 |
| 0.30 | 0.933 | 0.921 |
| 0.50 | 0.888 | 0.943 |

0.10 is the operating point: it answers on 97.1% of photos against 97.8% for the old model, and 3.8
pp more of those answers are right. It now lives in the model metadata as `threshold` and both
consumers read it, so the cut travels with the score scale that produced it instead of being a
constant in the demo and another one in the minimal solution.

Rejected in the same sweep: a top1/top2 gap rule. Gap >= 0.01 drops coverage to 0.772 at 0.874
precision, and 0.05 to 0.401. Negatives are near zero now, so the gap is almost the score itself,
and the rule only fires on photos where the year sits in two overlapping windows - it throws away
correct answers, not wrong ones.

Where the remaining error is: precision cannot exceed the 0.917 ceiling, and `recall@1@0.9 given a
good window` is 1.000, so nothing is left to win in the ranker. The 8.3% of photos without a
suitable window are a detection problem - the coin is small in the frame, or the circle found is not
the coin.

One practical rule for whoever regenerates the master file: it must not carry a UTF-8 BOM.
`read_manifest` opens it as plain utf-8, so a BOM renames the first column to `\ufeffimage` and every
consumer - this script and `date_annotations.py check` - silently sees no image paths at all.

## What did not work, and why

- **Class weights** cost 5.9 pp. With 21-26 images in the rare class the weights redistribute
  gradient mass away from well-represented classes without adding information; every digit got worse.
- **Higher resolution (896x448)** lost 5.4 pp on fold0 at the same 30-epoch budget. Train accuracy
  only reached 0.97 by epoch 28, so the run was underfed, not defeated - but a fair test needs
  60-90 epochs at 174 s each, i.e. ~7-14 h per cross-validation.
- **Crops of the native photo** lost 23 pp. Sampling a date crop, a slab-label crop and a random
  crop as separate samples means only ~12% of the supervision can actually read the year, and
  averaging the three views at evaluation dilutes the good view with two useless ones.
- **Stacked views** lost 10.7 pp with train accuracy 0.984: the extra detail buys memorisation on
  1400 training images, not date reading.
- **Test-time views** cost 0.3-2.0 pp. Without a flip (which would mirror the date) the three views
  are too correlated to help.
- **Bagging fold models** is invalid, not just weak: each fold model trained on the other folds'
  validation images, so it scores ~1.000 on them. Honest bagging needs several seeds per fold.

## Auditing the errors

The 260 misclassified images were ranked by confidence: the worst ones sit at p_true ~ 0.000, and
spot-checking five of them (two slabs, three raw coins) showed the filename year is correct in every
case. The model is confidently wrong because at 448x224 for the whole photo the year is 15-20 px
high; the same crop of the original, resized to the same 448x224, reads "1830" and
"1837 G.BRITAIN 1SOV" without effort. The information exists, this input throws it away, and the
attempts above show the network cannot recover it from 1400 images.

## Throughput notes

| step | before | after |
|---|---|---|
| epoch, 448x224 | 45-48 s | 21 s (pixel cache) |
| GPU utilisation | 40% | 88% |
| epoch with augmentation | 66 s | 26-29 s (loader workers) |
| epoch, native cache for crops | - | 4.7 GiB RAM |
| epoch, 896x448 | - | 174 s |
| deterministic mode | 20.9 s | 24.7-25.3 s (+18%) |

## Reproducing

```bash
uv sync --extra dev
uv run coinvision split data/manifests/all.csv --folds 5 --out-dir data/manifests/folds5
for i in 0 1 2 3 4; do
  uv run coinvision train --train-manifest data/manifests/folds5/fold${i}_train.csv \
    --val-manifest data/manifests/folds5/fold${i}_val.csv \
    --out-dir runs/cv5_combo/fold${i} --deterministic
done
uv run coinvision ensemble --manifest data/manifests/folds5/fold0_val.csv \
  --checkpoint runs/cv5_combo/fold0/best.pt --checkpoint runs/cv5_combo_seed1/fold0/best.pt
```

## Open levers

1. **Data.** 1833 (recall 0.753) and 1835 (0.679) hold most of the remaining error mass, and 1830 has
   26 images total. Nothing in the pipeline compensates for that.
2. **A real date detector.** Every crop attempt failed on where to crop, not on the crops themselves;
   a detector trained on a few hundred boxes would answer it, and the label text of a slab is nearly
   an OCR problem.
3. **Seed averaging in reporting.** Two seeds differ by up to 2 pp on one fold, so a single run is not
   a measurement: report the mean of two seeds (~0.880 for the current recipe).
4. **A negative class.** The five photos with no legible date cannot teach "there is no date", so the
   verdict can only fail by scoring a wrong window too high - it never knows that no window should be
   high. A hundred real dateless photos would give it that option (erased ones would not: a filled-in
   patch is a smooth-metal artifact, not the "illegible because small, blurred or glared" class that
   actually occurs), and the cache would have to include photos with no positive window at all.
