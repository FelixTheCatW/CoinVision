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
