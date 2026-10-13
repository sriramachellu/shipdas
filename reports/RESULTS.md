# Results: vessel detection and distance estimation from submarine-cable DAS

Dataset: [Marlinks-NS](https://doi.org/10.5281/zenodo.15611778) (74,771 ten-second windows, 247
usable channels x 100 bands, North Sea, 16-25 June 2023). Protocol: day-wise 10-fold
cross-validation; each fold holds out one whole day, and metrics are the mean over the 10 folds.
Detection label: closest AIS vessel within 1,000 m (32.4% of windows).

## Headline

A cascade of a **CNN detector** and an **XGBoost within-1-km distance regressor** beats the
paper's XGBoost baseline on both tasks:

| System | Accuracy | Weighted F1 | Macro F1 | AUC | Distance MAE |
| --- | --- | --- | --- | --- | --- |
| Paper baseline (XGBoost, 250-channel mean) | 89.58% | 89.43% | - | - | 171 m (paper protocol) |
| Our replication of that baseline | 89.35% | 89.21% | 87.58% | 0.943 | 158 m (paper protocol) |
| XGBoost, spatial features | 90.45% | 90.33% | 88.94% | 0.953 | **128.5 m** (paper protocol) |
| **CNN detector + XGBoost distance (cascade)** | **92.02%** | **91.94%** | **90.72%** | **0.963** | **117 m** on detected vessels |

- "Weighted F1" is what the paper calls global F1 (its class F1s, 83.10% and 92.47%, average to
  89.4% only when weighted by class support).
- "Paper protocol" MAE: a regressor trained and tested only on windows with a vessel within
  1 km. It assumes we already know which windows those are.
- Cascade MAE is end to end: the CNN flags windows (P >= 0.5) and the regressor estimates
  distance only for those. 117 m (median 85 m) is over true-near windows that were detected
  (84.6% of them). Over every flagged window, false alarms included, MAE is 174 m. Detected
  vessels are somewhat easier than missed ones, so compare 128.5 m, not 117 m, with the paper's 171 m.

## What each change bought

| Step | Change | Macro F1 | Distance MAE, near windows* |
| --- | --- | --- | --- |
| 1 | Paper features: log of channel-averaged band energy (100 features) | 87.58% | 422 m |
| 2 | + spread, max and 13 x 5 along-cable segment features (365 features) | 88.94% | 360 m |
| 3 | + causal 1-minute context features (565 features) | 88.33% | 348 m |
| 4 | CNN on the full 247 x 100 channel-band image | 88.81%** | 312 m** |
| 5 | CNN + 5 past windows (50 s causal context) + gain augmentation | **90.72%** | **246 m** |

\* Model trained on all distances, error measured on windows with y < 1 km.
\*\* Two held-out days only (16 and 20 June), the screening setup for steps 4-5.

Context helped the CNN (+0.9 F1 points, -38 m on the screening days) but not XGBoost.

## Things that did not work

- **CNN distance head under the paper protocol.** Trained only on near windows, a log-distance
  head reached 206 m and a linear-distance head with L1 loss 152 m on 16 June, against 129 m for
  XGBoost on spatial features. XGBoost stays the distance model in the cascade.
- **Hand-made temporal context for XGBoost** (step 3) slightly lowered macro F1.

## Where it fails

Recall for vessels within 1 km by AIS ship type (cascade, all folds; selected types with >= 200
near windows, full list in `reports/figures/recall_by_ship_type_cascade.csv`):

| Ship type | Near windows | Recall |
| --- | --- | --- |
| Fishing vessel | 359 | 36% |
| General Cargo Ship | 1,719 | 66% |
| Hopper Dredger | 2,582 | 71% |
| Refrigerated Cargo Ship | 231 | 79% |
| Bulk Carrier | 1,618 | 88% |
| Ro-Ro Cargo Ship | 3,777 | 91% |
| Chemical/Oil Products Tanker | 3,712 | 91% |
| Container Ship | 4,316 | 93% |
| LNG Tanker | 226 | 98% |

Small and slow vessels are missed most often, which matters for cable protection: fishing gear
and dredgers are real cable threats.

Per-day performance (cascade):

| Day | Macro F1 | Recall (near) | MAE, detected near | Max wave (m) | Max wind (m/s) |
| --- | --- | --- | --- | --- | --- |
| 06-16 | 90.0% | 82.1% | 104 m | 0.56 | 5.1 |
| 06-17 | 91.8% | 85.6% | 119 m | 0.60 | 4.7 |
| 06-18 | 90.8% | 87.2% | 121 m | 0.54 | 7.4 |
| 06-19 | **87.5%** | **73.2%** | 122 m | **1.31** | **9.5** |
| 06-20 | 89.4% | 79.1% | 121 m | 0.60 | 5.9 |
| 06-21 | 92.6% | 86.9% | 122 m | 0.90 | 7.5 |
| 06-22 | 89.2% | 87.4% | 129 m | 0.69 | 3.9 |
| 06-23 | 91.0% | 83.9% | 109 m | 0.88 | 6.3 |
| 06-24 | 91.6% | 88.9% | 122 m | 0.90 | 6.5 |
| 06-25 | 93.3% | 91.8% | 99 m | 0.68 | 7.6 |

The hardest day, 19 June, was also the roughest (highest waves, wind and current). Across all 10
days the link is weak (Spearman rho about -0.5 between max current speed and recall), and 10 days
are too few to call it a finding.

## Figures

- `reports/eda/spectra_near_far.png`: mean band energy along the cable, near vs far windows.
- `reports/figures/timeline_2023-06-20_cascade.png`: a held-out day, AIS vs predicted.
- `reports/figures/pr_and_distance_cascade.png`: precision-recall curve and distance scatter.
  In the scatter, points above 1 km are missed detections that fall back to the CNN's
  all-distance estimate.

## Caveats

- Model choices (context length, augmentation) were screened on two of the ten days, which are
  also in the final CV average. That is mild selection bias; the gain is consistent on the other
  eight days.
- Labels come from AIS: vessels without AIS are absent, and positions are interpolated.
- One 2.5 km cable segment, one sea, ten summer days. Cross-site generalisation is untested.
- The reported CNN (`cnn_c5`) read float16 data from RAM. The `--gpu-data` option keeps the
  dataset on the GPU as uint8 instead (about 0.05-decade steps, frees ~3.7 GB of RAM). On the two
  screening days it matched float16 on detection (89.8% vs 89.7% macro F1) and was 8 m worse on
  distance (282 m vs 275 m).

## Reproduce

```bash
python scripts/01_build_cache.py
python scripts/03_train_gbm.py --sets paper spatial spatial_ctx --tag paperhp
python scripts/03_train_gbm.py --sets spatial --tag cascade
python scripts/04_train_cnn.py --epochs 8 --context 5 --gain-aug 0.1 --save-model --tag cnn_c5
# add --gpu-data to keep the data on the GPU and free RAM
python scripts/06_cascade.py
python -m shipdas.serve --day 2023-06-20 --tag cnn_c5 --context 5   # replay demo on :8765
```

Hardware used: RTX 3060 Laptop (6 GB), 16 GB RAM. CNN: about 10 min per fold; XGBoost on GPU:
about 15 min for 10 folds of one feature set.
