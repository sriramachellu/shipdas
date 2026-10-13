# shipdas

Detect vessels near a submarine fibre-optic cable, and estimate how far away they are, using
Distributed Acoustic Sensing (DAS) data from the cable itself.

Data: the public [Marlinks-NS dataset](https://doi.org/10.5281/zenodo.15611778) (CC BY 4.0):
74,771 ten-second windows, 250 channels x 100 frequency bands, recorded on a 28 km cable off
Zeebrugge, Belgium, 16 to 25 June 2023. Labels are the distance to the closest AIS vessel.

**New here? Read [docs/shipdas_guide.pdf](docs/shipdas_guide.pdf).** It walks through the
dataset, every step of the code, why each model was chosen, and the results.

## Results

Mean over 10 held-out days (one day per fold):

| System | Weighted F1 | Macro F1 | Distance MAE |
| --- | --- | --- | --- |
| Paper baseline (XGBoost) | 89.43% | - | 171 m (within-1-km protocol) |
| XGBoost, spatial features | 90.33% | 88.94% | 128.5 m (within-1-km protocol) |
| CNN detector + XGBoost distance | 91.94% | 90.72% | 117 m on detected vessels |

More detail, ablations and failure cases: [reports/RESULTS.md](reports/RESULTS.md).

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

Download `dataset_sensor_range_1440_1690_0.h5` (14.1 GB, MD5 `407ad62bb8eb02f472cda329d0f31750`)
and `misc.zip` from Zenodo into `data/raw/`, and unzip `misc.zip` there.

## Run

| Step | Command | What it does |
| --- | --- | --- |
| 1 | `python scripts/01_build_cache.py` | 15 GB float64 HDF5 to a 3.7 GB float16 cache |
| 2 | `python scripts/02_eda.py` | label stats and spectra plots |
| 3 | `python scripts/03_train_gbm.py --sets paper spatial --tag paperhp` | XGBoost, 10 folds |
| 4 | `python scripts/03_train_gbm.py --sets spatial --tag cascade` | distance model for the cascade |
| 5 | `python scripts/04_train_cnn.py --epochs 8 --context 5 --gain-aug 0.1 --save-model --tag cnn_c5` | CNN, 10 folds, about 95 min |
| 6 | `python scripts/06_cascade.py` | CNN detects, XGBoost measures |
| 7 | `python scripts/05_report.py` | comparison table, recall by ship type |
| demo | `python -m shipdas.serve --day 2023-06-20` | replay a held-out day at http://127.0.0.1:8765 |

On a 16 GB machine add `--gpu-data` to step 5: the dataset then sits on the GPU as uint8
(1.85 GB) instead of in RAM.

To rebuild the guide: `python docs/make_figures.py` then `python docs/build_guide.py`.

## Layout

```
src/shipdas/   config, data cache, features, metrics, CNN, data stores, demo server
scripts/       numbered pipeline steps
tests/         unit tests (python -m pytest -q)
reports/       results, per-fold CSVs, predictions
docs/          the PDF guide and the scripts that build it
```

## Citation

Ramirez-Torres et al., Marlinks-NS DAS Dataset, Zenodo, doi:10.5281/zenodo.15611778, and
"Vessel Detection and Localization Using Distributed Acoustic Sensing in Submarine Optical
Fiber Cables", IEEE JSTARS 2026, doi:10.1109/JSTARS.2026.3716768.
