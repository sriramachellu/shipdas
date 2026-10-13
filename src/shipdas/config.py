from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_H5 = ROOT / "data/raw/dataset_sensor_range_1440_1690_0.h5"
CACHE_DIR = ROOT / "data/cache"
REPORTS_DIR = ROOT / "reports"

N_CHANNELS_RAW = 250
N_BANDS = 100
NOISY_CHANNELS = (59, 60, 61)  # zeroed in the release, the dataset README says to drop them
WINDOW_SECONDS = 10
THRESHOLD_M = 1000.0  # "vessel near" means closest AIS vessel within 1 km
EPS = 1e-30
