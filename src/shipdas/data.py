"""Turn the 15 GB HDF5 release into a small cache and load it back.

X in the release is float64 (74771, 250, 100), bigger than the RAM of a 16 GB laptop.
We read it in chunks, drop the 3 noisy channels, take log10 of the band energies,
subtract one global offset and store float16 (about 3.7 GB).
"""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as C

CACHE_X = C.CACHE_DIR / "logX_f16.npy"
CACHE_META = C.CACHE_DIR / "meta.parquet"
CACHE_INFO = C.CACHE_DIR / "meta.json"
KEEP = np.array([c for c in range(C.N_CHANNELS_RAW) if c not in C.NOISY_CHANNELS])


def decode(arr):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]


def build_cache(h5_path=C.RAW_H5, chunk=512):
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        X = f["X"]
        n = len(X)
        offset = float(np.median(np.log10(X[:chunk][:, KEEP] + C.EPS)))
        out = np.lib.format.open_memmap(CACHE_X, "w+", np.float16, (n, len(KEEP), C.N_BANDS))
        lo, hi = np.inf, -np.inf
        for s in tqdm(range(0, n, chunk), desc="caching"):
            block = np.log10(X[s:s + chunk][:, KEEP] + C.EPS) - offset
            lo, hi = min(lo, block.min()), max(hi, block.max())
            out[s:s + len(block)] = block
        out.flush()

        ship = f["ship_info"]
        meta = pd.DataFrame({
            "time": pd.to_datetime(decode(f["datetimes"][:]), utc=True),
            "y": f["y"][:].astype(float),
            "beam": ship["beam"][:],
            "length": ship["length"][:],
            "ship_type": decode(ship["type"][:]),
        })

    meta["day"] = meta["time"].dt.strftime("%Y-%m-%d")
    # windows are 10 s apart; a bigger gap means the recording stopped
    meta["contiguous_prev"] = meta["time"].diff().dt.total_seconds().eq(C.WINDOW_SECONDS)
    meta.to_parquet(CACHE_META)
    info = {"n": n, "log10_offset": offset, "log_min": float(lo), "log_max": float(hi)}
    CACHE_INFO.write_text(json.dumps(info, indent=2))
    print(info)


def load(in_memory=False):
    meta = pd.read_parquet(CACHE_META)
    y = meta["y"].to_numpy()
    return SimpleNamespace(
        X=np.load(CACHE_X, mmap_mode=None if in_memory else "r"),
        meta=meta,
        info=json.loads(CACHE_INFO.read_text()),
        y=y,
        y_det=(y <= C.THRESHOLD_M).astype(np.int8),
        days=meta["day"].to_numpy(),
    )


def day_folds(days):
    """One fold per recording day: train on 9 days, test on the held-out one."""
    for d in sorted(set(days)):
        test = days == d
        yield d, np.flatnonzero(~test), np.flatnonzero(test)
