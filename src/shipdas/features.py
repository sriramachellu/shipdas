"""Hand-made features for XGBoost, one row per 10 s window.

L is the cached log10 energy of one window, shape (247 channels, 100 bands).

mean   log10 of the channel-averaged linear energy, per band (the paper's input)
std    spread across channels, per band
max    loudest channel, per band
seg    13 cable segments x 5 frequency groups, a coarse picture of where along the cable it is loud
delta  change in `mean` since the previous window
past   average `mean` over the last 6 windows (1 minute)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

N_SEG, N_FGROUP, PAST = 13, 5, 6

FEATURE_SETS = {
    "paper": ["mean"],
    "spatial": ["mean", "std", "max", "seg"],
    "spatial_ctx": ["mean", "std", "max", "seg", "delta", "past"],
}


def block_features(L):
    L = L.astype(np.float32)
    top = L.max(axis=1)
    mean = top + np.log10(np.mean(10.0 ** (L - top[:, None]), axis=1))  # stable log-mean-exp
    ch = np.linspace(0, L.shape[1], N_SEG + 1).astype(int)
    fb = np.linspace(0, L.shape[2], N_FGROUP + 1).astype(int)
    seg = [L[:, ch[i]:ch[i + 1], fb[j]:fb[j + 1]].mean(axis=(1, 2))
           for i in range(N_SEG) for j in range(N_FGROUP)]
    return {"mean": mean, "std": L.std(axis=1), "max": top, "seg": np.stack(seg, axis=1)}


def window_features(X, chunk=2048):
    parts = [block_features(np.asarray(X[s:s + chunk])) for s in tqdm(range(0, len(X), chunk))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def context_features(mean, meta):
    """Only looks backwards, and restarts after every gap in the recording."""
    run = np.cumsum(~meta["contiguous_prev"].to_numpy())
    g = pd.DataFrame(mean).groupby(run)
    prev = g.shift(1).to_numpy()
    delta = np.nan_to_num(mean - prev)
    past = g.rolling(PAST, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    return {"delta": delta.astype(np.float32), "past": past.to_numpy(np.float32)}


def assemble(feats, name):
    return np.concatenate([feats[k] for k in FEATURE_SETS[name]], axis=1).astype(np.float32)
