"""Collect every run into reports/comparison.csv and recall by ship type."""

import numpy as np
import pandas as pd

from shipdas import config as C
from shipdas.data import load

COLS = ["accuracy", "f1_macro", "f1_near", "f1_far", "precision_near", "recall_near", "auc",
        "mae", "mae_near", "median_ae_near"]


def runs(kind):
    return [p for p in sorted((C.REPORTS_DIR / kind).glob("folds_*.csv")) if "smoke" not in p.stem]


rows = []
for p in runs("gbm"):
    tag = p.stem.removeprefix("folds_")
    for name, g in pd.read_csv(p).groupby("set"):
        rows.append({"model": f"xgb[{tag}:{name}]", "folds": len(g), **g[COLS].mean()})
for p in runs("cnn"):
    g = pd.read_csv(p)
    rows.append({"model": f"cnn[{p.stem.removeprefix('folds_')}]", "folds": len(g), **g[COLS].mean()})
table = pd.DataFrame(rows)
table.to_csv(C.REPORTS_DIR / "comparison.csv", index=False)
print(table.round(4).to_string(index=False))

ds = load()
prob = np.load(C.REPORTS_DIR / "cnn/oof_cnn_c5.npz")["prob"]  # the cascade's detector
near = ds.y_det == 1
hits = pd.DataFrame({"type": ds.meta["ship_type"][near], "hit": prob[near] >= 0.5})
recall = hits.groupby("type")["hit"].agg(n="size", recall="mean")
recall = recall[recall.n >= 200].sort_values("recall")
(C.REPORTS_DIR / "figures").mkdir(exist_ok=True)
recall.to_csv(C.REPORTS_DIR / "figures/recall_by_ship_type_cascade.csv")
print(recall.round(3).to_string())
