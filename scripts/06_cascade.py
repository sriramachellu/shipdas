"""End to end: the CNN decides "vessel within 1 km?", XGBoost estimates how far.

The paper scores its distance model only on windows already known to be within 1 km.
A real system has to detect first, so here distance is only produced for flagged windows.
"""

import json

import numpy as np
import pandas as pd

from shipdas import config as C
from shipdas.data import day_folds, load
from shipdas.metrics import detection_metrics

ds = load()
prob = np.load(C.REPORTS_DIR / "cnn/oof_cnn_c5.npz")["prob"]
dist = np.load(C.REPORTS_DIR / "gbm/oof_cascade_spatial.npz")["dist_near"]

rows = []
for day, _, te in day_folds(ds.days):
    p, d, y, near = prob[te], dist[te], ds.y[te], ds.y_det[te] == 1
    flag = p >= 0.5
    err = np.abs(d - y)
    rows.append({
        "day": day,
        **detection_metrics(ds.y_det[te], p),
        "mae_detected_near": err[flag & near].mean(),  # real vessels we caught
        "median_ae_detected_near": np.median(err[flag & near]),
        "mae_all_flagged": err[flag].mean(),  # includes false alarms
        "mae_paper_protocol": err[near].mean(),
        "share_near_detected": (flag & near).sum() / near.sum(),
    })

df = pd.DataFrame(rows)
out = C.REPORTS_DIR / "cascade"
out.mkdir(exist_ok=True)
df.to_csv(out / "folds_cnn_det+xgb_near_reg.csv", index=False)
mean = df.drop(columns="day").mean().round(4)
(out / "summary_cnn_det+xgb_near_reg.json").write_text(json.dumps(mean.to_dict(), indent=2))
np.savez(out / "oof_cascade.npz", prob=prob,
         dist=np.where(prob >= 0.5, dist, np.load(C.REPORTS_DIR / "cnn/oof_cnn_c5.npz")["dist"]))
print(df.round(3).to_string(index=False))
print(mean.to_string())
