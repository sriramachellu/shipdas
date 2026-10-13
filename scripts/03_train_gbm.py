"""XGBoost with day-wise 10-fold CV.

For every held-out day we fit three models on the other nine days:
  clf     vessel within 1 km or not
  reg     distance, trained on all windows
  reg_n   distance, trained only on windows within 1 km (the paper's protocol)

python scripts/03_train_gbm.py --sets paper spatial --tag mytag [--device cpu]
"""

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from shipdas import config as C
from shipdas.data import day_folds, load
from shipdas.features import assemble, context_features, window_features
from shipdas.metrics import detection_metrics, regression_metrics, restricted_regression_metrics

FEATURES = C.CACHE_DIR / "gbm_features.npz"
PARAMS = {"learning_rate": 0.05, "max_depth": 10, "n_estimators": 500}  # same as the paper


def get_features(ds):
    if not FEATURES.exists():
        f = window_features(ds.X)
        f.update(context_features(f["mean"], ds.meta))
        np.savez(FEATURES, **f)
    return dict(np.load(FEATURES))


def run(name, feats, ds, device):
    X, y, y_det = assemble(feats, name), ds.y, ds.y_det
    prob, dist, dist_near = (np.zeros(len(y)) for _ in range(3))
    rows = []
    for day, tr, te in day_folds(ds.days):
        kw = dict(PARAMS, tree_method="hist", device=device)
        clf = xgb.XGBClassifier(**kw).fit(X[tr], y_det[tr])
        reg = xgb.XGBRegressor(**kw).fit(X[tr], y[tr])
        trn = tr[y[tr] < C.THRESHOLD_M]
        reg_n = xgb.XGBRegressor(**kw).fit(X[trn], y[trn])

        prob[te] = clf.predict_proba(X[te])[:, 1]
        dist[te] = reg.predict(X[te])
        dist_near[te] = reg_n.predict(X[te])  # every test window, the cascade needs them all

        near = y[te] < C.THRESHOLD_M
        r = {"set": name, "day": day,
             **detection_metrics(y_det[te], prob[te]),
             **regression_metrics(y[te], dist[te]),
             **restricted_regression_metrics(y[te][near], dist_near[te][near])}
        rows.append(r)
        print(f"  {name} {day}: f1_macro={r['f1_macro']:.4f} mae_near={r['mae_near']:.1f} "
              f"mae_restricted={r['mae_restricted']:.1f}", flush=True)
    return rows, prob, dist, dist_near


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["paper", "spatial", "spatial_ctx"])
    ap.add_argument("--tag", default="gbm")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = load()
    feats = get_features(ds)
    out = C.REPORTS_DIR / "gbm"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.sets:
        t0 = time.time()
        r, prob, dist, dist_near = run(name, feats, ds, args.device)
        rows += r
        np.savez(out / f"oof_{args.tag}_{name}.npz", prob=prob, dist=dist, dist_near=dist_near)
        print(f"{name} done in {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"folds_{args.tag}.csv", index=False)
    summary = df.drop(columns="day").groupby("set").mean().round(4)
    summary.to_csv(out / f"summary_{args.tag}.csv")
    print(summary.to_string())
