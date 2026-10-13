"""Quick look at the labels and at what a passing vessel looks like on the cable."""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from shipdas import config as C
from shipdas.data import load

matplotlib.use("Agg")
OUT = C.REPORTS_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)

ds = load()
m, y = ds.meta, ds.meta["y"]
near = y <= C.THRESHOLD_M

summary = {
    "n": len(m),
    "near_fraction": near.mean(),
    "y_quantiles_m": y.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).round(1).to_dict(),
    "per_day": m.assign(near=near).groupby("day").agg(n=("y", "size"), near=("near", "mean")).to_dict("index"),
    "gaps": int((~m["contiguous_prev"]).sum()),
    "top_ship_types_near": m.loc[near, "ship_type"].value_counts().head(10).to_dict(),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(y.clip(upper=10_000), bins=100)
ax[0].axvline(1000, color="k", ls="--")
ax[0].set(xlabel="closest vessel distance (m)", ylabel="windows")
ax[1].plot(m["time"], y.clip(upper=10_000), lw=0.3)
ax[1].set(ylabel="distance (m)")
fig.tight_layout()
fig.savefig(OUT / "labels.png", dpi=130)

# average spectrum along the cable with a vessel close by vs far away
rng = np.random.default_rng(0)


def pick(mask):
    return np.sort(rng.choice(np.flatnonzero(mask), 2000, replace=False))


near_img = np.asarray(ds.X[pick(near)], dtype=np.float32).mean(0)
far_img = np.asarray(ds.X[pick(y > 3000)], dtype=np.float32).mean(0)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, img, title, cmap in [(ax[0], near_img, "near (<=1 km)", "magma"),
                            (ax[1], far_img, "far (>3 km)", "magma"),
                            (ax[2], near_img - far_img, "near - far", "RdBu_r")]:
    fig.colorbar(a.imshow(img.T, aspect="auto", origin="lower", cmap=cmap), ax=a)
    a.set(title=title, xlabel="channel", ylabel="band (4-98 Hz)")
fig.tight_layout()
fig.savefig(OUT / "spectra_near_far.png", dpi=130)
