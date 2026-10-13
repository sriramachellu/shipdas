"""Pastel, minimal figures for the guide PDF. Writes PNGs to docs/figures/."""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from shipdas import config as C
from shipdas.data import day_folds, load

matplotlib.use("Agg")
OUT = C.ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)
R = C.REPORTS_DIR

INK, MUTED, GRID = "#3d3b45", "#8b8994", "#e9e6ee"
BLUE, PINK, MINT, PEACH, LAV, LEMON = "#a8c8e8", "#f2b8c6", "#b4e0cc", "#f8d2b0", "#cdb8de", "#f6e7a6"
DEEP_BLUE, DEEP_PINK, DEEP_LAV = "#6f9cc9", "#d98aa0", "#9b83b8"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "figure.facecolor": "#fbfaf7", "savefig.facecolor": "#fbfaf7",
    "axes.facecolor": "#fbfaf7",
})
SEQ = LinearSegmentedColormap.from_list("pastel_seq", ["#fbfaf7", LAV, "#7e6aa0"])
DIV = LinearSegmentedColormap.from_list("pastel_div", [DEEP_BLUE, "#fbfaf7", DEEP_PINK])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


ds = load()
m, y = ds.meta, ds.y
near = ds.y_det == 1

# 1. one window as an image
i = int(np.flatnonzero((y < 150) & m["ship_type"].eq("Container Ship").to_numpy())[50])
img = np.asarray(ds.X[i], dtype=np.float32)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.imshow(img.T, aspect="auto", origin="lower", cmap=SEQ)
ax.set(xlabel="channel along the cable (every 10 m)", ylabel="frequency band (4 to 98 Hz)",
       title=f"One 10 s window: container ship {y[i]:.0f} m from the cable")
ax.grid(False)
save(fig, "sample_window.png")

# 2. label distribution
fig, ax = plt.subplots(figsize=(7, 3.4))
bins = np.linspace(0, 9500, 96)
ax.hist(y[near], bins=bins, color=PINK, label="near: vessel within 1 km (32%)")
ax.hist(y[~near], bins=bins, color=BLUE, label="far (68%)")
ax.axvline(1000, color=INK, lw=1, ls=(0, (4, 3)))
ax.set(xlabel="distance to the closest AIS vessel (m)", ylabel="windows",
       title="What we predict: distance to the closest vessel")
ax.legend()
save(fig, "labels.png")

# 3. near minus far
rng = np.random.default_rng(0)


def mean_img(mask):
    idx = np.sort(rng.choice(np.flatnonzero(mask), 2000, replace=False))
    return np.asarray(ds.X[idx], dtype=np.float32).mean(0)


diff = mean_img(near) - mean_img(y > 3000)
fig, ax = plt.subplots(figsize=(7, 3.4))
lim = np.abs(diff).max()
im = ax.imshow(diff.T, aspect="auto", origin="lower", cmap=DIV, vmin=-lim, vmax=lim)
fig.colorbar(im, ax=ax, label="log10 energy, near minus far", shrink=0.85).outline.set_visible(False)
ax.set(xlabel="channel along the cable", ylabel="frequency band",
       title="A nearby ship is louder almost everywhere, most at low and high bands")
ax.grid(False)
save(fig, "near_minus_far.png")

# 4. day-wise cross-validation scheme
days = sorted(set(ds.days))
fig, ax = plt.subplots(figsize=(7, 3.2))
for f in range(len(days)):
    for d in range(len(days)):
        ax.add_patch(plt.Rectangle((d, -f), 0.92, 0.8, color=PINK if d == f else BLUE, lw=0))
ax.set(xlim=(-0.1, 10), ylim=(-9.3, 1), title="Day-wise 10-fold cross-validation")
ax.set_xticks(np.arange(10) + 0.46, [d[5:] for d in days])
ax.set_yticks(-np.arange(10) + 0.4, [f"fold {k + 1}" for k in range(10)])
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.text(10.1, 0.4, "test", color=DEEP_PINK, va="center", fontweight="bold")
ax.text(10.1, -0.6, "train", color=DEEP_BLUE, va="center", fontweight="bold")
save(fig, "cv_scheme.png")

# 5. headline results
systems = ["Paper\nbaseline", "Our replication", "XGBoost\nspatial", "CNN + XGBoost\ncascade"]
wf1 = [89.43, 89.21, 90.33, 91.94]
mae = [171, 158, 128.5, 117]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
cols = [GRID, BLUE, MINT, PINK]
b = axes[0].bar(systems, wf1, color=cols, width=0.62)
axes[0].bar_label(b, fmt="%.2f%%", color=INK, fontsize=10, padding=3)
axes[0].set(ylim=(85, 93), title="Detection: weighted F1 (higher is better)")
b = axes[1].bar(systems, mae, color=cols, width=0.62)
axes[1].bar_label(b, fmt="%.0f m", color=INK, fontsize=10, padding=3)
axes[1].set(ylim=(0, 200), title="Distance error within 1 km (lower is better)")
for a in axes:
    a.tick_params(axis="x", labelsize=9, length=0)
    a.grid(axis="x", visible=False)
save(fig, "headline.png")

# 6. ablation
steps = ["paper\nfeatures", "+ spatial\nfeatures", "+ context\nfeatures", "CNN", "CNN +\n50 s context"]
f1 = [87.58, 88.94, 88.33, 88.81, 90.72]
mae_n = [422, 360, 348, 312, 246]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
ax.plot(steps, f1, color=DEEP_LAV, marker="o", lw=2, ms=7)
for k, v in enumerate(f1):
    ax.annotate(f"{v:.2f}%", (k, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10)
ax.set(ylabel="macro F1 (%)", ylim=(86.5, 92), title="What each step added (XGBoost first, then CNN)")
ax2 = ax.twinx()
ax2.bar(steps, mae_n, color=PEACH, width=0.45, alpha=0.7, zorder=0)
ax2.set(ylabel="MAE within 1 km (m)", ylim=(0, 900))
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
ax.tick_params(axis="x", length=0)
save(fig, "ablation.png")

# 7. per day
fold = pd.read_csv(R / "cascade/folds_cnn_det+xgb_near_reg.csv")
wx = pd.read_csv(C.ROOT / "data/raw/misc/meteorological_daily_summary.csv")
fig, ax = plt.subplots(figsize=(8.5, 3.3))
labels = [d[5:] for d in fold["day"]]
colors = [PINK if d == "2023-06-19" else MINT for d in fold["day"]]
b = ax.bar(labels, fold["recall_near"] * 100, color=colors, width=0.6)
ax.bar_label(b, fmt="%.0f", fontsize=9, padding=2)
ax.set(ylim=(60, 100), ylabel="recall of near vessels (%)",
       title="Per held-out day: the roughest day (06-19, waves 1.3 m) is the hardest")
ax2 = ax.twinx()
ax2.plot(labels, wx["waveHeight__sg__max [m]"], color=DEEP_BLUE, marker="o", lw=1.5)
ax2.set(ylabel="max wave height (m)", ylim=(0, 2))
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax.tick_params(axis="x", length=0)
save(fig, "per_day.png")

# 8. ship types
rec = pd.read_csv(R / "figures/recall_by_ship_type_cascade.csv").sort_values("recall")
fig, ax = plt.subplots(figsize=(7.5, 4))
colors = [PINK if r < 0.8 else MINT for r in rec["recall"]]
b = ax.barh(rec["type"], rec["recall"] * 100, color=colors, height=0.62)
ax.bar_label(b, fmt="%.0f%%", fontsize=9, padding=3)
ax.set(xlim=(0, 110), xlabel="recall within 1 km (%)", title="Small and slow vessels are missed most")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0, labelsize=9)
save(fig, "ship_types.png")

# 9. timeline of a held-out day, 2 hours
oof = np.load(R / "cascade/oof_cascade.npz")
sel = np.flatnonzero(ds.days == "2023-06-20")[4000:4720]
t = m["time"].to_numpy()[sel]
fig, axes = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
axes[0].plot(t, np.clip(y[sel], 0, 5000), color=DEEP_BLUE, lw=1.6, label="AIS (truth)")
axes[0].plot(t, np.clip(oof["dist"][sel], 0, 5000), color=DEEP_PINK, lw=1.2, label="predicted")
axes[0].axhline(1000, color=MUTED, lw=0.8, ls=(0, (4, 3)))
axes[0].set(ylabel="distance (m)", title="Two hours of a day the models never saw (20 June)")
axes[0].legend(loc="upper right", ncol=2)
axes[1].fill_between(t, 0, near[sel], step="mid", color=PINK, alpha=0.5, lw=0, label="vessel within 1 km")
axes[1].plot(t, oof["prob"][sel], color=DEEP_LAV, lw=1.2, label="CNN probability")
axes[1].set(ylabel="P(near)", ylim=(0, 1.05))
axes[1].legend(loc="upper right", ncol=2)
axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
save(fig, "timeline.png")

# 10. distance scatter for detected near vessels
cas = np.load(R / "gbm/oof_cascade_spatial.npz")["dist_near"]
ok = near & (oof["prob"] >= 0.5)
fig, ax = plt.subplots(figsize=(5.2, 4))
ax.scatter(y[ok], cas[ok], s=2, color=DEEP_LAV, alpha=0.15, lw=0)
ax.plot([0, 1000], [0, 1000], color=INK, lw=1, ls=(0, (4, 3)))
ax.set(xlim=(0, 1000), ylim=(0, 1400), xlabel="AIS distance (m)", ylabel="predicted (m)",
       title="Distance for detected vessels")
save(fig, "scatter.png")

# 11. training loss curve from the screening run log
loss = [0.4017, 0.2880, 0.2395, 0.1994, 0.1641, 0.1248, 0.0884, 0.0698]
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(range(1, 9), loss, color=DEEP_PINK, marker="o", lw=2)
ax.set(xlabel="epoch", ylabel="training loss", title="CNN training loss, one fold")
save(fig, "loss.png")

summary = {"sample_window_index": i, "near_fraction": float(near.mean()),
           "folds": [d for d, _, _ in day_folds(ds.days)]}
(OUT / "figures.json").write_text(json.dumps(summary, indent=2))
print("figures written to", OUT)
