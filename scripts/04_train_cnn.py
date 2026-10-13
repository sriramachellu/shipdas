"""CNN with day-wise 10-fold CV.

python scripts/04_train_cnn.py --epochs 8 --context 5 --gain-aug 0.1 --save-model --tag cnn_c5
Add --gpu-data to keep the data on the GPU (frees ~3.7 GB of RAM), --days to run a few folds.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from shipdas import config as C
from shipdas.cnn import DasNet, context_index
from shipdas.data import day_folds, load
from shipdas.metrics import detection_metrics, regression_metrics, restricted_regression_metrics
from shipdas.stores import CpuStore, GpuStore


def band_stats(X, idx, rng, n=4000):
    """Per-band mean and std from a random sample of training windows."""
    a = np.asarray(X[np.sort(rng.choice(idx, n, replace=False))], dtype=np.float32)
    return a.mean(axis=(0, 1)), a.std(axis=(0, 1)) + 1e-3


def batches(idx, size, rng=None):
    if rng is not None:
        idx = rng.permutation(idx)
    for s in range(0, len(idx), size):
        yield idx[s:s + size]


def train_fold(store, ctx, y_det, y, tr, te, args, rng, dev="cuda"):
    mu_np, sd_np = band_stats(store.np, tr, rng)
    mu, sd = torch.tensor(mu_np, device=dev), torch.tensor(sd_np, device=dev)

    def inputs(b, aug=0.0):
        x = store.get(ctx[b].ravel(), dev).view(len(b), ctx.shape[1], *store.shape[1:])
        x = (x - mu) / sd
        if aug:  # random overall gain, so the model does not rely on the absolute level
            x = x + aug * torch.randn(len(b), 1, 1, 1, device=dev)
        return x

    # distance target: log metres fits the 0 to 9 km range, linear metres + L1 fits MAE
    t = np.log1p(y) if args.reg_target == "log" else y.copy()
    use = y < C.THRESHOLD_M if args.reg_near_only else np.ones(len(y), bool)
    tm, ts = t[tr][use[tr]].mean(), t[tr][use[tr]].std()
    yd = torch.tensor(y_det, dtype=torch.float32)
    yt = torch.tensor((t - tm) / ts, dtype=torch.float32)
    w = torch.tensor(use, dtype=torch.float32)
    reg_loss = F.smooth_l1_loss if args.reg_target == "log" else F.l1_loss

    torch.manual_seed(args.seed)
    model = DasNet(ctx.shape[1], args.width, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * -(-len(tr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler()

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for b in batches(tr, args.bs, rng):
            bt = torch.from_numpy(b)
            with torch.autocast("cuda", dtype=torch.float16):
                logit, d = model(inputs(b, args.gain_aug))
            loss = F.binary_cross_entropy_with_logits(logit.float(), yd[bt].to(dev))
            r = reg_loss(d.float(), yt[bt].to(dev), reduction="none") * w[bt].to(dev)
            loss = loss + r.sum() / w[bt].sum().clamp(min=1).to(dev)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
        if args.verbose:
            print(f"    epoch {ep + 1}/{args.epochs} loss={total / len(tr):.4f}", flush=True)

    model.eval()
    probs, dists = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for b in batches(te, 512):
            logit, d = model(inputs(b))
            probs.append(torch.sigmoid(logit.float()).cpu().numpy())
            dists.append(d.float().cpu().numpy())
    dist = np.concatenate(dists) * ts + tm
    dist = np.expm1(dist) if args.reg_target == "log" else np.maximum(dist, 0)
    norm = {"mu": mu_np, "sd": sd_np, "ym": float(tm), "ys": float(ts), "reg_target": args.reg_target}
    return np.concatenate(probs), dist, model, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--context", type=int, default=0, help="past windows stacked as input planes")
    ap.add_argument("--gain-aug", type=float, default=0.0)
    ap.add_argument("--reg-target", choices=["log", "linear"], default="log")
    ap.add_argument("--reg-near-only", action="store_true", help="train distance on y < 1 km only")
    ap.add_argument("--days", nargs="*", help="only run these held-out days")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="cnn")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--gpu-data", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.gpu_data:
        ds = load()
        store = GpuStore(ds.X, ds.info["log_min"], ds.info["log_max"], "cuda")
    else:
        ds = load(in_memory=True)
        store = CpuStore(ds.X)
    ctx = context_index(ds.meta["contiguous_prev"].to_numpy(), args.context)
    y, y_det = ds.y, ds.y_det.astype(np.float32)

    out = C.REPORTS_DIR / "cnn"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    oof_prob, oof_dist = np.full(len(y), np.nan), np.full(len(y), np.nan)
    rows = []
    for day, tr, te in day_folds(ds.days):
        if args.days and day not in args.days:
            continue
        t0 = time.time()
        prob, dist, model, norm = train_fold(store, ctx, y_det, y, tr, te, args, rng)
        oof_prob[te], oof_dist[te] = prob, dist
        near = y[te] < C.THRESHOLD_M
        r = {"day": day,
             **detection_metrics(y_det[te].astype(int), prob),
             **regression_metrics(y[te], dist),
             **restricted_regression_metrics(y[te][near], dist[near])}
        rows.append(r)
        print(f"  {day}: f1_macro={r['f1_macro']:.4f} auc={r['auc']:.4f} "
              f"mae_near={r['mae_near']:.1f} ({time.time() - t0:.0f}s)", flush=True)
        if args.save_model:
            torch.save(model.state_dict(), out / f"model_{args.tag}_{day}.pt")
            np.savez(out / f"norm_{args.tag}_{day}.npz", **norm)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"folds_{args.tag}.csv", index=False)
    np.savez(out / f"oof_{args.tag}.npz", prob=oof_prob, dist=oof_dist)
    mean = df.drop(columns="day").mean().round(4)
    summary = {"args": vars(args), "mean": mean.to_dict()}
    (out / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(mean.to_string())


if __name__ == "__main__":
    main()
