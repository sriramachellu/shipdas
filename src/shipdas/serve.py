"""Replay a held-out day through the CNN trained without it.

python -m shipdas.serve --day 2023-06-20 --tag cnn_c5    then open http://127.0.0.1:8765
"""

import argparse

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from . import config as C
from .cnn import DasNet, context_index
from .data import load

PAGE = C.ROOT / "src/shipdas/static/replay.html"


class Replayer:
    def __init__(self, day, tag, context, device="cpu"):
        self.ds = ds = load()
        self.idx = np.flatnonzero(ds.days == day)
        self.device = device
        self.load_norm(tag, day)
        self.ctx = context_index(ds.meta["contiguous_prev"].to_numpy(), context)
        self.model = DasNet(in_planes=context + 1).to(device).eval()
        weights = C.REPORTS_DIR / f"cnn/model_{tag}_{day}.pt"
        self.model.load_state_dict(torch.load(weights, map_location=device))
        self.prob, self.dist = self.predict()

    def load_norm(self, tag, day):
        path = C.REPORTS_DIR / f"cnn/norm_{tag}_{day}.npz"
        if path.exists():
            n = np.load(path)
            self.mu, self.sd, self.ym, self.ys = n["mu"], n["sd"], float(n["ym"]), float(n["ys"])
            self.target = str(n["reg_target"]) if "reg_target" in n else "log"
            return
        # older runs did not save the stats, so redo them the way training does
        train = np.flatnonzero(self.ds.days != day)
        sub = np.sort(np.random.default_rng(0).choice(train, 4000, replace=False))
        a = np.asarray(self.ds.X[sub], dtype=np.float32)
        self.mu, self.sd = a.mean(axis=(0, 1)), a.std(axis=(0, 1)) + 1e-3
        ylog = np.log1p(self.ds.y[train])
        self.ym, self.ys, self.target = ylog.mean(), ylog.std(), "log"

    @torch.no_grad()
    def predict(self):
        probs, dists = [], []
        for s in range(0, len(self.idx), 256):
            b = self.idx[s:s + 256]
            x = np.asarray(self.ds.X[self.ctx[b].ravel()], dtype=np.float32)
            x = x.reshape(len(b), self.ctx.shape[1], *x.shape[1:])
            logit, d = self.model(torch.from_numpy((x - self.mu) / self.sd).to(self.device))
            d = d.cpu().numpy() * self.ys + self.ym
            probs.append(torch.sigmoid(logit).cpu().numpy())
            dists.append(np.expm1(d) if self.target == "log" else np.maximum(d, 0))
        return np.concatenate(probs), np.concatenate(dists)

    def timeline(self):
        m = self.ds.meta.iloc[self.idx]
        return {"time": m["time"].dt.strftime("%H:%M:%S").tolist(),
                "y": m["y"].round(1).tolist(),
                "prob": self.prob.round(4).tolist(),
                "dist": self.dist.round(1).tolist()}

    def window(self, i: int):
        if not 0 <= i < len(self.idx):
            raise HTTPException(404, "index out of range")
        j = self.idx[i]
        m = self.ds.meta.iloc[j]
        img = (np.asarray(self.ds.X[j], dtype=np.float32) - self.mu) / self.sd
        return {"time": str(m["time"]), "y": float(m["y"]), "ship_type": m["ship_type"],
                "length": float(m["length"]), "prob": float(self.prob[i]),
                "dist": float(self.dist[i]), "heatmap": img.round(2).tolist()}


def make_app(rep, day):
    app = FastAPI(title="shipdas replay")
    app.get("/", response_class=HTMLResponse)(
        lambda: PAGE.read_text(encoding="utf-8").replace("__DAY__", day))
    app.get("/api/timeline")(rep.timeline)
    app.get("/api/window/{i}")(rep.window)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2023-06-20")
    ap.add_argument("--tag", default="cnn_c5")
    ap.add_argument("--context", type=int, default=5)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    rep = Replayer(args.day, args.tag, args.context, args.device)
    print(f"{args.day}: {len(rep.idx)} windows")
    uvicorn.run(make_app(rep, args.day), host="127.0.0.1", port=args.port)
