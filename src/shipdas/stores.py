"""Where the training data lives: float16 in RAM, or uint8 on the GPU."""

import numpy as np
import torch


class CpuStore:
    def __init__(self, X):
        self.np, self.shape = X, X.shape
        self.t = torch.from_numpy(np.ascontiguousarray(X))  # no copy

    def get(self, idx, device):
        return self.t[torch.from_numpy(idx)].to(device, non_blocking=True).float()


class GpuStore:
    """The whole dataset as uint8 on the GPU (1.85 GB). Steps are about 0.05 decades,
    much smaller than the near/far differences we care about."""

    def __init__(self, X, lo, hi, device, chunk=4096):
        self.np, self.shape = X, X.shape
        self.lo, self.scale = lo, (hi - lo) / 255
        self.t = torch.empty(X.shape, dtype=torch.uint8, device=device)
        for s in range(0, len(X), chunk):
            a = np.asarray(X[s:s + chunk], dtype=np.float32)
            q = np.clip(np.rint((a - lo) / self.scale), 0, 255).astype(np.uint8)
            self.t[s:s + len(a)] = torch.from_numpy(q).to(device)

    def get(self, idx, device):
        return self.t[torch.from_numpy(idx).to(device)].float() * self.scale + self.lo
