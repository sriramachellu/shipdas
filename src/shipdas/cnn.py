"""Small CNN that looks at the (channel x band) image of a window.

Two outputs: a logit for "vessel within 1 km" and a standardised distance.
With context k, the k previous windows are stacked as extra input planes.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def context_index(contiguous_prev, k):
    """Row t holds the indices [t-k, ..., t-1, t]. At the start of the recording or after a
    gap we repeat the first available window, so we never look across a gap or into the future."""
    n = len(contiguous_prev)
    idx = np.empty((n, k + 1), dtype=np.int64)
    idx[:, k] = np.arange(n)
    for j in range(k - 1, -1, -1):
        nxt = idx[:, j + 1]
        idx[:, j] = np.where(contiguous_prev[nxt] & (nxt > 0), nxt - 1, nxt)
    return idx


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        return F.max_pool2d(x, 2)


class DasNet(nn.Module):
    def __init__(self, in_planes=1, width=32, dropout=0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            Block(in_planes, w), Block(w, 2 * w), Block(2 * w, 4 * w), Block(4 * w, 4 * w))
        self.head = nn.Sequential(
            nn.Linear(8 * w, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x):
        h = self.features(x)
        h = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)  # average + max pooling
        out = self.head(h)
        return out[:, 0], out[:, 1]
