import numpy as np
import pandas as pd
import torch

from shipdas.cnn import DasNet, context_index
from shipdas.data import day_folds
from shipdas.features import block_features, context_features
from shipdas.metrics import detection_metrics, regression_metrics
from shipdas.stores import CpuStore, GpuStore


def test_context_never_looks_ahead_or_across_gaps():
    idx = context_index(np.array([False, True, True, False, True]), 2)
    assert idx.tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 2], [3, 3, 3], [3, 3, 4]]


def test_day_folds_hold_out_whole_days():
    days = np.array(["a", "a", "b", "c", "b"])
    for d, tr, te in day_folds(days):
        assert set(days[te]) == {d} and d not in set(days[tr])
        assert len(tr) + len(te) == len(days)


def test_mean_feature_is_log_of_linear_mean():
    L = np.random.default_rng(0).normal(size=(3, 247, 100)).astype(np.float32)
    f = block_features(L)
    expected = np.log10(np.mean(10.0 ** L.astype(np.float64), axis=1))
    np.testing.assert_allclose(f["mean"], expected, atol=1e-4)
    assert f["seg"].shape == (3, 65)


def test_context_features_only_use_the_past():
    mean = np.arange(10, dtype=np.float32).reshape(5, 2)
    c = context_features(mean, pd.DataFrame({"contiguous_prev": [False, True, True, False, True]}))
    assert c["delta"][0].tolist() == [0, 0] and c["delta"][2].tolist() == [2, 2]
    assert c["delta"][3].tolist() == [0, 0]
    np.testing.assert_allclose(c["past"][2], mean[:3].mean(0))
    np.testing.assert_allclose(c["past"][3], mean[3])


def test_metrics():
    m = detection_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.4, 0.2]))
    assert m["accuracy"] == 0.75 and m["recall_near"] == 0.5
    r = regression_metrics(np.array([100.0, 5000.0]), np.array([200.0, 4000.0]))
    assert r["mae"] == 550 and r["mae_near"] == 100


def test_model_shapes():
    logit, dist = DasNet(in_planes=3, width=8)(torch.zeros(2, 3, 247, 100))
    assert logit.shape == dist.shape == (2,)


def test_uint8_store_error_is_at_most_half_a_step():
    X = np.random.default_rng(1).uniform(-3, 10, size=(50, 7, 5)).astype(np.float16)
    g, c = GpuStore(X, -3.0, 10.0, "cpu", chunk=16), CpuStore(X)
    idx = np.array([0, 3, 49, 3])
    assert (g.get(idx, "cpu") - c.get(idx, "cpu")).abs().max() <= g.scale / 2 + 1e-2
