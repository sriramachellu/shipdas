import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import THRESHOLD_M


def detection_metrics(y_true, prob, thr=0.5):
    """y_true is 1 when a vessel is within 1 km."""
    pred = (prob >= thr).astype(int)

    def f1(**kw):
        return f1_score(y_true, pred, zero_division=0, **kw)

    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1_macro": f1(average="macro"),
        "f1_weighted": f1(average="weighted"),  # what the paper calls global F1
        "f1_near": f1(pos_label=1),
        "f1_far": f1(pos_label=0),
        "precision_near": precision_score(y_true, pred, zero_division=0),
        "recall_near": recall_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob) if len(set(y_true)) == 2 else np.nan,
    }


def regression_metrics(y_true, y_pred):
    err = np.abs(y_true - y_pred)
    near = y_true < THRESHOLD_M
    return {"mae": err.mean(), "mae_near": err[near].mean(), "median_ae_near": np.median(err[near])}


def restricted_regression_metrics(y_true, y_pred):
    """Paper protocol: regressor trained and tested only on windows within 1 km."""
    err = np.abs(y_true - y_pred)
    return {"mae_restricted": err.mean(), "median_ae_restricted": np.median(err)}
