"""Evaluation metrics for the binary UP/DOWN task.

Why these metrics:
  * accuracy (= directional accuracy): the headline number, but misleading alone --
    USD/IDR goes up on ~55% of days, so "always predict UP" already scores ~0.55.
  * macro-F1: averages F1 of the UP and DOWN classes, so a model that ignores the
    minority class is penalised.
  * MCC (Matthews correlation): single balanced score in [-1, 1]; 0 = no better
    than chance / a constant guess. Our main model-selection metric.
  * ROC-AUC: ranking quality of the predicted probabilities (only for models that
    output probabilities).
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score


def evaluate(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": np.nan,
        "pred_up_rate": float(np.mean(y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    return out
