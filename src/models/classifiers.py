"""Trainable classifiers and their (small) hyper-parameter grids.

The SAME two model families are used for the baseline (exchange-rate history only)
and the combined model (history + NLP features), so any difference in results
comes from the features, not from a different algorithm.

  logreg  : L2-regularised logistic regression on standardised features --
            simple, interpretable coefficients, hard to overfit.
  xgboost : gradient-boosted trees -- captures non-linear interactions
            (e.g. "negative news only matters when volatility is high").
"""
from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import SEED


def make_logreg(C=1.0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEED),
    )


def make_xgboost(max_depth=3, n_estimators=200, learning_rate=0.05):
    return XGBClassifier(
        max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric="logloss", random_state=SEED, n_jobs=1,
    )


MODELS = {
    "logreg": (make_logreg, {"C": [0.01, 0.1, 1.0, 10.0]}),
    "xgboost": (make_xgboost, {"max_depth": [2, 3], "n_estimators": [100, 300],
                               "learning_rate": [0.03, 0.1]}),
}


def param_grid(grid):
    keys = list(grid)
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))
