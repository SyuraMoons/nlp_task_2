"""Naive reference baselines -- any real model has to beat these.

MajorityClass : always predicts the most common class seen in training
                ("USD always goes up").
Persistence   : predicts that today moves the same way as yesterday
                (y_t = direction of return_{t-1}); the classic random-walk-with-
                momentum baseline for exchange rates.
"""
import numpy as np


class MajorityClass:
    def fit(self, X, y):
        self.p_up_ = float(np.mean(y))
        self.label_ = int(self.p_up_ >= 0.5)
        return self

    def predict(self, X):
        return np.full(len(X), self.label_)

    def predict_proba(self, X):
        p = np.full(len(X), self.p_up_)
        return np.column_stack([1 - p, p])


class Persistence:
    """Needs the column `return_lag1` in X."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (X["return_lag1"].to_numpy() > 0).astype(int)

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.column_stack([1 - p, p])
