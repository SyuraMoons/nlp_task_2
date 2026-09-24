"""Exchange-rate history features -- the ONLY inputs of the baseline model.

They are computed in src/data/make_splits.py (from past returns only); this module
just names them so every model uses the same list.
"""
from src.config import N_LAGS, ROLLING_WINDOWS

MARKET_FEATURES = (
    [f"return_lag{k}" for k in range(1, N_LAGS + 1)]
    + [f"return_{stat}_{w}d" for w in ROLLING_WINDOWS for stat in ("mean", "std")]
    + ["up_days_last5", "calendar_days_since_prev"]
)


def market_features(df):
    return df[MARKET_FEATURES].astype(float)
