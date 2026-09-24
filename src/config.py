"""Central configuration for Task 2. Every script imports from here, so a design
choice (split ratios, lags, which NLP features are on) is changed in ONE place.

Run all scripts from the repository root, e.g. ``python -m src.train``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LEXICON_DIR = DATA_DIR / "lexicon"

# Task 1 output: one row per Bank Indonesia JISDOR trading day, with the news that
# was aligned to that day (all of it published BEFORE that day's 10:00 WIB fixing).
SOURCE_DATASET = DATA_DIR / "raw" / "final_dataset.csv"

SEED = 42

# ---------------------------------------------------------------------------
# Task formulation
# ---------------------------------------------------------------------------
# Row t = trading day t. Target: did the JISDOR rate go UP at fixing t vs. t-1?
#   y_t = 1 if rate_t > rate_{t-1}  (USD up / IDR weaker)
#   y_t = 0 if rate_t < rate_{t-1}  (USD down / IDR stronger)
# Flat fixings (rate unchanged, ~1% of days) are dropped.
#
# Information available when predicting y_t (i.e. just before 10:00 WIB on day t):
#   * exchange-rate history up to and including t-1  -> lag features below
#   * all news aligned to day t by Task 1 (published after fixing t-1 and before
#     fixing t)                                        -> NLP features
# Nothing from day t's fixing itself is ever used as a feature.
N_LAGS = 5            # return_lag1 .. return_lag5
ROLLING_WINDOWS = (5, 20)

# ---------------------------------------------------------------------------
# Chronological split (no shuffling -- avoids look-ahead leakage)
# ---------------------------------------------------------------------------
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------------------
# NLP features (src/features/nlp_features.py). Person B turns these on once the
# corresponding functions are implemented.
# ---------------------------------------------------------------------------
NLP_CONFIG = {
    "use_lexicon": False,         # InSet Indonesian sentiment lexicon on headlines
    "use_tfidf": False,           # TF-IDF on headlines, reduced with TruncatedSVD
    "tfidf_max_features": 5000,
    "tfidf_min_df": 5,
    "svd_components": 20,
}
