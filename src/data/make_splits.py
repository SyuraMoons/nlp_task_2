"""Build the modelling table from the Task 1 dataset and split it chronologically
into train / validation / test.

Steps
  1. Sort Task 1's final_dataset.csv by date (one row per JISDOR trading day).
  2. Target: y_t = 1 if the JISDOR rate went up at fixing t vs. t-1, else 0.
     Flat fixings are dropped.
  3. Exchange-rate history features, built only from PAST returns (shifted by >= 1
     day), so row t never sees rate_t:
        return_lag1..N, rolling mean / std of past returns, calendar gap.
  4. Chronological 70 / 15 / 15 split -- the oldest days train, the newest days
     test. No shuffling, so the model is always evaluated on the future.

Outputs (data/):
  train.csv, val.csv, test.csv  -- same columns, disjoint consecutive date ranges
  split_info.json               -- date range, size and class balance per split

Run with: python -m src.data.make_splits
"""
import json

import numpy as np
import pandas as pd

from src.config import DATA_DIR, N_LAGS, ROLLING_WINDOWS, SOURCE_DATASET, SPLIT_RATIOS

# Columns carried over from Task 1 that the NLP feature extractor needs.
NEWS_COLUMNS = ["n_articles", "n_bisnis", "n_kontan", "n_with_body",
                "n_non_trading_day_articles", "gdelt_tone_mean", "titles"]


def build_table(raw):
    df = raw.sort_values("date").reset_index(drop=True)
    ret = np.log(df["jisdor_rate"]).diff()          # return at fixing t (target source)

    out = pd.DataFrame({"date": df["date"]})
    out["prev_rate"] = df["jisdor_rate"].shift(1)    # last rate known before fixing t
    out["target_return"] = ret                       # kept for reference / regression, NOT a feature
    out["target"] = (ret > 0).astype(int)

    # --- exchange-rate history (all shifted: only information up to t-1) ---
    for k in range(1, N_LAGS + 1):
        out[f"return_lag{k}"] = ret.shift(k)
    for w in ROLLING_WINDOWS:
        past = ret.shift(1).rolling(w)
        out[f"return_mean_{w}d"] = past.mean()
        out[f"return_std_{w}d"] = past.std()
    out["up_days_last5"] = (ret.shift(1) > 0).astype(float).rolling(5).sum()
    # Known in advance: how many calendar days since the last fixing (3 on Mondays).
    out["calendar_days_since_prev"] = df["calendar_days_since_prev"]

    for col in NEWS_COLUMNS:
        out[col] = df[col]
    out["titles"] = out["titles"].fillna("")

    # Drop the warm-up rows (not enough history) and flat fixings.
    out = out.dropna(subset=[c for c in out.columns if c.startswith("return_")] + ["target_return"])
    out = out[out["target_return"] != 0]
    return out.reset_index(drop=True)


def chronological_split(df):
    n = len(df)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    return {
        "train": df.iloc[:n_train],
        "val": df.iloc[n_train:n_train + n_val],
        "test": df.iloc[n_train + n_val:],
    }


def main():
    raw = pd.read_csv(SOURCE_DATASET)
    table = build_table(raw)
    splits = chronological_split(table)

    info = {}
    for name, part in splits.items():
        part.to_csv(DATA_DIR / f"{name}.csv", index=False)
        info[name] = {
            "rows": len(part),
            "start": part["date"].iloc[0],
            "end": part["date"].iloc[-1],
            "up_rate": round(float(part["target"].mean()), 3),
            "days_with_news": int((part["n_articles"] > 0).sum()),
        }
        print(f"[make_splits] {name:5s}: {len(part):4d} rows  "
              f"{info[name]['start']} -> {info[name]['end']}  up_rate={info[name]['up_rate']}")

    # Sanity check: consecutive, non-overlapping date ranges.
    assert info["train"]["end"] < info["val"]["start"] < info["val"]["end"] < info["test"]["start"]
    with open(DATA_DIR / "split_info.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
