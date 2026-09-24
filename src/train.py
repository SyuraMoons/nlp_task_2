"""Train and evaluate every model: naive baselines, the exchange-rate-only baseline,
and the combined (exchange-rate + NLP) model.

Protocol (identical for every model, so the comparison is fair):
  1. Hyper-parameter search: fit on TRAIN, score on VALIDATION (metric: MCC).
  2. Refit the best setting on TRAIN + VALIDATION (the NLP extractor is refitted
     on the same data -- it never sees test text).
  3. Evaluate ONCE on TEST. The test split is not touched before this step.

Feature sets
  market     : exchange-rate history only            -> the Task 2 "baseline"
  nlp        : news features only                     -> ablation
  market+nlp : exchange-rate history + news features  -> the Task 2 "combined model"

Outputs (results/):
  metrics.csv           -- one row per (model, feature set, split)
  test_predictions.csv  -- per-day test predictions of every model
  results.md            -- the test table in markdown (paste into the report)

Run with: python -m src.train
"""
import numpy as np
import pandas as pd

from src.config import NLP_CONFIG, RESULTS_DIR
from src.data.load import load_all
from src.evaluation.metrics import evaluate
from src.features.market_features import MARKET_FEATURES, market_features
from src.features.nlp_features import NLPFeatureExtractor
from src.models.classifiers import MODELS, param_grid
from src.models.naive import MajorityClass, Persistence

FEATURE_SETS = ["market", "nlp", "market+nlp"]


def build_features(feature_set, fit_df, apply_dfs):
    """Fit the NLP extractor on fit_df only, then build X for every df in apply_dfs."""
    extractor = NLPFeatureExtractor().fit(fit_df) if "nlp" in feature_set else None
    out = []
    for df in apply_dfs:
        parts = []
        if "market" in feature_set:
            parts.append(market_features(df))
        if extractor is not None:
            parts.append(extractor.transform(df))
        out.append(pd.concat(parts, axis=1))
    return out


def run_naive(data, rows, preds):
    trainval = pd.concat([data["train"], data["val"]], ignore_index=True)
    for name, model in [("majority_class", MajorityClass()), ("persistence", Persistence())]:
        for split, fit_df in [("val", data["train"]), ("test", trainval)]:
            eval_df = data[split]
            model.fit(fit_df[MARKET_FEATURES], fit_df["target"])
            y_pred = model.predict(eval_df[MARKET_FEATURES])
            y_prob = model.predict_proba(eval_df[MARKET_FEATURES])[:, 1]
            rows.append({"model": name, "feature_set": "-", "split": split, "params": "",
                         **evaluate(eval_df["target"], y_pred, y_prob)})
            if split == "test":
                preds[name] = y_pred


def run_trained(data, rows, preds):
    trainval = pd.concat([data["train"], data["val"]], ignore_index=True)
    y_train, y_val, y_test = (data[s]["target"].to_numpy() for s in ("train", "val", "test"))
    y_trainval = trainval["target"].to_numpy()

    for feature_set in FEATURE_SETS:
        X_train, X_val = build_features(feature_set, data["train"], [data["train"], data["val"]])
        X_trainval, X_test = build_features(feature_set, trainval, [trainval, data["test"]])

        for model_name, (factory, grid) in MODELS.items():
            # 1. model selection on validation
            best = None
            for params in param_grid(grid):
                model = factory(**params).fit(X_train, y_train)
                scores = evaluate(y_val, model.predict(X_val), model.predict_proba(X_val)[:, 1])
                if best is None or scores["mcc"] > best[1]["mcc"]:
                    best = (params, scores)
            params, val_scores = best
            rows.append({"model": model_name, "feature_set": feature_set, "split": "val",
                         "params": str(params), **val_scores})

            # 2-3. refit on train+val, evaluate once on test
            model = factory(**params).fit(X_trainval, y_trainval)
            y_pred = model.predict(X_test)
            rows.append({"model": model_name, "feature_set": feature_set, "split": "test",
                         "params": str(params),
                         **evaluate(y_test, y_pred, model.predict_proba(X_test)[:, 1])})
            preds[f"{model_name}[{feature_set}]"] = y_pred
            print(f"[train] {model_name:8s} {feature_set:11s} best={params}  "
                  f"val_mcc={val_scores['mcc']:+.3f}")


def main():
    data = load_all()
    print(f"[train] NLP config: {NLP_CONFIG}")
    rows, preds = [], {}
    run_naive(data, rows, preds)
    run_trained(data, rows, preds)

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    pred_df = pd.DataFrame({"date": data["test"]["date"], "target": data["test"]["target"], **preds})
    pred_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    test = metrics[metrics["split"] == "test"].drop(columns=["split", "params"])
    table = test.round(3).to_markdown(index=False)
    (RESULTS_DIR / "results.md").write_text(
        f"# Test-set results\n\nNLP config: `{NLP_CONFIG}`\n\n"
        f"Test: {len(data['test'])} days, "
        f"{data['test']['date'].iloc[0]} to {data['test']['date'].iloc[-1]}, "
        f"UP rate {data['test']['target'].mean():.3f}\n\n{table}\n")
    print("\nTEST RESULTS\n" + table)


if __name__ == "__main__":
    np.random.seed(0)
    main()
