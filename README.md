# Task 2: Pipeline Proposal & Baseline Experimentation

NLP course project: *Global Geopolitical Event Prediction: Analyzing Impact on Dollar Exchange Rates.*
Task 2 proposes the NLP + ML pipeline for predicting USD/IDR movements and runs the first baseline experiments.

The input is the Task 1 dataset ([nlp_assignment_1](https://github.com/SyuraMoons/nlp_assignment_1)): one row per Bank Indonesia JISDOR trading day, with every Indonesian financial news headline aligned to that day.

👉 **Team members: start with [`TEAM_TASKS.md`](TEAM_TASKS.md)** to see who does what.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.make_splits   # data/raw/final_dataset.csv -> data/train|val|test.csv
python -m src.train              # trains everything -> results/
```

## Repository structure

```
data/
  raw/final_dataset.csv     # Task 1 output (input to this task)
  train.csv val.csv test.csv  # chronological split (built by src/data/make_splits.py)
  split_info.json           # date ranges, sizes, class balance
src/
  config.py                 # every design choice in one place
  data/make_splits.py       # target + past-only lag features + chronological split
  data/load.py
  features/market_features.py   # exchange-rate history features (baseline inputs)
  features/nlp_features.py      # text -> numeric features (fit on train only)
  features/check_nlp_features.py
  models/naive.py           # majority class, persistence
  models/classifiers.py     # logistic regression, XGBoost + hyper-parameter grids
  evaluation/metrics.py     # accuracy, macro-F1, MCC, ROC-AUC
  train.py                  # tune on val -> refit train+val -> evaluate on test
notebook/
  01_eda.ipynb              # exploratory data analysis
  02_results.ipynb          # initial experimental results
results/                    # metrics.csv, test_predictions.csv, results.md
report/                     # report outline, diagram draft, final PDF
```

## Task formulation

- **Target.** Binary: did the JISDOR USD/IDR rate go **UP** (1) or **DOWN** (0) at trading day *t* compared with *t−1*? Flat days are dropped.
- **Inputs at prediction time.** The prediction is made just before the 10:00 WIB fixing on day *t*. The model can use:
  - exchange-rate history up to *t−1*: return lags 1–5, rolling mean and std over 5 and 20 days, up-days in the last 5, and the calendar gap
  - the news aligned to *t* in Task 1, all of it published before that fixing

  Nothing from fixing *t* itself is a feature.
- **Models.** The same model families are used for every feature set, so any difference comes from the features:

  | Feature set | Role |
  |---|---|
  | naive baselines (majority class, persistence) | reference |
  | `market` (history only) | **baseline** |
  | `nlp` (news only) | ablation |
  | `market+nlp` | **combined model** |

  Each trainable set is run with logistic regression and with XGBoost.
- **Metrics.**
  - **MCC**, used for model selection: 0 means no better than a constant guess
  - **macro-F1**
  - **accuracy** (directional accuracy)
  - **ROC-AUC**

  Accuracy alone is misleading here: "always UP" scores 0.59 on test.

### Chronological split (no shuffling)

| Split | Days | Dates | UP rate |
|---|---|---|---|
| train | 817 | 2021-09-30 → 2025-02-24 | 0.562 |
| validation | 175 | 2025-02-25 → 2025-11-24 | 0.531 |
| test | 176 | 2025-11-25 → 2026-09-01 | 0.591 |

Hyper-parameters are chosen on validation. The best setting is refitted on train+validation and evaluated **once** on test. Anything learned from text (the TF-IDF vocabulary, SVD) is fitted on training data only.

## Current results (test set)

The NLP features here are only the placeholder set (news volume + GDELT tone). Lexicon sentiment and TF-IDF are Person B's part and will update this table.

| model | feature set | accuracy | macro-F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|
| majority class | – | 0.591 | 0.371 | 0.000 | 0.500 |
| persistence | – | 0.545 | 0.532 | 0.064 | 0.532 |
| logreg | market | 0.523 | 0.515 | 0.034 | 0.569 |
| xgboost | market | 0.625 | 0.507 | 0.177 | 0.533 |
| logreg | market+nlp | 0.506 | 0.505 | 0.057 | 0.577 |
| xgboost | market+nlp | 0.608 | 0.521 | 0.129 | 0.571 |

The full table, including the `nlp`-only ablation, is in `results/results.md`. With only ~800 training days, the differences are small and noisy. That is expected for daily FX direction, and it's what Tasks 3–4 will test properly.
