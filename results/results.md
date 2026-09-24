# Test-set results

NLP config: `{'use_lexicon': False, 'use_tfidf': False, 'tfidf_max_features': 5000, 'tfidf_min_df': 5, 'svd_components': 20}`

Test: 176 days, 2025-11-25 to 2026-09-01, UP rate 0.591

| model          | feature_set   |   accuracy |   f1_macro |    mcc |   roc_auc |   pred_up_rate |
|:---------------|:--------------|-----------:|-----------:|-------:|----------:|---------------:|
| majority_class | -             |      0.591 |      0.371 |  0     |     0.5   |          1     |
| persistence    | -             |      0.545 |      0.532 |  0.064 |     0.532 |          0.58  |
| logreg         | market        |      0.523 |      0.515 |  0.034 |     0.569 |          0.534 |
| xgboost        | market        |      0.58  |      0.546 |  0.101 |     0.548 |          0.682 |
| logreg         | nlp           |      0.494 |      0.48  |  0.097 |     0.615 |          0.244 |
| xgboost        | nlp           |      0.489 |      0.487 | -0.011 |     0.513 |          0.466 |
| logreg         | market+nlp    |      0.506 |      0.505 |  0.057 |     0.577 |          0.381 |
| xgboost        | market+nlp    |      0.597 |      0.542 |  0.119 |     0.579 |          0.756 |
