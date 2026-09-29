# Test-set results

NLP config: `{'use_lexicon': True, 'use_tfidf': True, 'tfidf_max_features': 5000, 'tfidf_min_df': 5, 'svd_components': 20}`

Test: 176 days, 2025-11-25 to 2026-09-01, UP rate 0.591

| model          | feature_set   |   accuracy |   f1_macro |    mcc |   roc_auc |   pred_up_rate |
|:---------------|:--------------|-----------:|-----------:|-------:|----------:|---------------:|
| majority_class | -             |      0.591 |      0.371 |  0     |     0.5   |          1     |
| persistence    | -             |      0.545 |      0.532 |  0.064 |     0.532 |          0.58  |
| logreg         | market        |      0.523 |      0.515 |  0.034 |     0.569 |          0.534 |
| xgboost        | market        |      0.625 |      0.507 |  0.177 |     0.533 |          0.898 |
| logreg         | nlp           |      0.489 |      0.489 |  0.015 |     0.509 |          0.398 |
| xgboost        | nlp           |      0.574 |      0.409 | -0.024 |     0.514 |          0.938 |
| logreg         | market+nlp    |      0.489 |      0.488 | -0.006 |     0.504 |          0.455 |
| xgboost        | market+nlp    |      0.602 |      0.478 |  0.101 |     0.579 |          0.898 |
