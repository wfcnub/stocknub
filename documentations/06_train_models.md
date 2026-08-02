# Train Models

**Pipeline module:** `pipeline.train_models`  
**Executor steps:** 5–7 and 13

## Purpose

Trains four complementary model versions for each requested label/horizon combination, saves serialized estimators, and records validation and locked-test evidence.

## Model versions

| Version | Training unit / identifier | Features | Estimator |
|---|---|---|---|
| V1 | One model per ticker | Technical plus same-date market/industry context | Tuned CatBoost ensemble |
| V2 | One pooled model per industry; `Ticker` and `Industry` are categorical | Same as V1 | Tuned CatBoost ensemble |
| V3 | One market-wide model saved as `IHSG` | Same as V2 | Tuned CatBoost ensemble |
| V4 | One market-wide stacker saved as `IHSG` | V1–V3 forecast probabilities | Tuned/scaled logistic regression |

The executor trains V1–V3 for both `median_gain` and `median_loss` at 5 and 10 days. V4 is trained only for `median_gain`, separately at 5 and 10 days.

## V1–V3 training workflow

1. Read label files and attach ticker/industry metadata; pooled versions concatenate the relevant universe.
2. Add causal, same-date market and industry medians and relative features.
3. Keep the configured test range locked away from tuning and purge development rows whose forward label horizon crosses into it.
4. Remove missing, constant, exact-duplicate, and overly sparse features (default maximum missing fraction 40%).
5. Build expanding walk-forward folds with a horizon-sized purge; defaults are four folds, 120 validation dates per fold, and at least 252 training dates.
6. Run up to 80 seeded Optuna trials over CatBoost parameters. The objective emphasizes median per-ticker AUC, pooled AUC, and average precision, with a fold-instability penalty.
7. Rank feature stability and retain at most 160 features, calibrate out-of-fold probabilities when calibration preserves ranking, choose an F1 decision threshold, and fit a three-member ensemble.
8. Evaluate train, out-of-fold validation, and untouched test data.

Training weights account for overlapping label events, optional recency decay, and the number of tickers observed per date.

## V4 training workflow

V4 reads `combined_forecasts_5dd` or `combined_forecasts_10dd`, uses all V1–V3 forecast columns as features, applies the shared chronological split, tunes logistic-regression regularization with Bayesian search on the predefined validation split, measures per-ticker validation/test results, and fits the final stacker on train-plus-validation data.

## Outputs

| Artifact | Pattern |
|---|---|
| Model | `data/dev/stock/model_v<V>/<label>/<IDENTIFIER>-<N>dd.pkl` |
| Aggregate metrics | `data/dev/stock/model_v<V>/performance/<label>/<N>dd.csv` |
| Failures | `data/dev/stock/model_v<V>/performance/failures.csv` |
| CatBoost studies | `data/dev/stock/tuning_studies/model_v<V>/*.sqlite3` |
| Artifacts | Model `artifacts/` directory: trials, feature importance, validation predictions, and metadata |

Metadata includes parameters, folds, selected/removed features, a development-data hash, library versions, prediction timing, and universe/test audit fields where applicable.

## Executor invocations

```bash
python -m pipeline.train_models --model_version 1 --windows 5,10 --label_types median_gain,median_loss
python -m pipeline.train_models --model_version 2 --windows 5,10 --label_types median_gain,median_loss
python -m pipeline.train_models --model_version 3 --windows 5,10 --label_types median_gain,median_loss --workers 4
python -m pipeline.train_models --model_version 4 --windows 5,10 --label_types median_gain
```

## Failure behavior and cautions

- The model and performance directories for the requested version/labels are deleted before training. Optuna SQLite studies are retained and keyed by the data/config identity.
- Failures of individual identifiers are written to `failures.csv`, but the module normally exits zero. The executor may proceed with only a subset of models.
- V1–V3 require the feature manifest and split JSONs. V4 requires the combined forecast datasets and their YAML column manifests.
- Performance rows use the name `Ticker` even when they identify an industry or market-wide model; forecasting logic translates these scopes back to tickers.

