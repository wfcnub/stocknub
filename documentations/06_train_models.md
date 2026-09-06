# Train Models: Detailed Technical Guide

**Pipeline module:** `pipeline.train_models`

**Implementation:** `trainModels/`

**Development-pipeline steps:** 5–7 and 13

## 1. Architecture

This stage trains four complementary binary-classification model families.

| Version | Training scope | Inputs | Estimator |
|---|---|---|---|
| V1 | One model per ticker | Technical and cross-sectional features | Tuned CatBoost ensemble |
| V2 | One model per industry | V1 inputs plus ticker/industry categories | Tuned CatBoost ensemble |
| V3 | One market-wide model, saved as `IHSG` | V2 inputs | Tuned CatBoost ensemble |
| V4 | One market-wide stacker, saved as `IHSG` | V1–V3 forecast probabilities | Scaled elastic-net logistic regression |

V1–V3 are base learners at progressively broader pooling levels. V4 learns how to combine their forecasts. The normal pipeline trains V1–V3 for `median_gain` and `median_loss` at 5 and 10 trading days. V4 is trained for `median_gain` on separate 5- and 10-day combined datasets.

## 2. Code map and execution flow

| File | Responsibility |
|---|---|
| `pipeline/train_models.py` | CLI, work-item construction, multiprocessing, aggregate results |
| `trainModels/main.py` | V1–V4 orchestration |
| `trainModels/features.py` | Metadata, cross-sectional context, feature hygiene |
| `trainModels/splits.py` | Horizon parsing, purging, walk-forward folds |
| `trainModels/tuning.py` | CatBoost search, weighting, selection, calibration, ensemble |
| `trainModels/modelling.py` | Assembly, V4 tuning, metrics, per-ticker evaluation |
| `trainModels/config.py` | Reproducible V1–V3 defaults |
| `trainModels/helper.py` | Serialization and result-table assembly |

One V1–V3 work item follows this path:

```text
CLI -> process_single_model -> develop_model_v1/v2/v3
    -> assemble rows and context
    -> isolate locked test
    -> feature hygiene
    -> purged walk-forward folds
    -> Optuna CatBoost search
    -> stable feature selection and new OOF predictions
    -> OOF probability calibration and F1 threshold
    -> final seeded ensemble on all development rows
    -> train, OOF-validation, and locked-test metrics
    -> model + artifacts + aggregate metrics
```

The CLI creates the Cartesian product of identifier, label type, and horizon. V1 identifiers are tickers, V2 identifiers are industries, and V3/V4 use `IHSG`. Work items run in separate processes. CatBoost threads default to `CPU count // workers`, limiting nested oversubscription.

## 3. Targets

`get_label_config()` resolves requests as follows:

| Request | Target | Positive | Negative |
|---|---|---|---|
| `median_gain`, N | `Median Gain Ndd` | `High Gain` | `Low Gain` |
| `median_loss`, N | `Median Loss Ndd` | `High Loss` | `Low Loss` |

The matching `Threshold ...` column records the ticker-specific cutoff used upstream to create the class. It is reported with metrics but is not a model feature. Missing-target rows are dropped. See [Generate Labels](05_generate_labels.md) for the financial label definition.

## 4. V1–V3 data and features

V1 reads one ticker CSV. V2 concatenates selected tickers in one industry. V3 concatenates every CSV in the label directory; if that directory contains files beyond the selected-universe list, V3 includes them. Rows are sorted by date, and all rows sharing a date remain on the same side of each split.

Base numerical candidates come from the technical-feature manifest. V2/V3 also use `Ticker` and `Industry` as native CatBoost categorical features. V1 attaches this metadata for context construction but does not model it.

### Cross-sectional context

For `Log Return 1D`, `Realized Volatility 20D`, `RSI Value`, and `Volume ZScore 20D`, the code computes on each date:

```text
market median x(t)
industry median x(t)
x relative to market   = x(i,t) - market median x(t)
x relative to industry = x(i,t) - industry median x(t)
```

This adds 16 candidates. Only same-date observations are merged, so no future date is used. However, the historical universe is rebuilt from currently present label files; see the survivorship caveat below.

Features include the current completed session, making these explicitly **after-close** signals. A pre-market deployment must shift applicable features by one completed session in both training and inference.

### Feature hygiene

Hygiene is learned from development data only. A candidate is removed if it is absent, more than 40% missing by default, constant, or exactly duplicates an earlier selected numerical column. Present categorical columns bypass the latter checks. Removal reasons are saved. CatBoost handles remaining numerical missing values; categorical missing values become `Unknown`.

## 5. Time splitting and leakage control

Two evaluation mechanisms have different purposes:

- **Walk-forward validation** lies inside development data and drives all V1–V3 choices.
- **Locked test** is the configured final historical block and is scored only after selection.

Horizon-specific dates come from `split_dates_<N>.json`. The initial development period runs from configured train start through validation end. Before use, its final `H` unique dates are removed, where `H` is parsed from a target such as `Median Gain 10dd`. This prevents their forward label windows crossing into test.

Within development data, up to four expanding folds are created by default:

```text
fold 1: [ expanding train ][purge H][validation: 120 dates]
fold 2: [       expanding train       ][purge H][validation: 120 dates]
...
locked:                                                     [test]
```

Each fold requires at least 252 training dates and at least five observations of each class in both train and validation. Invalid folds are skipped; at least two must remain. The validation blocks are contiguous and non-overlapping. Purging prevents a training label from looking into its validation interval. There is no additional post-validation embargo.

## 6. Sample weights

Every V1–V3 CatBoost fit uses a product of weights, normalized to mean one:

1. **Event uniqueness.** Adjacent forward labels share future sessions. Per ticker, observations are downweighted according to the average inverse concurrency across their `H`-date event spans.
2. **Recency.** Optuna selects a half-life of 0 (disabled), 252, 504, or 756 unique dates:

   ```text
   w_recency = 0.5 ** (age_in_dates / half_life)
   ```

3. **Equal date contribution.** Weight is divided by rows observed on that date, preventing pooled dates with more tickers from automatically dominating. This is normally neutral for V1.

## 7. CatBoost optimization

Each V1–V3 work item has one seeded Optuna TPE study. A trial samples:

| Parameter | Space |
|---|---|
| Depth | integer 3–8 |
| Learning rate | 0.01–0.15, log |
| L2 leaf regularization | 0.01–100, log |
| Random strength | 0.001–10, log |
| Bagging temperature | 0–5 |
| `rsm` | 0.5–1.0 |
| Positive-class weight | 1–8, log |
| Recency half-life | 0, 252, 504, 756 |

CatBoost uses binary log loss, AUC evaluation, at most 3,000 trees, and 100-round early stopping by default. Each trial is fitted and scored on every valid fold.

For a fold:

```text
score = 0.65 * median within-ticker AUC
      + 0.20 * pooled AUC
      + 0.15 * average precision
```

V1's within-ticker AUC is its sole ticker AUC. In pooled models, one-class ticker slices are omitted; if no ticker AUC exists, pooled AUC substitutes. The final objective is:

```text
mean(fold scores) - 0.10 * standard deviation(fold scores)
```

Thus the search rewards discrimination and precision while penalizing temporal instability. A median pruner can stop weak trials; at least `max(10, n_trials/5)` startup trials precede pruning.

Studies persist in SQLite with `load_if_exists=True`. Identity hashes development dates/target/features, objective-relevant config, and ordered feature names. Trial count, timeout, threads, ensemble size, and final feature cap are excluded because they do not define the trial objective. Identical runs resume until the requested total number of trials is reached.

## 8. Stable feature selection

The best trial is rerun across folds to collect feature importance. For each feature:

```text
NonZeroRate    = fraction of folds with importance > 0
StabilityScore = MeanImportance / (StdImportance + 1e-9)
```

Ranking uses non-zero rate, mean importance, then stability. If candidates exceed 160, features active in at least half the folds are prioritized; mandatory categoricals come first. If the set changes, OOF models are refitted with it. Hyperparameters are not searched again, so this is post-search selection rather than nested selection.

## 9. Calibration, threshold, and final ensemble

Fold validation predictions are concatenated into an out-of-fold (OOF) table. Each OOF row was predicted without training on that row or later dates.

A Platt-style logistic model calibrates `log(p/(1-p))` against the binary target, clipping probabilities to `[1e-7, 1-1e-7]`. A non-positive fitted slope would invert rankings and is discarded. Raw and calibrated probabilities are both saved.

The classifier threshold is selected from unique probability quantiles between the 5th and 95th percentiles in 0.5-percentile steps, maximizing positive-class F1 over OOF rows. It affects class predictions, precision, and recall, but not AUC, Gini, or average precision.

The final tree count is:

```text
min(max_iterations, max(50, round(1.10 * median(best fold iterations))))
```

Three models by default are fitted on **all development rows** using consecutive seeds from `10120024`. The saved wrapper selects features in order, averages ensemble probabilities, applies the OOF calibrator, and uses the OOF threshold. Test data participates in none of these decisions.

## 10. V4 stacking

V4 reads its exact forecast features, target, and threshold from the combined-forecast YAML. Standard 5-day data has six features (three base versions × gain/loss); 10-day data has twelve (three versions × gain/loss × two horizons).

Unlike V1–V3, V4 uses one predefined chronological validation block. It purges the final `H` training dates before validation and final `H` validation dates before test.

Its pipeline is:

```text
RobustScaler -> LogisticRegression(
    solver="saga", penalty="elasticnet", class_weight="balanced",
    max_iter=1500, random_state=10120024)
```

`BayesSearchCV` performs 30 trials maximizing validation ROC AUC. It searches `C` from `1e-4` to `1e3` log-uniformly and `l1_ratio` from 0 to 1. A validation-only model is then trained on train rows to generate per-ticker validation evidence. The final model is refitted on purged train plus validation rows.

V4 uses threshold 0.5, no probability calibration, no event/recency/date sample weights, and no multi-fold validation. Its per-ticker evaluator currently catches and skips ticker errors, so output coverage must be checked.

## 11. Evaluation

Aggregate columns are prefixed `Train -`, `Validation -`, or `Test -`. Pooled models are trained jointly but reported per ticker.

| Metric | Meaning |
|---|---|
| Accuracy | Correct thresholded classifications |
| Precision/recall per class | Class-specific selection quality and coverage |
| ROC AUC | Ranking discrimination |
| Gini | `2 * AUC - 1` |
| Average Precision | Precision-recall summary |
| Precision Top 10% | Positive rate in highest-probability decile |
| Lift Top 10% | Top-decile precision / prevalence |
| Brier Score | Probability squared error; lower is better |
| Log Loss | Confidence-sensitive probability loss; lower is better |
| Positive/Predicted Positive Rate | Actual/predicted class prevalence |
| Decision Threshold | OOF-selected for V1–V3; 0.5 for V4 |

Validation Gini also receives a 95% date-cluster bootstrap interval when at least 20 unique dates exist: 200 fixed-seed samples resample whole dates, preserving same-session cross-sectional dependence. A one-class slice produces `NaN` for rank metrics.

Train performance is in-sample. V1–V3 validation is multi-fold OOF evidence; V4 validation is a single fixed block. Test is the untouched final estimate. Repeated manual changes after inspecting test results gradually convert it into another validation set.

## 12. Outputs and audit trail

Paths are relative to `data/dev` during development and `data/prod` after promotion.

| Output | Pattern/content |
|---|---|
| Model | `stock/model_v<V>/<label>/<IDENTIFIER>-<N>dd.pkl` |
| Performance | `stock/model_v<V>/performance/<label>/<N>dd.csv` |
| Failures | `stock/model_v<V>/performance/failures.csv` |
| Study | `stock/tuning_studies/model_v<V>/*.sqlite3` |
| Artifacts | Trial history, importance, validation predictions, metadata |

V1–V3 metadata records parameters, final iterations, selected/removed features, fold dates, ensemble seeds, study identity, a development-data SHA-256, library versions, prediction timing, locked-test dates, and universe audit fields. The hash covers dates, target, and selected feature values—not every metadata field or discarded input.

## 13. Bias controls and limitations

Implemented controls include chronological splitting, whole-date grouping, horizon purges, a test set excluded from all selection, multi-fold validation, OOF calibration/thresholding, persistent audit artifacts, and per-ticker reporting.

Remaining limitations:

- **Universe survivorship:** context uses label files currently present. Metadata marks `point_in_time_test_universe_available` false unless the selection snapshot predates test start.
- **After-close timing:** current-bar inputs cannot support a before-close claim.
- **Dependence:** purging reduces direct label overlap but financial rows and tickers remain dependent.
- **Post-search feature selection:** feature reduction is not nested inside a second hyperparameter search.
- **OOF reuse:** calibration and threshold selection reuse OOF predictions; test is the less biased check.
- **No economic backtest:** classification metrics do not establish net profitability, turnover, capacity, or risk.
- **Partial success:** work-item failures are recorded rather than re-raised, so a zero process exit does not guarantee complete coverage.

## 14. Running it

```bash
python -m pipeline.train_models --model_version 1 --windows 5,10 --label_types median_gain,median_loss
python -m pipeline.train_models --model_version 2 --windows 5,10 --label_types median_gain,median_loss
python -m pipeline.train_models --model_version 3 --windows 5,10 --label_types median_gain,median_loss --workers 4
python -m pipeline.train_models --model_version 4 --windows 5,10 --label_types median_gain
```

Focused V1–V3 controls:

```bash
python -m pipeline.train_models \
  --model_version 1 --label_types median_gain --windows 5 \
  --workers 4 --n_trials 80 --cv_folds 4 \
  --validation_dates 120 --min_train_dates 252 \
  --max_iterations 3000 --early_stopping_rounds 100 \
  --max_features 160 --ensemble_size 3
```

Also available: `--catboost_threads` and `--tune_timeout_seconds`. The CLI does not expose minimum class count, minimum valid folds, missingness limit, top fraction, or seeds.

Before training a model version, its requested label model/performance directories are deleted and recreated; separate SQLite studies remain. Always inspect `failures.csv`, expected model counts, ticker coverage, validation uncertainty, locked-test discipline, and artifact metadata.

## 15. Default V1–V3 configuration

| Setting | Default |
|---|---:|
| Optuna trials | 80 |
| Requested folds | 4 |
| Validation dates/fold | 120 |
| Minimum training dates | 252 |
| Minimum examples/class/partition | 5 |
| Minimum valid folds | 2 |
| Maximum trees | 3,000 |
| Early stopping | 100 |
| Ensemble size | 3 |
| Maximum missing fraction | 0.40 |
| Maximum selected features | 160 |
| Top metric fraction | 0.10 |
| Optimizer/model seed | 10120024 |

Library versions and execution details can still affect bit-for-bit reproducibility; the saved audit artifacts are intended to diagnose those differences.
