# Forecast Stocks

**Pipeline module:** `pipeline.forecast_stocks`  
**Executor steps:** 8–10 and 14–15

## Purpose

Loads trained models, calculates positive-class probabilities for every eligible ticker/date, and writes a separate forecast table for each model version, label type, and horizon.

## Eligibility and model mapping

The module reads each requested combination's performance CSV, optionally keeps rows whose `Validation - Gini` meets `--min_validation_gini`, and takes the ticker intersection across all requested label/window combinations.

| Version | Model selected for a ticker | Prediction input |
|---|---|---|
| V1 | Ticker-specific model | `stock/label/<TICKER>.csv` |
| V2 | Ticker's industry model | Same label table |
| V3 | `IHSG` market model | Same label table |
| V4 | `IHSG` stacked model | `stock/combined_forecasts_<N>dd/<TICKER>.csv` |

For V1–V3, ticker metadata and the same-date cross-sectional context used in training are rebuilt before inference. The serialized model's `feature_names_` takes precedence over the current feature manifest.

## Output

Each task appends one probability column to the original input table and writes:

```text
data/dev/stock/forecast/model_v<V>/<label>/<N>dd/<TICKER>.csv
```

The probability column is named `Forecast High Gain <N>dd` or `Forecast High Loss <N>dd`.

## Executor invocations

Steps 8–10 run V1, V2, and V3 over both labels and both horizons using label data. Steps 14–15 run V4 `median_gain` independently over the 5-day and 10-day combined datasets. Every executor call sets `--min_validation_gini 0`, excluding negative-Gini models while accepting zero or better.

## Processing

1. Delete and recreate each requested forecast output directory.
2. Resolve the common ticker set from performance metrics.
3. Build all ticker × label × horizon tasks and run them in parallel.
4. Load each pickle, check required features, call `predict_proba`, and save successful outputs in the parent process.

## Failure behavior and cautions

Missing models, input files, features, or positive classes cause task-level failures. They are summarized, but the module normally exits zero. An empty eligible set also prints an error and returns successfully, so the top-level executor does not necessarily stop.

The `--csv_folder_path` argument is passed by the executor but is not used to locate inputs; effective paths come from `utils.paths` and the model version.
