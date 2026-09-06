# Generate Labels and Time Splits

**Pipeline module:** `pipeline.generate_labels`  
**Executor step:** 4

## Purpose

Adds forward-looking classification targets to every technical-feature table and creates common, chronological train/validation/test boundaries.

## Inputs and outputs

| Type | Location | Notes |
|---|---|---|
| Input | `data/dev/stock/technical/<TICKER>.csv` | Feature table with `Date` and target price column. |
| Output | `data/dev/stock/label/<TICKER>.csv` | Original table plus labels and threshold columns. |
| Output | `data/dev/split_dates_5.json` | Shared 5-day train/validation/test ranges. |
| Output | `data/dev/split_dates_10.json` | Shared 10-day train/validation/test ranges. |

## Label definitions

For each row and horizon, the process calculates the 40th percentile of the next `window` closing prices, expressed as a percentage change from the current close.

| Label type | Threshold estimated from purged training data | Positive class |
|---|---|---|
| `median_gain` | 90th percentile | `High Gain` when value is at or above the threshold |
| `median_loss` | 10th percentile | `High Loss` when value is at or below the threshold |

Threshold estimation excludes the configured validation and test tails and adds a horizon-sized embargo, preventing their future prices from influencing training thresholds. Each ticker receives columns such as `Median Gain 5dd` and `Threshold Median Gain 5dd`.

## Split generation

The module gathers all dates having a non-null label across successful tickers. From the ordered union it reserves the latest 80 dates for test and the preceding 40 for validation; all earlier dates become training. It writes one split JSON per window. If there are 120 or fewer valid dates, no split file is written for that horizon.

## Executor invocation

```bash
python -m pipeline.generate_labels \
  --windows 5,10 \
  --target_column Close \
  --label_types median_gain,median_loss
```

Standalone controls include `--workers`, `--test_length`, `--val_length`, and `--forecast_bool`. The latter skips split generation.

## Failure behavior and cautions

- The entire label directory is deleted before processing.
- Empty, missing, or near-constant price data is rejected per ticker. Per-ticker failures are printed but normally do not produce a non-zero module exit.
- Both label types use the same future 40th-percentile price statistic; their difference is the lower-tail versus upper-tail threshold and class mapping.
- CLI folder options do not currently redirect the `utils.paths`-based reads and writes.

