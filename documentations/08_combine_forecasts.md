# Combine Forecasts

**Pipeline module:** `pipeline.combine_forecasts`  
**Executor steps:** 11–12

## Purpose

Creates the level-one datasets used by model V4 by joining forecast probabilities from model versions 1, 2, and 3 for every ticker/date available in all required forecast sets.

## Executor variants

| Step | Requested inputs | Output dataset | Feature count |
|---|---|---|---:|
| 11 | V1–V3 × gain/loss × 5-day | `data/dev/stock/combined_forecasts_5dd/` | 6 |
| 12 | V1–V3 × gain/loss × 5- and 10-day | `data/dev/stock/combined_forecasts_10dd/` | 12 |

## Processing

1. Use the maximum requested horizon to choose the output directory; delete and recreate that directory.
2. Write `data/dev/combined_forecasts_columns_information_<N>dd.yaml`, listing forecast features, the V4 target, and its threshold column.
3. Intersect ticker filenames across every requested label and horizon for model versions 1–3.
4. For each ticker, read and rename each forecast probability with a ` - V<version>` suffix.
5. Inner-join all tables on `Date`. Target and threshold columns are retained from the maximum-window `median_gain` table.
6. Reorder columns to `Date`, forecast features, threshold, and target; save one CSV per ticker.

## Executor invocations

```bash
python -m pipeline.combine_forecasts --model_versions 1,2,3 --windows 5 --label_types median_gain,median_loss
python -m pipeline.combine_forecasts --model_versions 1,2,3 --windows 5,10 --label_types median_gain,median_loss
```

## Failure behavior and cautions

- Ticker discovery is hard-coded to inspect model versions 1–3, even if `--model_versions` specifies a different list. The supplied list does control which files are combined and named in the feature manifest.
- Inner joins keep only dates present in every source forecast.
- Per-ticker failures are reported but do not make the module exit non-zero.
- `--csv_folder_path` is not used to select the destination; output is determined by `utils.paths` and the maximum window.

