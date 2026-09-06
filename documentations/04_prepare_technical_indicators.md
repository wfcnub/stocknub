# Prepare Technical Indicators

**Pipeline module:** `pipeline.prepare_technical_indicators`  
**Executor step:** 3

## Purpose

Transforms selected tickers' OHLCV and IDX auxiliary histories into the feature tables used by model versions 1–3.

## Inputs and outputs

| Type | Location | Notes |
|---|---|---|
| Input | `data/dev/selected_ticker_and_industry_list.csv` | The executor enables selected-ticker filtering. |
| Input | `data/dev/stock/OHLCV/<TICKER>.csv` | Requires `Date`, `Open`, `High`, `Low`, `Close`, `Volume`. |
| Input | `data/dev/stock/foreign_flow_non_regular/<TICKER>.csv` | Requires foreign-flow and non-regular columns. |
| Output | `data/dev/stock/technical/<TICKER>.csv` | OHLCV plus generated features. |
| Output | `data/dev/technical_indicator_features.txt` | Sorted manifest of non-OHLCV feature names. |

## Processing

1. Delete and recreate the technical output directory.
2. Intersect OHLCV file names with the selected universe.
3. For each ticker, require non-empty OHLCV and auxiliary files and inner-join them by `Date` for auxiliary feature calculation.
4. Reject effectively constant close-price series.
5. Normalize and sort dates, reject duplicate dates, and require at least 279 OHLCV rows.
6. Generate feature families: price trends, channels, oscillators, volume-based signals, price transformations, foreign-flow/non-regular indicators, momentum, price characteristics, and market regime.
7. Replace infinities with missing values and discard the first 278 warm-up rows (driven by ADX convergence requirements).
8. Save each ticker table and derive the feature manifest from the first successful ticker.

The feature code includes ATR trailing stop, Aroon, ADX, Elder Ray, MACD, Supertrend, Vortex, Keltner/Donchian/Bollinger channels, RSI and stochastic signals, OBV/MFI/CMF/ADL/PVO, Fisher transform, and additional derived features.

## Executor invocation

```bash
python -m pipeline.prepare_technical_indicators --process_selected_ticker
```

## Failure behavior and cautions

- Individual ticker errors are collected and printed but do not make the module fail, so later stages may use a reduced universe.
- The output directory is destructive/rebuilt on every run.
- CLI folder arguments are printed, but processing and output lookups use `utils.paths` directly.
- `multiprocessing.set_start_method('spawn')` is set by the module; invoking it unusually within an already-configured Python process can raise a start-method error.

