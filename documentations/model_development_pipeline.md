# Model Development Pipeline Overview

**Entry point:** `model_development_pipeline.py`

## Goal

The development pipeline rebuilds the stock-analysis data and model stack in an isolated `data/dev` workspace, validates it through ordered processing stages, and promotes the completed workspace to `data/prod`.

```text
Yahoo OHLCV ─┐
             ├─> ticker selection ─> technical features ─> labels + time splits
IDX activity ┘                                      │
                                                    ├─> V1 ticker models ─┐
                                                    ├─> V2 industry models├─> combined forecasts ─> V4 stackers
                                                    └─> V3 market model ─┘
```

## Running it

From the repository root:

```bash
python model_development_pipeline.py
```

Use Docker-compatible Selenium initialization for the IDX fetch:

```bash
python model_development_pipeline.py --with_docker
```

Resume from a numbered step:

```bash
python model_development_pipeline.py --start_step 5
```

`--start_step` accepts 0–15 and runs that step and every later step; it does not run only one step.

## Ordered execution

| Step | Process | Fixed executor configuration | Documentation |
|---:|---|---|---|
| 0 | Fetch OHLCV | Start `2020-01-01` | [Fetch OHLCV](01_fetch_ohlcv_data.md) |
| 1 | Fetch IDX auxiliary data | Backfill; optional Docker mode | [Foreign flow and non-regular data](02_fetch_foreign_flow_non_regular_data.md) |
| 2 | Select universe | Default quality, score, and sector controls | [Ticker selection](03_select_ticker_to_process.md) |
| 3 | Generate technical features | Selected tickers only | [Technical indicators](04_prepare_technical_indicators.md) |
| 4 | Generate labels/splits | Close; gain/loss; 5/10 days | [Labels](05_generate_labels.md) |
| 5–7 | Train V1–V3 | Gain/loss; 5/10 days | [Training](06_train_models.md) |
| 8–10 | Forecast V1–V3 | Gain/loss; 5/10 days; validation Gini ≥ 0 | [Forecasting](07_forecast_stocks.md) |
| 11 | Combine 5-day forecasts | V1–V3, gain/loss, 5 days | [Combining](08_combine_forecasts.md) |
| 12 | Combine 5/10-day forecasts | V1–V3, gain/loss, 5/10 days | [Combining](08_combine_forecasts.md) |
| 13 | Train V4 | Gain only; separate 5/10-day stackers | [Training](06_train_models.md) |
| 14–15 | Forecast V4 | Gain only; 5 then 10 days; validation Gini ≥ 0 | [Forecasting](07_forecast_stocks.md) |

## Workspace lifecycle

1. The script sets `APP_ENV=dev`, so all `utils.paths` calls target `data/dev`.
2. Before any selected step runs, existing `data/dev` is deleted.
3. `data/prod` is copied to `data/dev`; if production does not exist, `data/base` is copied instead.
4. Each pipeline module runs in a new Python subprocess using the current interpreter.
5. Execution stops when a subprocess returns non-zero.
6. If all requested steps return zero, existing `data/prod` is deleted and the completed `data/dev` tree is copied into its place.

This provides run-level staging, but promotion is a delete-and-copy operation rather than an atomic directory swap.

## Main data contracts

| Stage | Principal artifact |
|---|---|
| Seed/configuration | `ticker_list.txt`, `ticker_and_industry_list.csv` |
| Market data | `stock/OHLCV/`, `stock/foreign_flow_non_regular/` |
| Universe | `selected_ticker_and_industry_list.csv`, tactical list, selection audit |
| Features | `stock/technical/`, `technical_indicator_features.txt` |
| Targets | `stock/label/`, `split_dates_5.json`, `split_dates_10.json` |
| Base models | `stock/model_v1/` through `stock/model_v3/` |
| Base forecasts | `stock/forecast/model_v1/` through `model_v3/` |
| Stacking data | `stock/combined_forecasts_5dd/`, `stock/combined_forecasts_10dd/` |
| Stacked models/forecasts | `stock/model_v4/`, `stock/forecast/model_v4/` |

Paths in this table are relative to `data/dev` during development and to `data/prod` after promotion.

## Model architecture

- **V1:** ticker-specific CatBoost models capture idiosyncratic behavior.
- **V2:** industry-level CatBoost models pool related stocks and use ticker/industry categorical features.
- **V3:** a market-wide CatBoost model pools the selected universe.
- **V4:** logistic-regression stackers combine V1–V3 probability outputs. The 5-day stacker has six inputs; the 10-day stacker uses twelve inputs spanning both horizons.

Gain and loss models predict whether the future-window 40th-percentile close return lies in an extreme training-derived tail. Shared chronological splits and horizon purges reduce forward-label leakage; V1–V3 tuning uses walk-forward validation and preserves a locked test period.

## Operational guarantees and limitations

- A non-zero module exit stops the run and prevents production promotion.
- Several modules treat per-ticker/date/model failures as successful process completion after printing or recording them. Consequently, **“All steps completed” means every subprocess returned zero, not that every unit of work succeeded.** Review selection audits, summaries, `failures.csv`, performance coverage, and forecast counts before relying on a promotion.
- Starting from a later step still rebuilds `data/dev` from current production/base and promotes it after the remaining steps succeed. Use this only when earlier required artifacts already exist in the copied source tree.
- Multiple stage CLIs expose folder arguments that their internal implementations do not consistently honor; the executor relies on the standard `utils.paths` layout.
- Most output directories are cleared by their owning stage. Avoid running overlapping pipeline instances against the same environment.

## Completion checklist

Before treating a promoted run as healthy, confirm:

1. The ticker-selection audit shows acceptable eligible/selected counts and fundamental coverage.
2. Technical and label summaries contain no unexpected ticker failures.
3. Every requested model performance CSV exists and any `failures.csv` is understood.
4. Forecast success totals and combined-dataset ticker counts match expectations.
5. V4 performance and final 5-day/10-day forecast directories are populated.

