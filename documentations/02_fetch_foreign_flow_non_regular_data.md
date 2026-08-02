# Fetch Foreign Flow and Non-Regular Data

**Pipeline module:** `pipeline.fetch_foreign_flow_non_regular_data`  
**Executor step:** 1

## Purpose

Scrapes daily IDX Stock Summary workbooks, extracts foreign buy/sell and non-regular-market activity, and reorganizes the results into one chronological CSV per ticker.

## Inputs and outputs

| Type | Location | Notes |
|---|---|---|
| Input | `data/dev/stock/OHLCV/*.csv` | Supplies the active market dates. |
| Existing cache | `data/dev/stock/raw_foreign_flow_non_regular/*.csv` | Date-level raw files; used to determine backfill dates. |
| Output | `data/dev/stock/raw_foreign_flow_non_regular/YYYYMMDD.csv` | Cleaned date-level IDX downloads. |
| Output | `data/dev/stock/foreign_flow_non_regular/<TICKER>.csv` | Per-ticker chronological auxiliary data. |

Per-ticker output includes `Date`, `Foreign Sell`, `Foreign Buy`, `Non Regular Volume`, `Non Regular Value`, and `Non Regular Frequency`.

## Processing

1. Derive market dates from downloaded OHLCV data.
2. With the default `--fetch_type backfill`, subtract dates already present in the raw cache. `--fetch_type all` requests all derived dates.
3. Start Selenium/Chrome, select each year and month on the IDX page, and download the requested Stock Summary workbook.
4. Read the required columns, convert Indonesian dates to `YYYY-MM-DD`, delete the workbook, and save a date-level CSV.
5. Concatenate every raw CSV, group rows by `Stock Code`, sort and deduplicate by date, then save one CSV per ticker in parallel.

## Executor invocation

```bash
python -m pipeline.fetch_foreign_flow_non_regular_data
```

When the top-level executor receives `--with_docker`, it forwards the same flag to this module so Chrome is initialized for the Docker environment.

## Failure behavior and cautions

- The raw cache is retained between runs; the per-ticker output directory is also not cleared.
- A failed date is recorded in the printed summary, but the process normally exits successfully. Partial source coverage can therefore reach later steps.
- The implementation of `_get_all_active_market_date()` currently returns the dates read from the last OHLCV file visited rather than the accumulated union. Backfill scope therefore depends on that file until this is corrected.
- If new dates exist, the code concatenates all raw CSVs. At least one readable raw CSV with a `Stock Code` column is required.
- The CLI path options are not consistently used internally; effective storage is controlled by `utils.paths`.

