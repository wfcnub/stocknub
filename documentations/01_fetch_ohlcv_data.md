# Fetch OHLCV Data

**Pipeline module:** `pipeline.fetch_ohlcv_data`  
**Executor step:** 0

## Purpose

Downloads daily Open, High, Low, Close, and Volume history for every configured Indonesian stock. Yahoo Finance is queried through `yfinance` using the `<TICKER>.JK` symbol convention.

## Inputs and outputs

| Type | Location / value | Notes |
|---|---|---|
| Input | `data/dev/ticker_list.txt` | One ticker code per line. |
| Input | `--start_date` | The executor fixes this to `2020-01-01`. |
| Input | `--end_date` | Empty by default, meaning the current date. |
| Output | `data/dev/stock/OHLCV/<TICKER>.csv` | Contains `Date`, OHLCV columns, and any remaining Yahoo history columns. |

All paths resolve under `data/dev` because the executor sets `APP_ENV=dev` before importing `utils.paths`.

## Processing

1. Delete and recreate the entire OHLCV output directory.
2. Read the ticker list; optionally intersect it with `selected_ticker_and_industry_list.csv` when `--process_selected_ticker` is supplied.
3. Fetch tickers in a multiprocessing pool.
4. For each ticker, request Yahoo history, remove `Dividends`, `Stock Splits`, and `Capital Gains` when present, normalize `Date`, and write one CSV.
5. On a fetch exception, wait 10 seconds and report the ticker as failed.
6. Print a success/failure summary.

## Executor invocation

```bash
python -m pipeline.fetch_ohlcv_data --start_date 2020-01-01
```

Useful standalone options are `--file_name`, `--csv_folder_path`, `--workers`, `--end_date`, and `--process_selected_ticker`.

## Failure behavior and cautions

- The output directory is removed before downloading, so a run does not preserve earlier OHLCV files.
- Individual ticker failures are reported but do not make the module exit non-zero. The parent executor can therefore continue after a partial download.
- Although the helper is structured as a three-iteration retry loop, its current `raise` is inside that loop. A ticker therefore receives only one attempt before being reported as failed.
- `--csv_folder_path` is displayed/configured by the CLI, but the implementation writes through `utils.paths`; changing the option does not currently redirect output.
- Network access and valid Yahoo Finance responses are required.
