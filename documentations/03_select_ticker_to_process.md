# Select Tickers to Process

**Pipeline module:** `pipeline.select_ticker_to_process`  
**Executor step:** 2

## Purpose

Builds a stable model-development universe by applying data-quality and liquidity gates, ranking eligible stocks on technical and fundamental strength, and enforcing sector diversification.

## Inputs and outputs

| Type | Default location | Purpose |
|---|---|---|
| Input | `data/dev/stock/OHLCV/*.csv` | Price history and liquidity/data-quality measurements. |
| Input | `data/dev/stock/foreign_flow_non_regular/*.csv` | Required auxiliary-data coverage by default. |
| Input | `data/dev/ticker_and_industry_list.csv` | `Ticker` to `Industry` mapping. |
| Input/cache | `data/dev/fundamental_history.csv` | Cached Yahoo fundamental history. |
| Input | Previous selected-universe file | Supplies the incumbent bonus when present. |
| Output | `data/dev/selected_ticker_and_industry_list.csv` | Ranked model universe, default maximum 150. |
| Output | `data/dev/tactical_ticker_list.csv` | Top 25 selected stocks by technical score. |
| Output | `data/dev/ticker_selection_audit.csv` | Full universe, metrics, scores, decisions, and rejection reasons. |

The CLI also prints a prospective history path, but this module does not write that file.

## Selection logic

1. **Readiness gates:** require 504 rows, freshness within 7 calendar days of the market-mode date, valid/unique bars, at least 220 traded days in the last 252, acceptable zero-volume and unchanged-price ratios, median 60-day traded value of at least IDR 5 billion, and at least 80% auxiliary-date overlap.
2. **Technical score:** cross-sectional percentile ranks combine trend (35%), momentum (30%), persistence (15%), downside risk/drawdown (10%), and volume confirmation (10%).
3. **Fundamental score:** industry-relative, winsorized ranks cover profitability, growth, valuation, financial health, and cash flow. Missing coverage is penalized. Financial-sector stocks omit the health and cash-flow factors from their aggregate.
4. **Eligibility floors:** fresh fundamentals, at least 50% fundamental metric coverage, fundamental score at least 0.20, and technical score at least 0.20.
5. **Final rank:** `0.60 × fundamental_score + 0.40 × technical_score`, plus a default 0.02 incumbent bonus.
6. **Sector controls:** seed up to two stocks per sector, then fill by rank while capping a sector at the larger of two stocks or 25% of the target universe.

## Fundamental data behavior

Fundamentals are fetched concurrently from Yahoo unless `--fundamental_snapshot_path` supplies an offline CSV. Cache entries default to a 30-day TTL. `--refresh_fundamentals` forces refresh; selection stops with `RuntimeError` if fresh fundamental coverage among OHLCV-eligible stocks is below 80%. The audit is saved before this check.

## Executor invocation

```bash
python -m pipeline.select_ticker_to_process
```

Important tuning flags include `--top_n`, `--tactical_top_n`, `--min_history_rows`, `--min_traded_days`, `--min_adv_60`, score floors, `--max_sector_share`, `--skip_auxiliary_data_check`, and `--disable_hysteresis`.

## Failure behavior

This stage raises on inadequate fundamental coverage and returns non-zero for uncaught input or processing errors. That stops the top-level executor. The audit provides the primary explanation surface for rejected stocks.

