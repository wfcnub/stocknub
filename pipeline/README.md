# Pipeline Scripts

This folder contains the individual composable scripts that make up the Stocknub data pipeline. Each script can be run independently or orchestrated together using `pipeline_orchestrator.py`.

**Important:** All pipeline scripts must be run as Python modules from the project root directory to ensure proper imports:

```bash
python -m pipeline.00_fetch_historical_data [options]
python -m pipeline.01_prepare_technical_indicators [options]
python -m pipeline.02_generate_labels [options]
```

## Pipeline Steps

### Step 0: Fetch Historical Data

**Module:** `pipeline.00_fetch_historical_data`

Fetches OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance for all tickers listed in `data/stock/emiten_list.txt`.

#### Usage

```bash
# Full fetch from 2021-01-01 to today
python -m pipeline.00_fetch_historical_data --start_date 2021-01-01 --workers 10

# Incremental update (reads last date from existing CSVs, fetches from next day)
python -m pipeline.00_fetch_historical_data --update today --workers 10

# Update but stop at yesterday (avoids incomplete today's data)
python -m pipeline.00_fetch_historical_data --update yesterday --workers 10

# Custom date range
python -m pipeline.00_fetch_historical_data --start_date 2023-01-01 --end_date 2023-12-31 --workers 10
```

#### Arguments

- `--start_date` - Start date in YYYY-MM-DD format (default: earliest available)
- `--end_date` - End date in YYYY-MM-DD format (default: today)
- `--file_name` - Path to emiten list file (default: `data/stock/emiten_list.txt`)
- `--csv_folder_path` - Output folder (default: `data/stock/00_historical`)
- `--workers` - Number of parallel workers (default: 10)
- `--update` - Update mode: `today` or `yesterday` (automatically reads last date from CSVs)

#### Output

- `data/stock/00_historical/{TICKER}.csv` - One CSV per ticker with columns: Date, Open, High, Low, Close, Volume

---

### Step 1: Generate Technical Indicators

**Module:** `pipeline.01_prepare_technical_indicators`

Generates technical indicators from historical price data. Calculates various indicators including trend, momentum, volatility, and volume-based metrics.

#### Usage

```bash
# Process all tickers (incremental update)
python -m pipeline.01_prepare_technical_indicators --workers 10

# Force reprocess all (ignore existing data)
python -m pipeline.01_prepare_technical_indicators --force --workers 10

# Custom input/output folders
python -m pipeline.01_prepare_technical_indicators \
    --historical_folder data/stock/00_historical \
    --technical_folder data/stock/01_technical \
    --workers 10
```

#### Arguments

- `--historical_folder` - Input folder with historical data (default: `data/stock/00_historical`)
- `--technical_folder` - Output folder (default: `data/stock/01_technical`)
- `--workers` - Number of parallel workers (default: 10)
- `--force` - Force reprocess all tickers, ignore existing data

#### Technical Indicators Generated

**Price Trends:**
- ATR Trailing Stop
- Aroon Indicator
- Average Directional Index (ADX)
- Elder Ray Index
- MACD (Moving Average Convergence Divergence)

**Price Channels:**
- Keltner Channels
- Donchian Channels
- Bollinger Bands

**Oscillators:**
- Relative Strength Index (RSI)
- Stochastic Oscillator

**Volume-Based:**
- On-Balance Volume (OBV)
- Money Flow Index (MFI)
- Chaikin Money Flow (CMF)
- Accumulation/Distribution Line (ADL)

**Price Transformations:**
- Ehler Fisher Transform
- Zig Zag

#### Output

- `data/stock/01_technical/{TICKER}.csv` - One CSV per ticker with all OHLCV columns + technical indicator columns

#### Incremental Updates

When run without `--force`:
- Compares last date in technical CSV with last date in historical CSV
- If historical has newer data:
  - Takes last 200 rows from existing technical data (context for proper calculation)
  - Generates indicators for context + new data
  - Appends only new rows to existing CSV

---

### Step 2: Generate Target Labels

**Module:** `pipeline.02_generate_labels`

Creates target labels for model training based on forward-looking price movements.

#### Usage

```bash
# Generate all default labels (median_gain, max_loss, linear_trend for 5, 10, 20 days)
python -m pipeline.02_generate_labels --workers 10

# Specific label types and windows
python -m pipeline.02_generate_labels \
    --label_types median_gain,max_loss \
    --windows 5,10,20 \
    --workers 10

# Custom target column (default is Close)
python -m pipeline.02_generate_labels \
    --target_column High \
    --label_types median_gain \
    --windows 5 \
    --workers 10

# Force reprocess
python -m pipeline.02_generate_labels --force --workers 10
```

#### Arguments

- `--technical_folder` - Input folder with technical indicators (default: `data/stock/01_technical`)
- `--labels_folder` - Output folder (default: `data/stock/02_label`)
- `--target_column` - Column to use for label calculation (default: `Close`)
- `--label_types` - Comma-separated label types (default: `median_gain,max_loss,linear_trend`)
- `--windows` - Comma-separated rolling windows in days (default: `5,10,20`)
- `--workers` - Number of parallel workers (default: 10)
- `--force` - Force reprocess all tickers, ignore existing data

#### Label Types

**`median_gain`** - Median Gain Over Window
- Calculates the median percentage gain over the next N days
- Useful for identifying typical upward price movements
- Less sensitive to outliers than mean

**`max_loss`** - Maximum Loss Over Window
- Calculates the maximum drawdown over the next N days
- Useful for risk assessment and stop-loss strategies
- Identifies worst-case scenarios

**`linear_trend`** - Linear Regression Gradient
- Fits a linear regression to the next N days
- Gradient indicates trend direction and strength
- Useful for trend-following strategies

#### Examples

For a 5-day window on ticker BBCA:
- `Median Gain 5dd` - Column with median gain labels
- `Max Loss 5dd` - Column with max loss labels
- `Linear Trend 5dd` - Column with linear trend labels

For a 10-day window:
- `Median Gain 10dd`
- `Max Loss 10dd`
- `Linear Trend 10dd`

#### Output

- `data/stock/02_label/{TICKER}.csv` - One CSV per ticker with all columns from step 1 + label columns

#### Incremental Updates

When run without `--force`:
- Compares last date in labels CSV with last date in technical CSV
- If technical has newer data:
  - Takes last (max_window + 50) rows from existing labels (context for forward-looking calculation)
  - Generates labels for context + new data
  - Appends only new rows to existing CSV

---

## Running Scripts in Sequence

To manually run the full pipeline:

```bash
# Step 0: Fetch historical data
python pipeline/00_fetch_historical_data.py --start_date 2021-01-01 --workers 10

# Step 1: Generate technical indicators
python pipeline/01_prepare_technical_indicators.py --workers 10

# Step 2: Generate labels
python pipeline/02_generate_labels.py --workers 10
```

Or use the orchestrator (recommended):

```bash
# Full run
python pipeline_orchestrator.py --full

# Incremental update
python pipeline_orchestrator.py --update-run
```

## Performance Tips

1. **Workers:** Adjust based on your CPU cores and RAM
   - More workers = faster, but uses more memory
   - Recommended: 10-20 workers for most systems

2. **Incremental Updates:** Always use incremental mode for daily updates
   - Much faster than reprocessing everything
   - Only use `--force` when data quality issues occur

3. **Data Quality:** If indicators look wrong:
   - Use `--force` on step 1 to recalculate from scratch
   - Check for gaps in historical data

## Troubleshooting

### Script fails with "No CSV files found"
- Ensure previous step completed successfully
- Check if CSV files exist in the input folder
- Verify folder paths in arguments

### Indicators/Labels look incorrect
- Try running with `--force` to recalculate from scratch
- Check for data gaps in historical data
- Verify the rolling windows are appropriate for your data frequency

### Out of memory errors
- Reduce `--workers` count
- Process tickers in batches using a custom emiten list

### Slow performance
- Increase `--workers` if you have CPU/RAM capacity
- For updates, ensure you're using incremental mode (not `--force`)
- Check disk I/O speed (SSD recommended)
