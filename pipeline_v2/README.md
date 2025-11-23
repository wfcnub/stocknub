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

# Process specific tickers only (comma-separated)
python -m pipeline.01_prepare_technical_indicators --tickers "BBCA,BBRI,TLKM" --workers 4

# Process specific tickers with force reprocess
python -m pipeline.01_prepare_technical_indicators --tickers "BBCA,BBRI" --force

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
- `--tickers` - Comma-separated list of specific tickers to process (e.g., `"BBCA,BBRI,TLKM"`). If not provided, all tickers will be processed

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

# Process specific tickers only (comma-separated)
python -m pipeline.02_generate_labels --tickers BBCA,BBRI,TLKM --workers 4

# Process specific tickers with force reprocess
python -m pipeline.02_generate_labels --tickers BBCA,BBRI --force

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
- `--tickers` - Comma-separated list of specific tickers to process (e.g., `BBCA,BBRI,TLKM`). If not provided, all tickers will be processed

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

### Step 3: Train Models

**Module:** `pipeline.03_train_models`

Trains machine learning models for stock prediction using multiprocessing for faster parallel execution.

#### Usage

```bash
# Train with defaults (median_gain and max_loss for 5, 10, 20 days, auto workers)
python -m pipeline.03_train_models

# Train with specific label types and windows
python -m pipeline.03_train_models --label_types median_gain,max_loss --windows 5,10,20

# Specify number of parallel workers
python -m pipeline.03_train_models --workers 10

# Train specific tickers only (comma-separated)
python -m pipeline.03_train_models --tickers BBCA,BBRI,TLKM --workers 4

# Train specific tickers with specific label types
python -m pipeline.03_train_models --tickers BBCA,BBRI --label_types median_gain --windows 10

# Train all label types with custom workers
python -m pipeline.03_train_models --label_types linear_trend,median_gain,max_loss --windows 5,10,20 --workers 8

# Train only one label type and window
python -m pipeline.03_train_models --label_types median_gain --windows 10 --workers 4
```

#### Arguments

- `--label_types` - Comma-separated label types to train models for (default: `median_gain,max_loss`)
  - Options: `linear_trend`, `median_gain`, `max_loss`
- `--windows` - Comma-separated rolling windows in days (default: `5,10,20`)
- `--workers` - Number of parallel workers (default: CPU count - 1)
- `--tickers` - Comma-separated list of specific tickers to process (e.g., `BBCA,BBRI,TLKM`). If not provided, all tickers will be processed

#### Model Training

The script uses multiprocessing to train multiple stocks in parallel:
- Each worker processes one stock at a time
- For each stock, trains models for all specified label types and windows
- Default workers: automatically detects CPU cores and uses (cores - 1)
- Progress bar shows real-time processing status

For each combination of stock, label type, and window, the script:
1. Loads data from `data/stock/02_label/{TICKER}.csv`
2. Uses all technical indicators as features
3. Trains a model using the specified label as the target
4. Splits data into train/test sets with temporal ordering
5. Evaluates model performance (Gini, AUC, Accuracy, etc.)
6. Saves the trained model and performance metrics

#### Output

**Trained Models:**
- `data/stock/03_model/MedianGain/{TICKER}-5dd.pkl`
- `data/stock/03_model/MedianGain/{TICKER}-10dd.pkl`
- `data/stock/03_model/MaxLoss/{TICKER}-5dd.pkl`
- `data/stock/03_model/LinearTrend/{TICKER}-20dd.pkl`

**Performance Metrics:**
- `data/stock/03_model/performance/MedianGain/5dd.csv` - All stocks' performance for median gain 5-day models
- `data/stock/03_model/performance/MaxLoss/10dd.csv` - All stocks' performance for max loss 10-day models

Each performance CSV contains:
- `Kode` - Stock ticker
- `Train - Gini`, `Train - AUC`, `Train - Accuracy`, etc. - Training set metrics
- `Test - Gini`, `Test - AUC`, `Test - Accuracy`, etc. - Test set metrics
- `Threshold` - Classification threshold used

#### Data Requirements

- Stocks with < 100 clean data points (after removing NaNs) are automatically skipped
- All technical indicators must be present in the input files
- Label columns must match the specified label types and windows

#### Progress & Error Handling

- Uses multiprocessing Pool for parallel execution
- Shows progress bar with `tqdm` for all stocks being processed
- Each worker processes stocks independently
- Continues processing if individual models fail
- Reports summary of failed trainings at the end
- Automatically skips stocks missing required label columns

#### Example Output

```
Found 847 stocks to process
Label types: median_gain, max_loss
Rolling windows: 5, 10, 20 days
Workers: 7

Processing stocks: 100%|██████████| 847/847 [18:45<00:00,  1.33s/it]

============================================================
Training complete! Total stocks: 847

Failed trainings (5):
  - ARMY (median_gain 5dd): Insufficient positive samples
  - BOBA (max_loss 10dd): Insufficient negative samples
  ...

============================================================
```

---

### Step 4: Generate Forecasts

**Module:** `pipeline.04_forecast`

Generates forecasts using trained models from step 3. For each stock and model combination, it prepares the latest technical indicators and generates probability predictions for the target labels.

#### Usage

```bash
# Forecast with default settings (median_gain and max_loss, Gini >= 0.3)
python -m pipeline.04_forecast --windows 5,10,20 --label_types median_gain,max_loss --min_test_gini 0.3 --workers 10

# Forecast without Gini filter (use all available models)
python -m pipeline.04_forecast --windows 5,10,20 --label_types median_gain,max_loss --workers 10

# Forecast specific tickers only
python -m pipeline.04_forecast --tickers "BBCA,BBRI,TLKM" --windows 5,10 --label_types median_gain --workers 4

# Forecast single label type and window
python -m pipeline.04_forecast --label_types median_gain --windows 10 --min_test_gini 0.35 --workers 8
```

#### Arguments

- `--label_types` - Comma-separated label types (default: `median_gain,max_loss`)
  - Options: `linear_trend`, `median_gain`, `max_loss`
- `--windows` - Comma-separated forecast windows in days (default: `5,10,20`)
- `--min_test_gini` - Minimum test Gini coefficient for model filtering (default: `None`)
  - If specified, only uses models with `Test - Gini >= min_test_gini`
  - If `None`, uses all available models
  - Example: `0.3` will only use models with good predictive power
- `--tickers` - Comma-separated list of specific tickers to forecast (default: all models that meet criteria)
  - Example: `"BBCA,BBRI,TLKM"`
- `--workers` - Number of parallel workers (default: `10`)

#### How It Works

1. **Model Selection**:
   - If `--tickers` specified: Uses those specific tickers
   - Otherwise: Filters models based on `min_test_gini` from performance CSVs
   - Takes intersection of emiten that have models meeting criteria for all label types and windows

2. **Data Preparation**:
   - For each selected stock, reads latest technical indicators from step 1 output
   - Takes the most recent row (tail) with all calculated features
   - No need to download data or recalculate indicators

3. **Prediction**:
   - Loads trained model for each label type and window
   - Generates probability prediction for positive class
   - Saves forecast with ticker code, date, and probability

4. **Output**:
   - Clears old forecast files to avoid duplicates
   - Collects all results from parallel workers
   - Writes all forecasts in batch to CSV files
   - Organized by label type and window

#### Output

**Forecast Files:**
- `data/stock/04_forecast/MedianGain/5dd.csv` - High gain probability forecasts for 5-day window
- `data/stock/04_forecast/MedianGain/10dd.csv` - High gain probability forecasts for 10-day window
- `data/stock/04_forecast/MaxLoss/5dd.csv` - Low risk probability forecasts for 5-day window
- `data/stock/04_forecast/LinearTrend/20dd.csv` - Up trend probability forecasts for 20-day window

Each forecast CSV contains:
- `Kode` - Stock ticker
- `Date` - Forecast date (most recent date with available data)
- `Forecast {Label} {Window}dd` - Probability prediction (0.0 to 1.0)
  - For `median_gain`: Probability of "High Gain"
  - For `max_loss`: Probability of "Low Risk"
  - For `linear_trend`: Probability of "Up Trend"

#### Quality Control

The script automatically:
- Skips models that don't exist
- Validates that all required features are present
- Handles missing or incomplete data gracefully
- Reports failures with descriptive messages

#### Example Output

```
Using 127 technical indicators as features

Finding emiten with models meeting criteria...
Min Test Gini: 0.3
Found 245 emiten meeting criteria

Starting forecasts for 245 emiten × 2 label types × 3 windows = 1470 tasks
Using 10 parallel workers

Generating forecasts: 100%|██████████| 1470/1470 [05:23<00:00,  4.54it/s]

================================================================================
FORECAST SUMMARY
================================================================================
ABDA (median_gain, 5dd): Model not found: data/stock/03_model/MedianGain/ABDA-5dd.pkl
ARMY (max_loss, 10dd): Failed to prepare data: Insufficient data
   ... and 8 more failures

Successful: 1460/1470
Failed: 10/1470

Forecasts saved to: data/stock/04_forecast/{label_type}/{window}dd.csv

================================================================================
```

#### Best Practices

1. **Use Gini Filtering**: Set `--min_test_gini` to focus on well-performing models
   - `0.2` - Liberal threshold, includes most models
   - `0.3` - Moderate threshold, good balance
   - `0.4` - Conservative threshold, only strong models

2. **Targeted Forecasting**: Use `--tickers` when you only need specific stocks
   - Faster execution
   - Lower resource usage

3. **Regular Updates**: Run daily to get fresh forecasts with latest data
   - Ensure Steps 0-1 are run first to have up-to-date technical indicators
   - Forecasts use the most recent date available in technical indicators CSV
   - Old forecast files are automatically cleared to prevent duplicates

4. **Model Performance**: Review performance CSVs before forecasting
   - Check `data/stock/03_model/performance/{label_type}/{window}dd.csv`
   - Verify models have acceptable Gini coefficients

---

## Running Scripts in Sequence

To manually run the full pipeline:

```bash
# Step 0: Fetch historical data
python -m pipeline.00_fetch_historical_data --start_date 2021-01-01 --workers 10

# Step 1: Generate technical indicators
python -m pipeline.01_prepare_technical_indicators --workers 10

# Step 2: Generate labels
python -m pipeline.02_generate_labels --workers 10

# Step 3: Train models
python -m pipeline.03_train_models --label_types median_gain,max_loss --windows 5,10,20 --workers 10

# Step 4: Generate forecasts
python -m pipeline.04_forecast --label_types median_gain,max_loss --windows 5,10,20 --min_test_gini 0.3 --workers 10
```

Or use the orchestrator (recommended):

```bash
# Full run (includes model training)
python pipeline_orchestrator.py --full

# Incremental update
python pipeline_orchestrator.py --update-run
```

## Performance Tips

1. **Workers:** Adjust based on your CPU cores and RAM
   - **Steps 0-2:** More workers = faster, but uses more memory (default: 10-20 workers)
   - **Step 3:** Parallel model training (default: CPU count - 1)
   - Recommended: Start with defaults, adjust if memory issues occur
   - Example: `--workers 8` for 8-core CPU with limited RAM

2. **Incremental Updates:** Always use incremental mode for daily updates (Steps 0-2)
   - Much faster than reprocessing everything
   - Only use `--force` when data quality issues occur
   - Step 3 always trains from scratch on full datasets

3. **Data Quality:** If indicators look wrong:
   - Use `--force` on step 1 to recalculate from scratch
   - Check for gaps in historical data

4. **Model Training Speed:** 
   - Training speed scales with worker count (8 workers ≈ 8x faster)
   - More workers = higher CPU and memory usage
   - Reduce workers if system becomes unresponsive
   - Can take 15-60 minutes for 800+ stocks depending on worker count
   - Focus on specific label types/windows if you don't need all combinations

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
- Use `--tickers` to process specific stocks instead of all at once
- Process tickers in batches using a custom emiten list

### Slow performance on technical indicators
- Use `--tickers` to process only the stocks you need
- Example: `python -m pipeline.01_prepare_technical_indicators --tickers "BBCA,BBRI,TLKM"`
- Increase `--workers` if you have CPU/RAM capacity
- For updates, ensure you're using incremental mode (not `--force`)
- Check disk I/O speed (SSD recommended)

### Model training takes too long
- Train only the label types and windows you need
- Example: `python -m pipeline.03_train_models --label_types median_gain --windows 10`
- Consider training in batches by manually selecting stocks
- Training time scales linearly with number of stocks × label types × windows

### Models failing to train
- Check if label columns exist in `data/stock/02_label/` files
- Ensure stocks have sufficient data (>100 rows after NaN removal)
- Review error messages in the summary report
- Common issues: class imbalance, insufficient data, missing features
