# Stocknub Pipeline

A composable, batch-oriented data pipeline for Indonesian stock market analysis and prediction model development.

## Overview

Stocknub automates the entire workflow from raw stock data acquisition to model-ready datasets with technical indicators and target labels. The pipeline is designed for efficiency with incremental updates, parallel processing, and resumability.

## Pipeline Architecture

### Data Flow

```
data/stock/emiten_list.txt
    ↓
data/stock/00_historical/     ← Step 0: Raw OHLCV data from Yahoo Finance
    ↓
data/stock/01_technical/      ← Step 1: OHLCV + Technical Indicators (RSI, MACD, etc.)
    ↓
data/stock/02_label/          ← Step 2: Features + Target Labels (median_gain, max_loss, linear_trend)
    ↓
data/stock/03_model/          ← Step 3: Trained ML Models + Performance Metrics
    ↓
data/stock/04_forecast/       ← Step 4: Stock Forecasts with Probability Predictions
```

### Pipeline Steps

1. **Step 0: Fetch Historical Data** (`pipeline/00_fetch_historical_data.py`)
   - Downloads OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
   - Supports incremental updates (fetches only new data)
   - Parallel processing for all tickers

2. **Step 1: Generate Technical Indicators** (`pipeline/01_prepare_technical_indicators.py`)
   - Calculates technical indicators: RSI, MACD, Bollinger Bands, ADX, Stochastic, etc.
   - Incremental updates with context rows for proper calculation
   - Saves indicators as separate CSV files

3. **Step 2: Generate Target Labels** (`pipeline/02_generate_labels.py`)
   - Creates target labels for model training
   - Supports multiple label types:
     - `median_gain`: Median price gain over rolling window
     - `max_loss`: Maximum drawdown over rolling window
     - `linear_trend`: Linear regression gradient
   - Multiple rolling windows (e.g., 5, 10, 20 days)
   - Incremental updates with forward-looking context

4. **Step 3: Train Models** (`pipeline/03_train_models.py`)
   - Trains machine learning models for stock prediction
   - Uses all technical indicators as features
   - Parallel model training with multiprocessing
   - Evaluates models with train/test metrics (Gini, AUC, Accuracy)
   - Saves trained models and performance metrics

5. **Step 4: Generate Forecasts** (`pipeline/04_forecast.py`)
   - Generates probability predictions using trained models
   - Filters models by minimum test Gini performance
   - Prepares latest technical indicators for forecasting
   - Parallel forecast generation across multiple stocks
   - Saves forecasts with ticker, date, and probability

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### 🚀 Full Run Mode (First Time Setup)

Use this when starting from scratch or when `data/stock/` is empty:

```bash
python pipeline_orchestrator.py --full
```

**What it does:**
- Fetches all data from **2021-01-01 to today**
- Generates all technical indicators from scratch
- Creates all target labels
- Trains models for all stocks (if running steps 3-4)
- Generates forecasts (if running step 4)
- Uses default settings:
  - Workers: 10 parallel processes
  - Label types: median_gain, max_loss
  - Rolling windows: 5, 10, 20 days

---

#### 🔄 Update Run Mode (Daily Updates)

Use this for incremental updates after initial setup:

```bash
python pipeline_orchestrator.py --update-run
```

**What it does:**
- Reads the **last date** from existing historical data
- Fetches only **new data** from last date + 1 to today
- Incrementally updates technical indicators
- Incrementally updates target labels
- Re-trains models with updated data (if running steps 3-4)
- Generates fresh forecasts (if running step 4)
- Much faster than full run for data updates

**Recommended:** Run this daily or as needed to keep data current

---

### Advanced Usage

#### Run Specific Steps

```bash
# Run only data fetching
python pipeline_orchestrator.py --steps 0

# Run technical indicators and labels
python pipeline_orchestrator.py --steps 1 2

# Run model training and forecasting
python pipeline_orchestrator.py --steps 3 4 --label_types median_gain,max_loss --windows 5,10,20

# Run only forecasting (requires trained models)
python pipeline_orchestrator.py --steps 4 --min_test_gini 0.3

# Run with custom configuration
python pipeline_orchestrator.py --steps 0 --start_date 2023-01-01 --workers 5
```

---

### Running Individual Pipeline Steps

Each pipeline step can be run independently as a Python module from the project root:

```bash
# Step 0: Fetch historical data
python -m pipeline.00_fetch_historical_data --start_date 2021-01-01 --workers 10

# Step 1: Generate technical indicators  
python -m pipeline.01_prepare_technical_indicators --workers 10

# Step 2: Generate target labels
python -m pipeline.02_generate_labels --label_types median_gain,max_loss --windows 5,10,20 --workers 10

# Step 3: Train models
python -m pipeline.03_train_models --label_types median_gain,max_loss --windows 5,10,20 --workers 10

# Step 4: Generate forecasts
python -m pipeline.04_forecast --label_types median_gain,max_loss --windows 5,10,20 --min_test_gini 0.3 --workers 10
```

**Note:** Always run scripts as modules (`python -m pipeline.script_name`) from the project root to ensure proper module imports.

#### Custom Configuration

```bash
# Custom date range for fetching
python pipeline_orchestrator.py --steps 0 --start_date 2020-01-01 --end_date 2024-12-31

# Custom label types and windows for training and forecasting
python pipeline_orchestrator.py --steps 2 3 4 --label_types median_gain,max_loss --windows 10,20,30

# Forecast with minimum Gini threshold
python pipeline_orchestrator.py --steps 4 --min_test_gini 0.35 --workers 10

# Force reprocess all data (ignore existing)
python pipeline_orchestrator.py --all --force
```

**📖 For detailed documentation on individual scripts, arguments, and advanced usage, see [pipeline/README.md](pipeline/README.md)**

## Features

✅ **Composable** - Each step is independent and can run separately  
✅ **Batch Processing** - All tickers processed in parallel  
✅ **Incremental Updates** - Only processes new data, saves time  
✅ **Resumable** - Failed tickers are logged, pipeline continues  
✅ **No Initialization** - No setup phase needed, direct batch processing  
✅ **Context-Aware** - Maintains historical context for proper indicator/label calculation  

## Configuration

### Ticker List

Edit `data/stock/emiten_list.txt` to specify which stocks to process:

```
BBCA
BBRI
BBNI
TLKM
ASII
```

### Default Settings

Can be modified in `pipeline_orchestrator.py` or via command-line arguments:

- **Start Date**: 2021-01-01 (full run mode)
- **Workers**: 10 parallel processes
- **Label Types**: median_gain, max_loss
- **Rolling Windows**: 5, 10, 20 days
- **Target Column**: Close price
- **Min Test Gini**: None (use all models for forecasting)

## Output Structure

After running the pipeline, your data structure will look like:

```
data/stock/
├── emiten_list.txt
├── 00_historical/
│   ├── BBCA.csv          # Raw OHLCV data
│   ├── BBRI.csv
│   └── ...
├── 01_technical/
│   ├── BBCA.csv          # OHLCV + Technical Indicators
│   ├── BBRI.csv
│   └── ...
├── 02_label/
│   ├── BBCA.csv          # OHLCV + Indicators + Target Labels
│   ├── BBRI.csv
│   └── ...
├── 03_model/
│   ├── MedianGain/
│   │   ├── BBCA-5dd.pkl  # Trained models
│   │   ├── BBCA-10dd.pkl
│   │   └── ...
│   ├── MaxLoss/
│   │   └── ...
│   └── performance/
│       ├── MedianGain/
│       │   ├── 5dd.csv   # Model performance metrics
│       │   ├── 10dd.csv
│       │   └── ...
│       └── MaxLoss/
│           └── ...
└── 04_forecast/
    ├── MedianGain/
    │   ├── 5dd.csv       # Forecast probabilities
    │   ├── 10dd.csv
    │   └── ...
    └── MaxLoss/
        └── ...
```

## Troubleshooting

### No data fetched
- Check internet connection
- Verify ticker symbols are valid on Yahoo Finance (format: `TICKER.JK`)
- Check if tickers are listed in `data/stock/emiten_list.txt`

### Pipeline fails at Step 1, 2, 3, or 4
- Ensure previous steps completed successfully
- Check if CSV files exist in previous step's folder
- Try running with `--force` flag to reprocess (steps 1-2)
- For step 3: Ensure labels exist in `data/stock/02_label/`
- For step 4: Ensure models exist in `data/stock/03_model/`

### Slow performance
- Increase `--workers` count (but be mindful of CPU/memory limits)
- For daily updates, use `--update-run` instead of `--full`
- For step 3: Reduce number of label types or windows
- For step 4: Use `--min_test_gini` to forecast fewer stocks

## Next Steps

After running the full pipeline:
- **Analyze forecasts** in `data/stock/04_forecast/` for trading decisions
- **Review model performance** in `data/stock/03_model/performance/`
- **Backtest strategies** using historical predictions
- **Feature engineering** to improve model accuracy
- **Portfolio optimization** using forecast probabilities
- **Risk management** using max_loss forecasts

## Contributing

Contributions are welcome! Please ensure all pipeline steps remain composable and support incremental updates.

## License

[Add your license here]

