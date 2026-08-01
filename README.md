# Stocknub Pipeline

This repository contains the end-to-end pipeline for stock data forecasting and analysis. The system supports full model development, executing daily forecasts, and visualizing the results through an interactive hub.

## Environment Setup

You can set up your development environment using either **Conda** or **Docker Compose**. Choose the method that best fits your workflow.

### Option 1: Setting up with Conda

To easily set up your local environment and manage dependencies, you can create a Conda environment using the provided `environment.yml` file. This will install Python 3.10, .NET SDK 8, and all required Python packages (such as Pandas, CMake, CatBoost, Streamlit, etc.).

1. Construct the conda environment:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the newly created environment:
   ```bash
   conda activate base_stocknub
   ```

### Option 2: Setting up with Docker Compose

If you prefer containerization, a `compose.yaml` file is included, which builds a custom image containing both .NET 8 and Python 3.10.

The `compose.yaml` defines three main services:
- `pipeline-service`: Designed to execute the background pipeline scripts robustly.
- `initialize-jupyter-lab`: An interactive Jupyter Lab server exposed on port `8888`.
- `initialize-fastapi`: A robust REST API serving endpoints powered by FastAPI, exposed on port `8000`.

Before running the services, you must first build them:
```bash
docker compose build pipeline-service
docker compose build initialize-jupyter-lab
docker compose build initialize-fastapi
```
*(Alternatively, you can build both at once using `docker compose build`)*

To run scripts inside the Docker container, you can execute:
```bash
docker compose run --rm pipeline-service python <script_name.py> --with_docker
```
Or to start the interactive JupyterLab server:
```bash
docker compose up initialize-jupyter-lab
```

---

## Execution Overview

Once your environment is properly configured and activated, there are three primary entry points to interact with the project:

### 1. Model Development
To train and develop the underlying forecasting models, execute the model development pipeline. This script runs the necessary functions for processing data and fitting your models:

**Using Conda:**
```bash
python model_development_pipeline.py
```

**Using Docker:**
```bash
docker compose run --rm pipeline-service python model_development_pipeline.py --with_docker
```

CatBoost V1-V3 development uses a locked final test period. Hyperparameters are
selected by one seeded Optuna study over purged expanding-window folds; the test
period is scored only after selection. Each fitted model also stores out-of-fold
calibration, a validation-tuned decision threshold, its selected features, and
the exact fold/trial history under the model's `artifacts` directory.
Optuna checkpoints studies under `stock/tuning_studies`; an interrupted run or
an identical rerun resumes instead of discarding completed trials.

For a focused training run, the principal compute controls are:

```bash
python -m pipeline.train_models \
  --model_version 1 \
  --n_trials 80 \
  --cv_folds 4 \
  --validation_dates 120 \
  --max_iterations 3000 \
  --early_stopping_rounds 100 \
  --ensemble_size 3 \
  --workers 4
```

The pipeline automatically limits CatBoost threads per worker so that outer
multiprocessing does not oversubscribe the machine. Forecast filtering uses
`Validation - Gini`; `--min_test_gini` remains only as a deprecated CLI alias
for `--min_validation_gini`.

Technical features include the current session's OHLCV bar, so V1-V3 forecasts
are explicitly **after-close** signals. A pre-market deployment must use the
previous completed bar (shift those features by one session) before training and
inference. Every run records the ticker-universe snapshot date; until historical
selection snapshots accumulate, the artifact marks point-in-time universe
validation as unavailable rather than presenting the current universe as a
historically unbiased one.

### 2. Daily Forecasts
To generate new forecasts using the latest available data, run the daily forecast script. This is intended to be executed on a regular, daily basis:

**Using Conda:**
```bash
python daily_forecasts.py
```

**Using Docker:**
```bash
docker compose run --rm pipeline-service python daily_forecasts.py --with_docker
```

### Ticker selection

The model-development pipeline selects a configurable universe using four stages:
historical OHLCV/data-quality gates, local traded-value gates, industry-relative
fundamental scoring, and cross-sectional technical scoring. The default is 75
model-development tickers plus a 25-name tactical shortlist.

The selector writes:

- `selected_ticker_and_industry_list.csv`: stable model-development universe;
- `tactical_ticker_list.csv`: technically strongest current subset;
- `ticker_selection_audit.csv`: every ticker, score component, and rejection reason;
- `ticker_selection_history.csv`: dated snapshots for prospective validation;
- `fundamental_history.csv`: cached, dated provider snapshots;
- `ticker_selection_forward_returns.csv`: per-ticker forward validation detail;
- `ticker_selection_validation_summary.csv`: aggregate return, drawdown, and stability metrics.

Important thresholds are CLI options:

```bash
python -m pipeline.select_ticker_to_process \
  --top_n 75 \
  --tactical_top_n 25 \
  --min_adv_60 5000000000 \
  --min_history_rows 504
```

Run the model-universe selection on a quarterly schedule to avoid unnecessary
per-ticker retraining. The tactical shortlist can be regenerated more frequently.
Once selection history has accumulated future observations, evaluate it with:

```bash
python -m pipeline.evaluate_ticker_selection \
  --benchmark_path path/to/ihsg.csv
```

### 3. Analytics Hub
For interactive analysis and visual exploration of your data and forecasts, launch the Streamlit-based Analytics Hub. It will start a local web server you can access via your browser:

**Using Conda:**
```bash
streamlit run analytics_hub.py
```

**Using Docker:**
```bash
docker compose run -p 8501:8501 --rm pipeline-service streamlit run analytics_hub.py
```

*(Note: If you are running the Streamlit app from within Docker, ensure that you expose Streamlit's default port `8501` in your `compose.yaml` file.)*

### 4. REST API (FastAPI)
To expose the pipeline data programmatically via a RESTful JSON API, you can launch the FastAPI server. It includes fully structured layers (Router, Controller, Service, Repository) and built-in interactive Swagger documentation available at `http://127.0.0.1:8000/docs`.

**Using Conda:**
```bash
uvicorn main:app --reload
```

**Using Docker:**
```bash
docker compose up initialize-fastapi
```
