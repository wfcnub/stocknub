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

The `compose.yaml` defines two main services:
- `pipeline-service`: Designed to execute the background pipeline scripts robustly.
- `initialize-jupyter-lab`: An interactive Jupyter Lab server exposed on port `8888`.

Before running the services, you must first build them:
```bash
docker compose build pipeline-service
docker compose build initialize-jupyter-lab
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