"""
Run the stocknub pipeline to process data, generate v1 model, and forecast stocks

Usage:
    python model_v1_full_scoring_pipeline.py
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from warnings import simplefilter
simplefilter("ignore")


PIPELINE_STEPS = {
    0: {
        "name": "Fetch Historical Data",
        "module": "pipeline.fetch_historical_data",
        "description": "Download OHLCV data from Yahoo Finance",
    },
    1: {
        "name": "Generate Technical Indicators",
        "module": "pipeline.prepare_technical_indicators",
        "description": "Calculate technical indicators from historical data",
    },
    2: {
        "name": "Generate Target Labels",
        "module": "pipeline.generate_labels",
        "description": "Create target labels for model training",
    },
    3: {
        "name": "Train Models",
        "module": "pipeline.train_models_v1",
        "description": "Train machine learning models for stock prediction",
    },
    4: {
        "name": "Generate Forecasts",
        "module": "pipeline.forecast_stocks",
        "description": "Generate forecasts using trained models",
    },
}

def run_step(step_num, args):
    """Run a single pipeline step as a Python module."""
    step = PIPELINE_STEPS[step_num]

    print(f"\n{'=' * 80}")
    print(f"RUNNING STEP {step_num}: {step['name'].upper()}")
    print(f"{'=' * 80}\n")

    cmd = [sys.executable, "-m", step["module"]]

    if step_num == 0:
        cmd.extend(["--start_date", '2021-01-01'])

    elif step_num == 1:
        if args.tickers:
            cmd.extend(["--tickers", args.tickers])

    elif step_num == 2:
        cmd.extend(["--windows", '5,10,15'])
        cmd.extend(["--target_column", 'Close'])
        cmd.extend(["--label_types", 'median_gain'])

    elif step_num == 3:
        cmd.extend(["--windows", '5,10,15'])
        cmd.extend(["--label_types", 'median_gain'])
        cmd.extend(["--labels_folder", 'data/stock/label'])


    elif step_num == 4:
        cmd.extend(["--windows", '5,10,15'])
        cmd.extend(["--model_version", 'model_v1'])
        cmd.extend(["--label_types", 'median_gain'])
        cmd.extend(["--technical_folder", 'data/stock/technical'])

    try:
        subprocess.run(cmd, check=True)
        print(f"\nStep {step_num} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nStep {step_num} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the stocknub data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of specific tickers to process (e.g., 'BBCA,BBRI,TLKM'). If not provided, all tickers will be processed.",
    )

    Path("data/stock/historical").mkdir(parents=True, exist_ok=True)
    Path("data/stock/technical").mkdir(parents=True, exist_ok=True)
    Path("data/stock/label").mkdir(parents=True, exist_ok=True)
    Path("data/stock/model_v1").mkdir(parents=True, exist_ok=True)
    Path("data/stock/forecast").mkdir(parents=True, exist_ok=True)
    
    args = parser.parse_args()

    steps_to_run = sorted(PIPELINE_STEPS.keys())

    print("\n" + "=" * 80)
    print("STOCKNUB DATA PIPELINE")
    print("=" * 80)
    print(f"Steps to run: {steps_to_run}")
    
    failed_steps = []
    for step_num in steps_to_run:
        success = run_step(step_num, args)
        if not success:
            failed_steps.append(step_num)
            print(f"\nStopping pipeline due to failure in step {step_num}")
            break

    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    if not failed_steps:
        print("All steps completed!")
        return 0
    else:
        print(f"Pipeline failed at step {failed_steps[0]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
