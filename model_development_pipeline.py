import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from warnings import simplefilter
simplefilter("ignore")

PIPELINE_STEPS = {
    0: {
        "name": "Fetch OHLCV Data",
        "module": "pipeline.fetch_ohlcv_data",
        "description": "Fetch Ticker's Open, High, Low, Close, and Volume (OHLCV) Historical Data using yfinance",
    },
    1: {
        "name": "Fetch Foreign Flow and Non Regular Data",
        "module": "pipeline.fetch_foreign_flow_non_regular_data",
        "description": "Fetch an additional historical data containing foreign flow and non-regular market from the IDX website",
    },
    2: {
        "name": "Select Ticker to Process",
        "module": "pipeline.select_ticker_to_process",
        "description": "Select Ticker to Process Based on The Recent Average Valuation",
    },    
    3: {
        "name": "Prepare Technical Indicators",
        "module": "pipeline.prepare_technical_indicators",
        "description": "Generate technical indicators for all downloaded stock data",
    },
    4: {
        "name": "Generate Labels",
        "module": "pipeline.generate_labels",
        "description": "Generate target labels for all tickers with technical indicators",
    },
    5: {
        "name": "Train Model V1",
        "module": "pipeline.train_models",
        "description": "Generate target labels for all tickers with technical indicators",
    },
    6: {
        "name": "Train Model V2",
        "module": "pipeline.train_models",
        "description": "Generate target labels for all tickers with technical indicators",
    },
    7: {
        "name": "Train Model V3",
        "module": "pipeline.train_models",
        "description": "Generate target labels for all tickers with technical indicators",
    },
    8: {
        "name": "Forecast Stocks V1",
        "module": "pipeline.forecast_stocks",
        "description": "Generate stock forecasts using the trained models",
    },
    9: {
        "name": "Forecast Stocks V2",
        "module": "pipeline.forecast_stocks",
        "description": "Generate stock forecasts using the trained models",
    },
    10: {
        "name": "Forecast Stocks V3",
        "module": "pipeline.forecast_stocks",
        "description": "Generate stock forecasts using the trained models",
    },
    11: {
        "name": "Combine Forecasts",
        "module": "pipeline.combine_forecasts",
        "description": "Combine forecast data from several model variations into a single file",
    },
    12: {
        "name": "Train Model V4",
        "module": "pipeline.train_models",
        "description": "Generate target labels for all tickers with technical indicators",
    },
    13: {
        "name": "Forecast Stocks V4",
        "module": "pipeline.forecast_stocks",
        "description": "Generate stock forecasts using the trained models",
    }
}

def run_step(step_num, args):
    """Run a single pipeline step as a Python module"""
    step = PIPELINE_STEPS[step_num]

    print(f"\n{'=' * 80}")
    print(f"RUNNING STEP {step_num}: {step['name'].upper()}")
    print(f"{'=' * 80}\n")

    cmd = [sys.executable, "-m", step["module"]]

    if step['module'] == "pipeline.train_models" and args.with_docker:
        cmd.extend(["--with_docker"])

    if step_num == 0:
        cmd.extend(["--start_date", '2020-01-01'])

    elif step_num == 1:
        pass
    
    elif step_num == 2:
        pass

    elif step_num == 3:
        cmd.extend(["--process_selected_ticker"])

    elif step_num == 4:
        cmd.extend(["--windows", '5,10'])
        cmd.extend(["--target_column", 'Close'])
        cmd.extend(["--label_types", 'median_gain,median_loss'])

    elif step_num == 5:
        cmd.extend(["--model_version", '1'])
        cmd.extend(["--windows", '5,10'])
        cmd.extend(["--label_types", 'median_gain,median_loss'])

    elif step_num == 6:
        cmd.extend(["--model_version", '2'])
        cmd.extend(["--windows", '5,10'])
        cmd.extend(["--label_types", 'median_gain,median_loss'])
    
    elif step_num == 7:
        cmd.extend(["--model_version", '3'])
        cmd.extend(["--windows", '5,10'])
        cmd.extend(["--label_types", 'median_gain,median_loss'])
    
    elif step_num == 8:
        cmd.extend(["--model_version", '1'])
        cmd.extend(["--windows", '5,10'])
        cmd.extend(["--label_types", 'median_gain,median_loss'])
        cmd.extend(["--csv_folder_path", 'data/stock/label'])
        cmd.extend(["--min_test_gini", '0'])
    
    elif step_num == 9:
        cmd.extend(["--model_version", '2'])
        cmd.extend(["--windows", '5,10'])
        cmd.extend(["--label_types", 'median_gain,median_loss'])
        cmd.extend(["--csv_folder_path", 'data/stock/label'])
        cmd.extend(["--min_test_gini", '0'])
    
    elif step_num == 10:
        cmd.extend(["--model_version", '3'])
        cmd.extend(["--windows", '5,10'])
        cmd.extend(["--label_types", 'median_gain,median_loss'])
        cmd.extend(["--csv_folder_path", 'data/stock/label'])
        cmd.extend(["--min_test_gini", '0'])
    
    elif step_num == 11:
        cmd.extend(["--model_versions", '1,2,3'])
        cmd.extend(["--windows", '5,10'])
        cmd.extend(["--label_types", 'median_gain,median_loss'])
    
    elif step_num == 12:
        cmd.extend(["--model_version", '4'])
        cmd.extend(["--windows", '5,10'])
        cmd.extend(["--label_types", 'median_gain,median_loss'])
    
    elif step_num == 13:
        cmd.extend(["--model_version", '4'])
        cmd.extend(["--windows", '10'])
        cmd.extend(["--label_types", 'median_gain'])
        cmd.extend(["--csv_folder_path", 'data/stock/combined_forecasts'])
        cmd.extend(["--min_test_gini", '0'])

    try:
        subprocess.run(cmd, check=True)
        print(f"\nStep {step_num} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nStep {step_num} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run the stocknub model development pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--with_docker",
        type=bool,
        default=False,
        help="A boolean for stating whether the system uses docker. If True, than the program wouldn't us multiprocessing",
    )

    args = parser.parse_args()

    steps_to_run = sorted(PIPELINE_STEPS.keys())

    print("\n" + "=" * 80)
    print("STOCKNUB DATA PIPELINE")
    print("=" * 80)
    print(f"Steps to run: {steps_to_run}")
    
    failed_steps = []
    for step_num in steps_to_run[5:6]:
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