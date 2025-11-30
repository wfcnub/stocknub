"""
Pipeline Step 4: Forecast Stock Performance

This script generates forecasts using trained models from step 3.
It reads models from data/stock/{model_version}/{label_type}/*.pkl and outputs:
- Forecast results to data/stock/forecast_stocks/{label_type}/{window}dd.csv

The script:
- Filters stocks by minimum test Gini performance
- Reads technical indicators from existing CSV files (step 1 output)
- Applies trained models to generate probability predictions
- Saves forecasts for the most recent date

Usage:
    # Forecast with default settings (all models with min_test_gini >= 0.3)
    python -m pipeline.forecast_stocks --windows 5,10,15 --label_types median_gain,max_loss --min_test_gini 0.3

    # Forecast without Gini filter (use all available models)
    python -m pipeline.forecast_stocks --windows 5,10,15 --label_types median_gain

    # Forecast specific tickers only
    python -m pipeline.forecast_stocks --tickers BBCA,BBRI,TLKM --windows 5,10,15 --label_types median_gain
"""

import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from warnings import simplefilter
simplefilter(action="ignore")

from forecastStocksV1.main import process_single_ticker
from prepareTechnicalIndicators.helper import get_all_technical_indicators
from forecastStocksV1.helper import _ensure_directories_exist, _clear_forecast_files, _get_filtered_emiten_list, _save_forecast


def main():
    parser = argparse.ArgumentParser(
        description="Generate stock forecasts using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--label_types",
        type=str,
        default="median_gain,max_loss",
        help="Comma-separated label types (default: median_gain,max_loss)",
    )

    parser.add_argument(
        "--windows",
        type=str,
        default="5,10,20",
        help="Comma-separated forecast windows in days (default: 5,10,20)",
    )

    parser.add_argument(
        "--min_test_gini",
        type=float,
        default=None,
        help="Minimum test Gini coefficient for model filtering (default: None, use all models)",
    )

    parser.add_argument(
        "--technical_folder",
        type=str,
        default="data/stock/technical",
        help="Folder to save the technical (default: data/stock/technical)",
    )

    parser.add_argument(
        "--model_version",
        type=str,
        default="model_v1",
        help="Model version to be develop",
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help='Comma-separated list of specific tickers to forecast (e.g., "BBCA,BBRI,TLKM")',
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers (default: CPU count)",
    )

    args = parser.parse_args()

    label_types = [lt.strip() for lt in args.label_types.split(",")]
    windows = [int(w.strip()) for w in args.windows.split(",")]

    valid_label_types = ["linear_trend", "median_gain", "max_loss"]
    for label_type in label_types:
        if label_type not in valid_label_types:
            print(f"ERROR: Invalid label type: {label_type}")
            print(f"   Valid types: {', '.join(valid_label_types)}")
            return

    _ensure_directories_exist(args.model_version, label_types, windows)

    print("\nClearing old forecast files...")
    _clear_forecast_files(args.model_version, label_types, windows)

    feature_columns = get_all_technical_indicators()
    print(f"Using {len(feature_columns)} technical indicators as features")

    if args.tickers:
        emiten_list = [ticker.strip() for ticker in args.tickers.split(",")]
        print(f"\nForecasting specified tickers: {', '.join(emiten_list)}")
    else:
        print("\nFinding emiten with models meeting criteria...")
        if args.min_test_gini is not None:
            print(f"   Min Test Gini: {args.min_test_gini}")
        else:
            print("   Min Test Gini: None (using all available models)")

        emiten_list = _get_filtered_emiten_list(args.model_version, label_types, windows, args.min_test_gini)

        if not emiten_list:
            print("ERROR: No emiten found meeting the criteria")
            return

        print(f"Found {len(emiten_list)} emiten meeting criteria")

    forecast_tasks = []
    for emiten in emiten_list:
        for label_type in label_types:
            for window in windows:
                forecast_tasks.append((args.model_version, args.technical_folder, emiten, label_type, window, feature_columns))

    total_tasks = len(forecast_tasks)
    print(
        f"\nStarting forecasts for {len(emiten_list)} emiten × {len(label_types)} label types × {len(windows)} windows = {total_tasks} tasks"
    )
    print(f"Using {args.workers} parallel workers\n")

    successful = 0
    failed = 0

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_ticker, forecast_tasks),
                total=total_tasks,
                desc="Generating forecasts",
            )
        )

    print("\n" + "=" * 80)
    print("FORECAST SUMMARY")
    print("=" * 80)

    for emiten, label_type, window, success, message, forecast_data in results:
        if success and forecast_data is not None:
            successful += 1
            _save_forecast(forecast_data, args.model_version, label_type, window, emiten)
        else:
            failed += 1
            if failed <= 10:
                print(f"FAILED: {emiten} ({label_type}, {window}dd): {message}")

    if failed > 10:
        print(f"   ... and {failed - 10} more failures")

    print(f"\nSuccessful: {successful}/{total_tasks}")
    print(f"Failed: {failed}/{total_tasks}")

    if successful > 0:
        print(
            f"\nForecasts saved to: data/stock/forecast/{args.model_version}/{label_type}/{window}dd"
        )

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()