import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from forecastStocks.main import process_single_ticker
from prepareTechnicalIndicators.helper import get_all_technical_indicators
from combineForecasts.helper import _get_combined_forecasts_features_target_threshold
from forecastStocks.helper import (
    _ensure_directories_exist, 
    _get_filtered_ticker_list, 
    _save_forecast
)

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Description: Generate stock forecasts using the trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--label_types",
        type=str,
        default="median_gain,median_loss",
        help="Comma-separated label types (default: median_gain,max_loss)",
    )

    parser.add_argument(
        "--windows",
        type=str,
        default="5,10",
        help="Comma-separated forecast windows in days (default: 5,10,20)",
    )

    parser.add_argument(
        "--min_test_gini",
        type=float,
        default=None,
        help="Minimum test Gini coefficient for model filtering (default: None, use all models)",
    )

    parser.add_argument(
        "--csv_folder_path",
        type=str,
        default="data/stock/label",
        help="Folder to save the technical and label (default: data/stock/data/stock/label)",
    )

    parser.add_argument(
        "--model_version",
        type=int,
        default=1,
        help="Model version to be used for making forecasts",
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

    valid_label_types = ["median_gain", "median_loss"]
    for label_type in label_types:
        if label_type not in valid_label_types:
            print(f"ERROR: Invalid label type: {label_type}")
            print(f"   Valid types: {', '.join(valid_label_types)}")
            return

    print("=" * 80)
    print(f"PIPELINE DESCRIPTION: FORECAST USING MODEL V{args.model_version}")
    print("=" * 80)
    _ensure_directories_exist(args.model_version, label_types, windows)

    if args.model_version in [1, 2, 3]:
        feature_columns = get_all_technical_indicators()
        print(f"Using {len(feature_columns)} technical indicators as features")
    elif args.model_version == 4:
        feature_columns, _, _ = _get_combined_forecasts_features_target_threshold(np.max(windows))
        print(f"Using {len(feature_columns)} forecasts as features")

    print("\nFinding ticker with models meeting criteria...")
    if args.min_test_gini is not None:
        print(f"   Min Test Gini: {args.min_test_gini}")
    else:
        print("   Min Test Gini: None (using all available models)")

    ticker_list = _get_filtered_ticker_list(args.model_version, label_types, windows, args.min_test_gini)

    if not ticker_list:
        print("ERROR: No ticker found meeting the criteria")
        return

    print(f"Found {len(ticker_list)} ticker meeting criteria")

    if args.model_version == 1:
        model_identifier_list = ticker_list
    elif args.model_version == 2:
        ticker_industry_df = pd.read_csv(Path('data/selected_ticker_and_industry_list.csv'))
        ticker_industry_df = ticker_industry_df[ticker_industry_df['Ticker'].isin(ticker_list)]
        ticker_list = ticker_industry_df['Ticker'].values.tolist()
        model_identifier_list = ticker_industry_df['Industry'].values.tolist()
    elif args.model_version in [3, 4]:
        model_identifier_list = ['IHSG' for _ in range(len(ticker_list))]

    forecast_tasks = []
    for model_identifier, ticker in zip(model_identifier_list, ticker_list):
        for label_type in label_types:
            for window in windows:
                forecast_tasks.append((args.model_version, args.csv_folder_path, model_identifier, ticker, label_type, window, feature_columns))

    total_tasks = len(forecast_tasks)
    print(
        f"\nStarting forecasts for {len(ticker_list)} ticker × {len(label_types)} label types × {len(windows)} windows = {total_tasks} tasks"
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

    for ticker, label_type, window, success, message, forecast_data in results:
        if success and forecast_data is not None:
            successful += 1
            _save_forecast(forecast_data, args.model_version, label_type, window, ticker)
        else:
            failed += 1
            if failed <= 10:
                print(f"FAILED: {ticker} ({label_type}, {window}dd): {message}")

    if failed > 10:
        print(f"   ... and {failed - 10} more failures")

    print(f"\nSuccessful: {successful}/{total_tasks}")
    print(f"Failed: {failed}/{total_tasks}")

    if successful > 0:
        for label_type in label_types:
            for window in windows:
                print(
                    f"Forecasts saved to: data/stock/forecast/model_v{args.model_version}/{label_type}/{window}dd"
                )

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()