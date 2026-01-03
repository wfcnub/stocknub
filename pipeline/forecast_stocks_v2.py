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

from forecastStocksV2.main import process_single_ticker
from prepareTechnicalIndicators.helper import get_all_technical_indicators
from forecastStocksV2.helper import _ensure_directories_exist, _clear_forecast_files, _get_filtered_emiten_list, _save_forecast


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
        default="5",
        help="Comma-separated forecast windows in days (default: 5,10,20)",
    )

    parser.add_argument(
        "--technical_folder",
        type=str,
        default="data/stock/technical",
        help="Folder to save the technical (default: data/stock/technical)",
    )

    parser.add_argument(
        "--industry",
        type=str,
        default=None,
        help='',
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

    _ensure_directories_exist(label_types, windows)

    print("\nClearing old forecast files...")
    _clear_forecast_files(label_types, windows)

    feature_columns = get_all_technical_indicators()
    print(f"Using {len(feature_columns)} technical indicators as features")

    emiten_and_industry_df = pd.read_csv('data/emiten_and_industry_list.csv')
    if args.industry:
        selected_industry = [t.strip().title() for t in args.industry.split(",")]

        selected_emiten_and_industry_df = emiten_and_industry_df.loc[emiten_and_industry_df['Industri'].isin(selected_industry), :]

        emiten_list = selected_emiten_and_industry_df['Kode'].values.tolist()
        industry_list = selected_emiten_and_industry_df['Industri'].values.tolist()

        print(f"\nForecasting specified industry: {', '.join(selected_industry)}")
    else:
        emiten_list = emiten_and_industry_df['Kode'].values.tolist()
        industry_list = emiten_and_industry_df['Industri'].values.tolist()

        print(f"Found {len(emiten_list)} emiten meeting criteria")

    forecast_tasks = []
    for emiten, industry in zip(emiten_list, industry_list):
        for label_type in label_types:
            for window in windows:
                forecast_tasks.append((args.technical_folder, industry, emiten, label_type, window, feature_columns))

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
            _save_forecast(forecast_data, label_type, window, emiten)
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
            f"\nForecasts saved to: data/stock/forecast/model_v2/{label_type}/{window}dd"
        )

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()