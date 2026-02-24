"""
Pipeline Description: Fetch Historical Stock Data by Web Scraping

Usage:
    # Full download
    python -m pipeline.fetch_additional_historical_data --fetch_type all

    # Daily update or backfill
    python -m pipeline.fetch_additional_historical_data --fetch_type backfill
"""

import os
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from combineForecasts.main import process_single_ticker
from combineForecasts.helper import _get_emiten_available_on_all_forecasts, _list_combined_forecasts_feature_columns

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Description: Fetch additional historical stock data from IDX web"
    )
    parser.add_argument(
        "--csv_folder_path",
        type=str,
        default="data/stock/combined_forecasts",
        help="Directory path where CSV files will be saved (default: data/stock/combined_forecasts)",
    )
    parser.add_argument(
        "--label_types",
        type=str,
        default="median_gain,median_loss",
        help="Comma-separated label types (default: median_gain,median_loss)",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="5,10",
        help="Comma-separated rolling windows in days (default: 5,10)",
    )
    parser.add_argument(
        "--model_versions",
        type=str,
        default="1,2,3",
        help="Comma-separated of the model versions (default: 1,2,3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers (default: CPU count)",
    )

    args = parser.parse_args()

    label_types = [lt.strip() for lt in args.label_types.split(",")]
    rolling_windows = [int(w.strip()) for w in args.windows.split(",")]
    model_versions = [mv.strip() for mv in args.model_versions.split(",")]
    
    Path(args.csv_folder_path).mkdir(parents=True, exist_ok=True)
    
    csv_folder_path = os.path.join(os.getcwd(), args.csv_folder_path)

    print("=" * 80)
    print("PIPELINE DESCRIPTION: POST PROCESS ALL FORECASTS")
    print("=" * 80)
    
    print("Selecting emitens that are present on every model types")
    all_emiten = _get_emiten_available_on_all_forecasts(label_types, rolling_windows)

    print(f"Acquired {len(all_emiten)} that are present on every model types")

    process_args = [
        (
            emiten,
            label_types,
            rolling_windows, 
            model_versions, 
            args.csv_folder_path
        )
        for emiten in all_emiten
    ]

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_ticker, process_args),
                total=len(process_args),
                desc="Post Process Forecasts",
                unit="ticker",
            )
        )
    
    feature_columns = _list_combined_forecasts_feature_columns(label_types, rolling_windows, model_versions)
    output_path = 'data/forecast_features.txt'
    with open(output_path, "w") as file:
        for fea_col in feature_columns:
            file.write(fea_col + "\n")

    print("\n" + "=" * 80)
    print("FETCH SUMMARY")
    print("=" * 80)
    success_count = 0
    successful_tickers = []
    failed_tickers = []
    total_new_rows = 0

    for ticker, success, message, num_new_rows in results:
        if success:
            success_count += 1
            total_new_rows += num_new_rows
            successful_tickers.append((ticker, message, num_new_rows))
        else:
            failed_tickers.append((ticker, message))

    print(f"Successfully processed: {success_count}/{len(all_emiten)} tickers")
    print(f"Total new rows generated: {total_new_rows}")

    if failed_tickers:
        print(f"\nFailed: {len(failed_tickers)} tickers")
        print("-" * 80)

        for ticker, message in failed_tickers:
            print(f" - {ticker}: {message}")

    print("=" * 80)

if __name__ == "__main__":
    main()