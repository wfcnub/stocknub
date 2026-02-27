import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from prepareTechnicalIndicators.main import process_single_ticker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Description: Generate technical indicators for all downloaded stock data"
    )
    parser.add_argument(
        "--ohlcv_folder_path",
        type=str,
        default="data/stock/OHLCV",
        help="Folder containing historical stock data (default: data/stock/OHLCV)",
    )
    parser.add_argument(
        "--foreign_flow_non_regular_folder_path",
        type=str,
        default="data/stock/foreign_flow_non_regular",
        help="Folder containing historical stock data (default: data/stock/foreign_flow_non_regular)",
    )
    parser.add_argument(
        "--technical_folder_path",
        type=str,
        default="data/stock/technical",
        help="Folder to save technical indicators (default: data/stock/technical)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--process_selected_ticker",
        type=bool,
        default=True,
        help="A boolean enusring that the tickers being processed are just the selected ones",
    )

    args = parser.parse_args()

    if Path(args.technical_folder_path).exists():
        shutil.rmtree(args.technical_folder_path)

    Path(args.technical_folder_path).mkdir(parents=True, exist_ok=True)

    all_tickers = [file.stem for file in Path(args.ohlcv_folder_path).rglob('*.csv')]

    if not all_tickers:
        print(f"Error: No CSV files found in {args.ohlcv_folder_path}")

    if args.process_selected_ticker:
        selected_ticker_to_process_df = pd.read_csv('data/selected_ticker_and_industry_list.csv')
        selected_tickers = selected_ticker_to_process_df['Ticker'].values

        all_tickers_to_process = list(set(selected_tickers).intersection(set(all_tickers)))

    else:
        all_tickers_to_process = all_tickers

    print("=" * 80)
    print("PIPELINE DESCRIPTION: GENERATE TECHNICAL INDICATORS")
    print("=" * 80)
    print(f"Found {len(all_tickers_to_process)} tickers to process")
    print(f"Historical data folder: {args.ohlcv_folder_path}")
    print(f"Technical indicators folder: {args.technical_folder_path}")
    print(f"Workers: {args.workers}")
    print()

    process_args = [
        (ticker, args.ohlcv_folder_path, args.foreign_flow_non_regular_folder_path, args.technical_folder_path)
        for ticker in all_tickers_to_process
    ]

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_ticker, process_args),
                total=len(process_args),
                desc="Generating technical indicators",
                unit="ticker",
            )
        )

    print("\n" + "=" * 80)
    print("TECHNICAL INDICATORS GENERATION SUMMARY")
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

    print(f"Successfully processed: {success_count}/{len(all_tickers_to_process)} tickers")
    print(f"Total new rows generated: {total_new_rows}")

    if failed_tickers:
        print(f"\nFailed: {len(failed_tickers)} tickers")
        print("-" * 80)

        for ticker, message in failed_tickers:
            print(f"{ticker} - {message}")

    print("=" * 80)