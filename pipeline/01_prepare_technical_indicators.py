"""
Pipeline Step 1: Generate Technical Indicators

This script generates technical indicators from historical stock data.
It reads from data/stock/00_historical/*.csv and outputs to data/stock/01_technical/*.csv

Usage:
    # Process all tickers (incremental update)
    python -m pipeline.01_prepare_technical_indicators --workers 10
"""

import os
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from prepareTechnicalIndicators.main import process_single_ticker

def main():
    parser = argparse.ArgumentParser(
        description="Generate technical indicators for all downloaded stock data"
    )
    parser.add_argument(
        "--historical_folder",
        type=str,
        default="data/stock/00_historical",
        help="Folder containing historical stock data (default: data/stock/00_historical)",
    )
    parser.add_argument(
        "--technical_folder",
        type=str,
        default="data/stock/01_technical",
        help="Folder to save technical indicators (default: data/stock/01_technical)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated list of specific tickers to process (e.g., 'BBCA,BBRI,TLKM'). If not provided, all tickers will be processed.",
    )

    args = parser.parse_args()

    Path(args.technical_folder).mkdir(parents=True, exist_ok=True)

    all_historical_files = [
        f.replace(".csv", "")
        for f in os.listdir(args.historical_folder)
        if f.endswith(".csv")
    ]

    if args.tickers:
        specified_tickers = [t.strip().upper() for t in args.tickers.split(",")]
        historical_files = [t for t in all_historical_files if t in specified_tickers]

        missing_tickers = set(specified_tickers) - set(historical_files)
        if missing_tickers:
            print(
                f"Warning: The following tickers were not found: {', '.join(missing_tickers)}"
            )

        if not historical_files:
            print(
                f"Error: None of the specified tickers found in {args.historical_folder}"
            )
            return
    else:
        historical_files = all_historical_files

    if not historical_files:
        print(f"Error: No CSV files found in {args.historical_folder}")
        return

    print("=" * 80)
    print("PIPELINE STEP 1: GENERATE TECHNICAL INDICATORS")
    print("=" * 80)
    print(f"Found {len(historical_files)} tickers to process")
    if args.tickers:
        print(f"Processing specific tickers: {', '.join(historical_files)}")
    print(f"Historical data folder: {args.historical_folder}")
    print(f"Technical indicators folder: {args.technical_folder}")
    print(f"Workers: {args.workers}")
    print()

    process_args = [
        (ticker, args.historical_folder, args.technical_folder)
        for ticker in historical_files
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

    print(f"Successfully processed: {success_count}/{len(historical_files)} tickers")
    print(f"Total new rows generated: {total_new_rows}")

    if failed_tickers:
        print(f"\nFailed: {len(failed_tickers)} tickers")
        print("-" * 80)

        suspended_delisted = []
        other_errors = []

        for ticker, message in failed_tickers:
            if "No price variation" in message or "suspended/delisted" in message:
                suspended_delisted.append((ticker, message))
            else:
                other_errors.append((ticker, message))

        if suspended_delisted:
            ticker_names = [ticker for ticker, _ in suspended_delisted]
            print(f"\nSuspended/Delisted ({len(suspended_delisted)} tickers):")
            print(f"Tickers: {', '.join(ticker_names)}")
            for ticker, msg in suspended_delisted:
                print(f"  - {msg}")

        if other_errors:
            ticker_names = [ticker for ticker, _ in other_errors]
            print(f"\nOther Errors ({len(other_errors)} tickers):")
            print(f"Tickers: {', '.join(ticker_names)}")
            for ticker, msg in other_errors:
                print(f"  - {msg}")

    print("=" * 80)

if __name__ == "__main__":
    main()