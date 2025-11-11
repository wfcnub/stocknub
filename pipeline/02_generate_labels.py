"""
Pipeline Step 2: Generate Target Labels

This script generates target labels for model training from technical indicator data.
It reads from data/stock/01_technical/*.csv and outputs to data/stock/02_label/*.csv

The script supports:
- Batch processing all tickers
- Multiple label types (linear_trend, median_gain, max_loss)
- Multiple rolling windows (e.g., 5, 10, 20 days)

Usage:
    # Generate labels with default settings
    python -m pipeline.02_generate_labels --label_types median_gain,max_loss --windows 5,10,20 --workers 10

    # Process specific tickers
    python -m pipeline.02_generate_labels --tickers BBCA,BBRI,TLKM
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

from generateLabels.main import process_single_ticker

def main():
    parser = argparse.ArgumentParser(
        description="Generate target labels for all tickers with technical indicators"
    )
    parser.add_argument(
        "--technical_folder",
        type=str,
        default="data/stock/01_technical",
        help="Folder containing technical indicators (default: data/stock/01_technical)",
    )
    parser.add_argument(
        "--labels_folder",
        type=str,
        default="data/stock/02_label",
        help="Folder to save labels (default: data/stock/02_label)",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="Close",
        help="Target column for label generation (default: Close)",
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
        default="5,10",
        help="Comma-separated rolling windows in days (default: 5,10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated list of specific tickers to process (e.g., 'BBCA,BBRI,TLKM'). If not provided, all tickers will be processed.",
    )

    args = parser.parse_args()

    label_types = [lt.strip() for lt in args.label_types.split(",")]
    rolling_windows = [int(w.strip()) for w in args.windows.split(",")]

    Path(args.labels_folder).mkdir(parents=True, exist_ok=True)

    all_technical_files = [
        f.replace(".csv", "")
        for f in os.listdir(args.technical_folder)
        if f.endswith(".csv")
    ]

    if args.tickers:
        specified_tickers = [t.strip().upper() for t in args.tickers.split(",")]
        technical_files = [t for t in all_technical_files if t in specified_tickers]

        missing_tickers = set(specified_tickers) - set(technical_files)
        if missing_tickers:
            print(
                f"Warning: The following tickers were not found: {', '.join(missing_tickers)}"
            )

        if not technical_files:
            print(
                f"Error: None of the specified tickers found in {args.technical_folder}"
            )
            return
    else:
        technical_files = all_technical_files

    if not technical_files:
        print(f"Error: No CSV files found in {args.technical_folder}")
        return

    print("=" * 80)
    print("PIPELINE STEP 2: GENERATE TARGET LABELS")
    print("=" * 80)
    print(f"Found {len(technical_files)} tickers to process")
    if args.tickers:
        print(f"Processing specific tickers: {', '.join(technical_files)}")
    print(f"Technical data folder: {args.technical_folder}")
    print(f"Labels folder: {args.labels_folder}")
    print(f"Target column: {args.target_column}")
    print(f"Label types: {', '.join(label_types)}")
    print(f"Rolling windows: {', '.join([f'{w}d' for w in rolling_windows])}")
    print(f"Workers: {args.workers}")
    print()

    process_args = [
        (
            ticker,
            args.technical_folder,
            args.labels_folder,
            args.target_column,
            rolling_windows,
            label_types,
        )
        for ticker in technical_files
    ]

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_ticker, process_args),
                total=len(process_args),
                desc="Generating labels",
                unit="ticker",
            )
        )

    print("\n" + "=" * 80)
    print("LABEL GENERATION SUMMARY")
    print("=" * 80)

    success_count = 0
    failed_tickers = []
    limited_data_tickers = []
    total_new_rows = 0

    for ticker, success, message, num_new_rows in results:
        if success:
            success_count += 1
            total_new_rows += num_new_rows
            if "warning: only" in message:
                limited_data_tickers.append((ticker, message))
        else:
            failed_tickers.append((ticker, message))

    print(f"Successfully processed: {success_count}/{len(technical_files)} tickers")
    print(f"Total new rows generated: {total_new_rows}")

    if failed_tickers:
        print(f"\nFailed: {len(failed_tickers)} tickers")
        print("-" * 80)

        insufficient_data = []
        suspended_delisted = []
        generation_errors = []
        other_errors = []

        for ticker, message in failed_tickers:
            if "Insufficient data" in message:
                insufficient_data.append((ticker, message))
            elif "No price variation" in message or "suspended/delisted" in message:
                suspended_delisted.append((ticker, message))
            elif "empty dataframe" in message or "data quality" in message:
                generation_errors.append((ticker, message))
            else:
                other_errors.append((ticker, message))

        if insufficient_data:
            ticker_names = [ticker for ticker, _ in insufficient_data]
            print(f"\nInsufficient Data ({len(insufficient_data)} tickers):")
            for ticker, msg in insufficient_data:
                print(f"  - {msg}")
            print(f"Tickers: {','.join(ticker_names)}")

        if suspended_delisted:
            ticker_names = [ticker for ticker, _ in suspended_delisted]
            print(f"\nSuspended/Delisted ({len(suspended_delisted)} tickers):")
            for ticker, msg in suspended_delisted:
                print(f"  - {msg}")
            print(f"Tickers: {','.join(ticker_names)}")

        if generation_errors:
            ticker_names = [ticker for ticker, _ in generation_errors]
            print(f"\nData Quality Issues ({len(generation_errors)} tickers):")
            for ticker, msg in generation_errors:
                print(f"  - {msg}")
            print(f"Tickers: {','.join(ticker_names)}")

        if other_errors:
            ticker_names = [ticker for ticker, _ in other_errors]
            print(f"\nOther Errors ({len(other_errors)} tickers):")
            for ticker, msg in other_errors:
                print(f"  - {msg}")
            print(f"Tickers: {','.join(ticker_names)}")

    print("=" * 80)


if __name__ == "__main__":
    main()
