import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from generateLabels.main import process_single_ticker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Description: Generate target labels for all tickers with technical indicators"
    )

    parser.add_argument(
        "--technical_folder_path",
        type=str,
        default="data/stock/technical",
        help="Folder containing technical indicators (default: data/stock/technical)",
    )

    parser.add_argument(
        "--labels_folder_path",
        type=str,
        default="data/stock/label",
        help="Folder to save labels (default: data/stock/label)",
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
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers (default: CPU count)",
    )

    args = parser.parse_args()

    label_types = [lt.strip() for lt in args.label_types.split(",")]
    rolling_windows = [int(w.strip()) for w in args.windows.split(",")]

    if Path(args.labels_folder_path).exists():
        shutil.rmtree(args.labels_folder_path)

    Path(args.labels_folder_path).mkdir(parents=True, exist_ok=True)

    all_tickers = [file.stem for file in Path(args.technical_folder_path).rglob('*.csv')]

    print("=" * 80)
    print("PIPELINE DESCRIPTION: GENERATE TARGET LABELS")
    print("=" * 80)
    print(f"Found {len(all_tickers)} tickers to process")
    print(f"Technical data folder: {args.technical_folder_path}")
    print(f"Labels folder: {args.labels_folder_path}")
    print(f"Target column: {args.target_column}")
    print(f"Label types: {', '.join(label_types)}")
    print(f"Rolling windows: {', '.join([f'{w}d' for w in rolling_windows])}")
    print(f"Workers: {args.workers}")
    print()

    process_args = [
        (
            ticker,
            args.technical_folder_path,
            args.labels_folder_path,
            args.target_column,
            rolling_windows,
            label_types,
        )
        for ticker in all_tickers
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

    print(f"Successfully processed: {success_count}/{len(all_tickers)} tickers")
    print(f"Total new rows generated: {total_new_rows}")

    if failed_tickers:
        print(f"\nFailed: {len(failed_tickers)} tickers")
        print("-" * 80)

        for ticker, message in failed_tickers:
            print(f'{ticker} - {message}')

    print("=" * 80)