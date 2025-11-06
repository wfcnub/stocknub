"""
Pipeline Step 2: Generate Target Labels

This script generates target labels for model training from technical indicator data.
It reads from data/stock/01_technical/*.csv and outputs to data/stock/02_label/*.csv

The script supports:
- Batch processing all tickers
- Multiple label types (linear_trend, median_gain, max_loss)
- Multiple rolling windows (e.g., 5, 10, 20 days)
- Incremental updates

Usage:
    # Generate labels with default settings
    python -m pipeline.02_generate_labels --label_types median_gain,max_loss --windows 5,10,20 --workers 10

    # Force regenerate all labels
    python -m pipeline.02_generate_labels --force --workers 10
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

from dataPreparation.helper import _generate_labels_based_on_label_type


def get_last_date_from_csv(csv_file_path: str) -> str:
    """
    Get the last date from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        str: Last date in 'YYYY-MM-DD' format, or empty string if file doesn't exist
    """
    if not os.path.isfile(csv_file_path):
        return ""

    try:
        df = pd.read_csv(csv_file_path)
        if df.empty:
            return ""
        return str(df.iloc[-1]["Date"])
    except Exception:
        return ""


def process_single_ticker(args_tuple):
    """
    Process a single ticker: read technical data, generate labels, and save.

    Args:
        args_tuple: Tuple containing (emiten, technical_folder, labels_folder,
                    target_column, rolling_windows, label_types, force_reprocess)

    Returns:
        Tuple of (emiten, success, message, num_new_rows)
    """
    (
        emiten,
        technical_folder,
        labels_folder,
        target_column,
        rolling_windows,
        label_types,
        force_reprocess,
    ) = args_tuple

    try:
        technical_path = f"{technical_folder}/{emiten}.csv"
        labels_path = f"{labels_folder}/{emiten}.csv"

        # Check if technical data exists
        if not os.path.exists(technical_path):
            return (emiten, False, f"Technical data not found for {emiten}", 0)

        # Read technical data
        technical_df = pd.read_csv(technical_path)

        if technical_df.empty:
            return (emiten, False, f"Technical data is empty for {emiten}", 0)

        # Check if we need to process (incremental update logic)
        if not force_reprocess and os.path.exists(labels_path):
            last_labels_date = get_last_date_from_csv(labels_path)
            last_technical_date = str(technical_df.iloc[-1]["Date"])

            # If labels are up to date, skip
            if last_labels_date == last_technical_date:
                return (emiten, True, f"Already up to date for {emiten}", 0)

            # Otherwise, we need to append new data
            existing_labels_df = pd.read_csv(labels_path)

            # Find rows in technical data that are newer than last labels date
            technical_df["Date"] = pd.to_datetime(technical_df["Date"])
            if last_labels_date:
                last_date_dt = pd.to_datetime(last_labels_date)
                new_technical_df = technical_df[
                    technical_df["Date"] > last_date_dt
                ].copy()
            else:
                new_technical_df = technical_df.copy()

            if new_technical_df.empty:
                return (emiten, True, f"No new data to process for {emiten}", 0)

            # Need context for label calculation (looking forward N days)
            max_window = max(rolling_windows)
            context_rows = max_window + 50  # Extra buffer

            if len(existing_labels_df) > 0:
                existing_labels_df["Date"] = pd.to_datetime(existing_labels_df["Date"])
                last_n_dates = existing_labels_df["Date"].tail(context_rows)

                context_df = technical_df[technical_df["Date"].isin(last_n_dates)]
                combined_df = pd.concat(
                    [context_df, new_technical_df], ignore_index=True
                )
            else:
                combined_df = new_technical_df

            # Generate labels for combined data
            combined_df["Date"] = combined_df["Date"].dt.strftime("%Y-%m-%d")
            labels_df = _generate_labels_based_on_label_type(
                combined_df, target_column, rolling_windows, label_types
            )

            # Keep only the new rows (after the last labels date)
            if last_labels_date:
                labels_df["Date"] = pd.to_datetime(labels_df["Date"])
                new_labels_df = labels_df[labels_df["Date"] > last_date_dt].copy()
                new_labels_df["Date"] = new_labels_df["Date"].dt.strftime("%Y-%m-%d")
            else:
                new_labels_df = labels_df.copy()
                new_labels_df["Date"] = labels_df["Date"].dt.strftime("%Y-%m-%d")

            if not new_labels_df.empty:
                # Append to existing file
                new_labels_df.to_csv(labels_path, mode="a", header=False, index=False)
                num_new_rows = len(new_labels_df)
                return (
                    emiten,
                    True,
                    f"Appended {num_new_rows} new rows for {emiten}",
                    num_new_rows,
                )
            else:
                return (emiten, True, f"No new label rows for {emiten}", 0)

        else:
            # Full reprocess: generate all labels from scratch
            technical_df["Date"] = pd.to_datetime(technical_df["Date"]).dt.strftime(
                "%Y-%m-%d"
            )
            labels_df = _generate_labels_based_on_label_type(
                technical_df, target_column, rolling_windows, label_types
            )

            # Save to file
            labels_df.to_csv(labels_path, index=False)
            num_rows = len(labels_df)
            return (
                emiten,
                True,
                f"Generated {num_rows} rows of labels for {emiten}",
                num_rows,
            )

    except Exception as e:
        return (emiten, False, f"Error processing {emiten}: {str(e)}", 0)


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
        default="median_gain,max_loss,linear_trend",
        help="Comma-separated label types (default: median_gain,max_loss,linear_trend)",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="5,10,20",
        help="Comma-separated rolling windows in days (default: 5,10,20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess all tickers even if up to date",
    )

    args = parser.parse_args()

    # Parse label types and windows
    label_types = [lt.strip() for lt in args.label_types.split(",")]
    rolling_windows = [int(w.strip()) for w in args.windows.split(",")]

    # Ensure output directory exists
    Path(args.labels_folder).mkdir(parents=True, exist_ok=True)

    # Get list of all tickers from technical folder
    technical_files = [
        f.replace(".csv", "")
        for f in os.listdir(args.technical_folder)
        if f.endswith(".csv")
    ]

    if not technical_files:
        print(f"❌ No CSV files found in {args.technical_folder}")
        return

    print("=" * 80)
    print("PIPELINE STEP 2: GENERATE TARGET LABELS")
    print("=" * 80)
    print(f"Found {len(technical_files)} tickers to process")
    print(f"Technical data folder: {args.technical_folder}")
    print(f"Labels folder: {args.labels_folder}")
    print(f"Target column: {args.target_column}")
    print(f"Label types: {', '.join(label_types)}")
    print(f"Rolling windows: {', '.join([f'{w}d' for w in rolling_windows])}")
    print(f"Workers: {args.workers}")
    print(f"Force reprocess: {args.force}")
    print()

    # Prepare arguments for multiprocessing
    process_args = [
        (
            ticker,
            args.technical_folder,
            args.labels_folder,
            args.target_column,
            rolling_windows,
            label_types,
            args.force,
        )
        for ticker in technical_files
    ]

    # Process in parallel
    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_ticker, process_args),
                total=len(process_args),
                desc="Generating labels",
                unit="ticker",
            )
        )

    # Print summary
    print("\n" + "=" * 80)
    print("LABEL GENERATION SUMMARY")
    print("=" * 80)

    success_count = 0
    failed_tickers = []
    total_new_rows = 0

    for ticker, success, message, num_new_rows in results:
        if success:
            success_count += 1
            total_new_rows += num_new_rows
        else:
            failed_tickers.append((ticker, message))

    print(f"Successfully processed: {success_count}/{len(technical_files)} tickers")
    print(f"Total new rows generated: {total_new_rows}")

    if failed_tickers:
        print(f"\nFailed: {len(failed_tickers)} tickers")
        for ticker, message in failed_tickers:
            print(f"  - {message}")

    print("=" * 80)


if __name__ == "__main__":
    main()
