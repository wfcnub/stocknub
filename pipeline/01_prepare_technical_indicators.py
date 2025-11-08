"""
Pipeline Step 1: Generate Technical Indicators

This script generates technical indicators from historical stock data.
It reads from data/stock/00_historical/*.csv and outputs to data/stock/01_technical/*.csv

Usage:
    # Process all tickers (incremental update)
    python -m pipeline.01_prepare_technical_indicators --workers 10

    # Force reprocess all (ignore existing data)
    python -m pipeline.01_prepare_technical_indicators --force --workers 10
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

from technicalIndicators.main import generate_all_technical_indicators
from utils.io import get_last_date_from_csv


def process_single_ticker(args_tuple):
    """
    Process a single ticker: read historical data, generate technical indicators,
    and save to technical indicators folder.

    Args:
        args_tuple: Tuple containing (emiten, historical_folder, technical_folder, force_reprocess)

    Returns:
        Tuple of (emiten, success, message, num_new_rows)
    """
    emiten, historical_folder, technical_folder, force_reprocess = args_tuple

    try:
        historical_path = f"{historical_folder}/{emiten}.csv"
        technical_path = f"{technical_folder}/{emiten}.csv"

        # Check if historical data exists
        if not os.path.exists(historical_path):
            return (emiten, False, f"Historical data not found for {emiten}", 0)

        # Read historical data
        historical_df = pd.read_csv(historical_path)

        if historical_df.empty:
            return (emiten, False, f"Historical data is empty for {emiten}", 0)

        # Check for price variation (suspended/delisted stocks)
        if "Close" in historical_df.columns:
            import numpy as np

            close_variance = np.var(historical_df["Close"].values)
            if close_variance < 1e-10:
                return (
                    emiten,
                    False,
                    f"{emiten} - No price variation (likely suspended/delisted, variance={close_variance:.2e})",
                    0,
                )

        # Check if we need to process (incremental update logic)
        if not force_reprocess and os.path.exists(technical_path):
            last_technical_date = get_last_date_from_csv(technical_path)
            last_historical_date = str(historical_df.iloc[-1]["Date"])

            # If technical indicators are up to date, skip
            if last_technical_date == last_historical_date:
                return (emiten, True, f"Already up to date for {emiten}", 0)

            # Otherwise, we need to append new data
            # Read existing technical data
            existing_technical_df = pd.read_csv(technical_path)

            # Find rows in historical data that are newer than last technical date
            historical_df["Date"] = pd.to_datetime(historical_df["Date"])
            if last_technical_date:
                last_date_dt = pd.to_datetime(last_technical_date)
                new_historical_df = historical_df[
                    historical_df["Date"] > last_date_dt
                ].copy()
            else:
                new_historical_df = historical_df.copy()

            if new_historical_df.empty:
                return (emiten, True, f"No new data to process for {emiten}", 0)

            # Need to include some historical context for technical indicators calculation
            if len(existing_technical_df) > 0:
                # Get the last N dates from existing technical data
                existing_technical_df["Date"] = pd.to_datetime(
                    existing_technical_df["Date"]
                )
                last_n_dates = existing_technical_df["Date"]

                # Get corresponding rows from historical data
                context_df = historical_df[historical_df["Date"].isin(last_n_dates)]

                # Combine context with new data
                combined_df = pd.concat(
                    [context_df, new_historical_df], ignore_index=True
                )
            else:
                combined_df = new_historical_df

            # Generate technical indicators for combined data
            combined_df["Date"] = combined_df["Date"].dt.strftime("%Y-%m-%d")
            technical_df = generate_all_technical_indicators(combined_df)
            technical_df.reset_index(inplace=True)

            # Keep only the new rows (after the last technical date)
            if last_technical_date:
                technical_df["Date"] = pd.to_datetime(technical_df["Date"])
                new_technical_df = technical_df[
                    technical_df["Date"] > last_date_dt
                ].copy()
                new_technical_df["Date"] = new_technical_df["Date"].dt.strftime(
                    "%Y-%m-%d"
                )
            else:
                new_technical_df = technical_df.copy()
                new_technical_df["Date"] = new_technical_df["Date"].dt.strftime(
                    "%Y-%m-%d"
                )

            if not new_technical_df.empty:
                # Append to existing file
                new_technical_df.to_csv(
                    technical_path, mode="a", header=False, index=False
                )
                num_new_rows = len(new_technical_df)
                return (
                    emiten,
                    True,
                    f"Appended {num_new_rows} new rows for {emiten}",
                    num_new_rows,
                )
            else:
                return (
                    emiten,
                    True,
                    f"No new technical indicator rows for {emiten}",
                    0,
                )

        else:
            # Full reprocess: generate all technical indicators from scratch
            historical_df["Date"] = pd.to_datetime(historical_df["Date"]).dt.strftime(
                "%Y-%m-%d"
            )
            technical_df = generate_all_technical_indicators(historical_df)
            technical_df.reset_index(inplace=True)
            technical_df["Date"] = pd.to_datetime(technical_df["Date"]).dt.strftime(
                "%Y-%m-%d"
            )

            # Save to file
            technical_df.to_csv(technical_path, index=False)
            num_rows = len(technical_df)
            return (
                emiten,
                True,
                f"Generated {num_rows} rows of technical indicators for {emiten}",
                num_rows,
            )

    except Exception as e:
        return (emiten, False, f"Error processing {emiten}: {str(e)}", 0)


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
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess all tickers even if up to date",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated list of specific tickers to process (e.g., 'BBCA,BBRI,TLKM'). If not provided, all tickers will be processed.",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.technical_folder).mkdir(parents=True, exist_ok=True)

    # Get list of all tickers from historical folder
    all_historical_files = [
        f.replace(".csv", "")
        for f in os.listdir(args.historical_folder)
        if f.endswith(".csv")
    ]

    # Filter by specific tickers if provided
    if args.tickers:
        specified_tickers = [t.strip().upper() for t in args.tickers.split(",")]
        historical_files = [t for t in all_historical_files if t in specified_tickers]

        # Check if any specified tickers were not found
        missing_tickers = set(specified_tickers) - set(historical_files)
        if missing_tickers:
            print(
                f"⚠️  Warning: The following tickers were not found: {', '.join(missing_tickers)}"
            )

        if not historical_files:
            print(f"❌ None of the specified tickers found in {args.historical_folder}")
            return
    else:
        historical_files = all_historical_files

    if not historical_files:
        print(f"❌ No CSV files found in {args.historical_folder}")
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
    print(f"Force reprocess: {args.force}")
    print()

    # Prepare arguments for multiprocessing
    process_args = [
        (ticker, args.historical_folder, args.technical_folder, args.force)
        for ticker in historical_files
    ]

    # Process in parallel
    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_ticker, process_args),
                total=len(process_args),
                desc="Generating technical indicators",
                unit="ticker",
            )
        )

    # Print summary
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

    # Show successful tickers if using --tickers flag
    if args.tickers and successful_tickers:
        ticker_names = [ticker for ticker, _, _ in successful_tickers]
        print(f"\nSuccessful ({len(successful_tickers)} tickers):")
        print(f"Tickers: {', '.join(ticker_names)}")
        for ticker, msg, rows in successful_tickers:
            if rows > 0:
                print(f"  - {msg}")

    if failed_tickers:
        print(f"\nFailed: {len(failed_tickers)} tickers")
        print("-" * 80)

        # Categorize failures
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
