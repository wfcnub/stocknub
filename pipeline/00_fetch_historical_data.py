"""
Pipeline Step 0: Fetch Historical Stock Data

This script downloads historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.
It reads from data/stock/emiten_list.txt and outputs to data/stock/00_historical/*.csv

The script supports:
- Batch downloading all tickers in parallel
- Incremental updates (only fetches new data since last download)
- Resume from failures

Usage:
    # Full download from 2021
    python -m pipeline.00_fetch_historical_data --start_date 2021-01-01 --workers 10

    # Daily update (fetch until yesterday to avoid incomplete data)
    python -m pipeline.00_fetch_historical_data --update yesterday --workers 10

    # Daily update (fetch until today)
    python -m pipeline.00_fetch_historical_data --update today --workers 10
"""

import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from fetchHistoricalData.main import fetch_emiten_data
from fetchHistoricalData.helper import _get_yesterday_date

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Step 0: Fetch historical stock data from Yahoo Finance"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="",
        help="Start date in YYYY-MM-DD format (default: earliest available)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="",
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="data/emiten_list.txt",
        help="Path to the file containing emiten list (default: data/stock/emiten_list.txt)",
    )
    parser.add_argument(
        "--csv_folder_path",
        type=str,
        default="data/stock/00_historical",
        help="Directory path where CSV files will be saved (default: data/stock/00_historical)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers to use (default: CPU count)",
    )
    parser.add_argument(
        "--update",
        type=str,
        choices=["today", "yesterday"],
        default=None,
        help="Update existing data by reading the last date from each CSV file. Use 'today' to fetch until today, 'yesterday' to fetch until yesterday (useful for avoiding incomplete today's data)",
    )

    args = parser.parse_args()

    Path(args.csv_folder_path).mkdir(parents=True, exist_ok=True)

    if args.update:
        if args.update == "yesterday" and not args.end_date:
            args.end_date = _get_yesterday_date()
        
    with open(args.file_name, "r") as f:
        emiten_list = f.read().splitlines()

    fetch_args = [
        (emiten, args.start_date, args.end_date, args.csv_folder_path, args.update)
        for emiten in emiten_list
    ]

    mode_str = f"UPDATE mode (until {args.update})" if args.update else "FETCH mode"
    print("=" * 80)
    print("PIPELINE STEP 0: FETCH HISTORICAL DATA")
    print("=" * 80)
    print(
        f"Starting parallel fetch with {args.workers} workers for {len(emiten_list)} emitens ({mode_str})..."
    )
    if args.update:
        print(
            "Update mode: Will read last date from existing CSV files and fetch new data from that point."
        )
        if args.update == "yesterday":
            print(
                f"Fetching until yesterday ({args.end_date}) to avoid incomplete today's data."
            )
    print(
        f"Arguments: start_date='{args.start_date}', end_date='{args.end_date}', csv_folder_path='{args.csv_folder_path}'"
    )
    print()

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(fetch_emiten_data, fetch_args),
                total=len(fetch_args),
                desc="Fetching stock data",
                unit="emiten",
            )
        )

    print("\n" + "=" * 80)
    print("FETCH SUMMARY")
    print("=" * 80)
    success_count = 0
    failed_emitens = []

    for emiten, success, message in results:
        if success:
            success_count += 1
        else:
            failed_emitens.append((emiten, message))

    if failed_emitens:
        print("Failed data fetch:")
        for emiten, message in failed_emitens:
            print(f"  - {message}")
    else:
        print("All emitens fetched successfully!")

    print("=" * 80)
    print(f"Fetched: {success_count}/{len(emiten_list)} emitens")
    print("=" * 80)
