import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from fetchForeignFlowAndNonRegularData.main import(
    fetch_foreign_flow_and_non_regular_ticker_data, 
    process_foreign_flow_and_non_regular_ticker_data
)
from fetchForeignFlowAndNonRegularData.helper import (
    _get_all_active_market_date, 
    _get_all_active_market_date_to_backfill
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Description: Fetch an additional historical data containing foreign flow and non-regular market from the IDX website"
    )
    parser.add_argument(
        "--raw_csv_folder_path",
        type=str,
        default="data/stock/raw_foreign_flow_non_regular",
        help="Directory path where CSV files will be saved (default: data/stock/raw_foreign_flow_non_regular)",
    )
    parser.add_argument(
        "--csv_folder_path",
        type=str,
        default="data/stock/foreign_flow_non_regular",
        help="Directory path where CSV files will be saved (default: data/stock/foreign_flow_non_regular)",
    )
    parser.add_argument(
        "--fetch_type",
        type=str,
        choices=["all", "backfill"],
        default='backfill',
        help="",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers to use (default: CPU count)",
    )

    args = parser.parse_args()

    Path(args.raw_csv_folder_path).mkdir(parents=True, exist_ok=True)
    Path(args.csv_folder_path).mkdir(parents=True, exist_ok=True)

    if args.fetch_type == 'all':
        active_market_dates = _get_all_active_market_date()

    elif args.fetch_type == 'backfill':
        fetched_active_market_dates = _get_all_active_market_date()
        active_market_dates = _get_all_active_market_date_to_backfill(args.raw_csv_folder_path, fetched_active_market_dates)

    print("=" * 80)
    print("PIPELINE DESCRIPTION: FETCH FOREIGN FLOW AND NON-REGULAR DATA")
    print("=" * 80)
    print()
    
    if len(active_market_dates) > 0:
        print(f"Starting fetch for {len(active_market_dates)} active market dates")

        results = fetch_foreign_flow_and_non_regular_ticker_data(active_market_dates, args.raw_csv_folder_path)

        csv_files = Path(args.raw_csv_folder_path).rglob('*.csv')
        combined_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
        all_tickers = combined_df['Stock Code'].unique()

        fetch_args = [
            (ticker, combined_df, args.csv_folder_path)
            for ticker in all_tickers
        ]

        with Pool(processes=args.workers) as pool:
            _ = list(
                tqdm(
                    pool.imap(process_foreign_flow_and_non_regular_ticker_data, fetch_args),
                    total=len(fetch_args),
                    desc="Processing ticker's Foreign Flow and Non Regular data",
                    unit="ticker",
                )
            )

        print("\n" + "=" * 80)
        print("FETCH SUMMARY")
        print("=" * 80)
        
        success_count = 0
        failed_active_market_dates = []

        for active_market_date, success, message in results:
            if success:
                success_count += 1
            else:
                failed_active_market_dates.append((active_market_date, message))

        if failed_active_market_dates:
            print("Failed active market date fetch:")
            for active_market_date, message in failed_active_market_dates:
                print(f"{active_market_date}  - {message}")

        else:
            print("All active market dates fetched successfully!")

        print("=" * 80)
        print(f"Fetched: {success_count}/{len(active_market_dates)} weekday dates")
        print("=" * 80)
    
    else:
        print("=" * 80)
        print('All active market dates has been fetched')
        print("=" * 80)