import os
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from fetchAdditionalHistoricalData.main import fetch_additional_emiten_data, process_additional_historical_data
from fetchAdditionalHistoricalData.helper import _get_latest_date, _get_all_weekdays_from_selected_date, _get_all_weekstart_to_backfill

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Step 0: Fetch historical stock data from Yahoo Finance"
    )
    parser.add_argument(
        "--csv_folder_path",
        type=str,
        default="data/stock/raw_additional_historical",
        help="Directory path where CSV files will be saved (default: data/stock/additional_historical)",
    )
    parser.add_argument(
        "--processed_folder_path",
        type=str,
        default="data/stock/additional_historical",
        help="Directory path where CSV files will be saved (default: data/stock/additional_historical)",
    )
    parser.add_argument(
        "--fetch_type",
        type=str,
        choices=["update", "all", "backfill"],
        default='update',
        help="",
    )

    args = parser.parse_args()

    Path(args.csv_folder_path).mkdir(parents=True, exist_ok=True)
    Path(args.processed_folder_path).mkdir(parents=True, exist_ok=True)

    csv_folder_path = os.path.join(os.getcwd(), args.csv_folder_path)

    if args.fetch_type == 'update':
        latest_date = _get_latest_date(args.csv_folder_path)
        weekday_dates = _get_all_weekdays_from_selected_date(latest_date)

    elif args.fetch_type == 'all':
        weekday_dates = _get_all_weekdays_from_selected_date('2025-12-01')

    elif args.fetch_type == 'backfill':
        all_weekday_dates = _get_all_weekdays_from_selected_date('2025-12-01')
        weekday_dates = _get_all_weekstart_to_backfill(csv_folder_path, all_weekday_dates)

    print("=" * 80)
    print("PIPELINE STEP 0.2: FETCH ADDITIONAL HISTORICAL DATA")
    print("=" * 80)
    print(
        f"Starting fetch for {len(weekday_dates)} weekday dates..."
    )

    results = fetch_additional_emiten_data(weekday_dates, csv_folder_path)

    process_additional_historical_data(csv_folder_path, args.processed_folder_path)

    print("\n" + "=" * 80)
    print("FETCH SUMMARY")
    print("=" * 80)
    success_count = 0
    failed_weekday_dates = []

    for weekday_dt, success, message in results:
        if success:
            success_count += 1
        else:
            failed_weekday_dates.append((weekday_dt, message))

    if failed_weekday_dates:
        print("Failed data fetch:")
        for weekday_dt, message in failed_weekday_dates:
            print(f"{weekday_dt}  - {message}")
    else:
        print("All weekday dates fetched successfully!")

    print("=" * 80)
    print(f"Fetched: {success_count}/{len(weekday_dates)} weekday dates")
    print("=" * 80)
