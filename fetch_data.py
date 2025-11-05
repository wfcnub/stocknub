import argparse
from multiprocessing import Pool
from datetime import datetime, timedelta
from tqdm import tqdm
from utils.data.downloader import (
    download_stock_data,
    append_df_to_csv,
    get_last_date_from_csv,
)


def get_yesterday_date():
    """
    Get yesterday's date in YYYY-MM-DD format.

    Returns:
        str: Yesterday's date in 'YYYY-MM-DD' format
    """
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def fetch_emiten_data(args_tuple):
    """
    Fetch data for a single emiten.

    Args:
        args_tuple: Tuple containing (emiten, start_date, end_date, csv_folder_path, update_mode)

    Returns:
        Tuple of (emiten, success, message)
    """
    emiten, start_date, end_date, csv_folder_path, update_mode = args_tuple

    try:
        csv_file_path = f"{csv_folder_path}/{emiten}.csv"

        # If update mode is enabled, get the last date from the CSV
        if update_mode:
            last_date = get_last_date_from_csv(csv_file_path)
            if last_date:
                start_date = last_date
            # If no last date found (file doesn't exist or is empty), use provided start_date

        df = download_stock_data(emiten, start_date=start_date, end_date=end_date)

        if df is not None and not df.empty:
            append_df_to_csv(df, csv_file_path)
            date_range = f"from {start_date or 'earliest'} to {end_date or 'today'}"
            return (
                emiten,
                True,
                f"Data for {emiten} {date_range} saved to {csv_file_path}.",
            )
        else:
            return (emiten, False, f"No data found for {emiten}.")
    except Exception as e:
        return (emiten, False, f"Error fetching {emiten}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock data from Yahoo Finance")
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
        default="data/stock/emiten_list.txt",
        help="Path to the file containing emiten list (default: data/stock/emiten_list.txt)",
    )
    parser.add_argument(
        "--csv_folder_path",
        type=str,
        default="data/stock/historical",
        help="Directory path where CSV files will be saved (default: data/stock/historical)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers to use (default: 10)",
    )
    parser.add_argument(
        "--update",
        type=str,
        choices=["today", "yesterday"],
        default=None,
        help="Update existing data by reading the last date from each CSV file. Use 'today' to fetch until today, 'yesterday' to fetch until yesterday (useful for avoiding incomplete today's data)",
    )

    args = parser.parse_args()

    # Handle update mode and set end_date based on the update option
    if args.update:
        if args.update == "yesterday" and not args.end_date:
            args.end_date = get_yesterday_date()
        # For "today", end_date remains empty (default behavior)

    # read emiten list from specified file
    with open(args.file_name, "r") as f:
        emiten_list = f.read().splitlines()

    # Prepare arguments for multiprocessing
    fetch_args = [
        (emiten, args.start_date, args.end_date, args.csv_folder_path, args.update)
        for emiten in emiten_list
    ]

    # Use multiprocessing to fetch data in parallel
    mode_str = f"UPDATE mode (until {args.update})" if args.update else "FETCH mode"
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

    # Print all results
    print("\n" + "=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)
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

    print("=" * 60)
    print(f"Fetched: {success_count}/{len(emiten_list)} emitens")
    print("=" * 60)
