import argparse
from multiprocessing import Pool
from tqdm import tqdm
from utils.data.downloader import download_stock_data, append_df_to_csv


def fetch_emiten_data(args_tuple):
    """
    Fetch data for a single emiten.

    Args:
        args_tuple: Tuple containing (emiten, start_date, end_date, csv_folder_path)

    Returns:
        Tuple of (emiten, success, message)
    """
    emiten, start_date, end_date, csv_folder_path = args_tuple

    try:
        df = download_stock_data(emiten, start_date=start_date, end_date=end_date)

        if df is not None and not df.empty:
            csv_file_path = f"{csv_folder_path}/{emiten}.csv"
            append_df_to_csv(df, csv_file_path)
            return (emiten, True, f"Data for {emiten} saved to {csv_file_path}.")
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

    args = parser.parse_args()

    # read emiten list from specified file
    with open(args.file_name, "r") as f:
        emiten_list = f.read().splitlines()

    # Prepare arguments for multiprocessing
    fetch_args = [
        (emiten, args.start_date, args.end_date, args.csv_folder_path)
        for emiten in emiten_list
    ]

    # Use multiprocessing to fetch data in parallel
    print(
        f"Starting parallel fetch with {args.workers} workers for {len(emiten_list)} emitens..."
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
