import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from fetchOHLCVData.main import fetch_ticker_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Description: Fetch Ticker's Open, High, Low, Close, and Volume (OHLCV) Historical Data using yfinance"
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="2021-01-01",
        help="Start date in YYYY-MM-DD format (default: 2021-01-01)",
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
        default="data/ticker_list.txt",
        help="Path to the file containing a list all ticker (default: data/ticker_list.txt)",
    )

    parser.add_argument(
        "--csv_folder_path",
        type=str,
        default="data/stock/OHLCV",
        help="Directory path where the CSV files will be saved (default: data/stock/OHLCV)",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers to use (default: CPU count)",
    )

    args = parser.parse_args()
    
    if Path(args.csv_folder_path).exists():
        shutil.rmtree(args.csv_folder_path)

    Path(args.csv_folder_path).mkdir(parents=True, exist_ok=True)
        
    with open(args.file_name, "r") as f:
        ticker_list = f.read().splitlines()

    fetch_args = [
        (ticker, args.start_date, args.end_date, args.csv_folder_path)
        for ticker in ticker_list
    ]

    print("=" * 80)
    print("PIPELINE DESCRIPTION: FETCH OHLCV DATA")
    print("=" * 80)
    print(f"Starting parallel fetch with {args.workers} workers for {len(ticker_list)} tickers")
    print()

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(fetch_ticker_data, fetch_args),
                total=len(fetch_args),
                desc="Fetching ticker's OHLCV data",
                unit="ticker",
            )
        )

    print("\n" + "=" * 80)
    print("FETCH SUMMARY")
    print("=" * 80)

    success_count = 0
    failed_tickers = []

    for ticker, success, message in results:
        if success:
            success_count += 1
        else:
            failed_tickers.append((ticker, message))

    if failed_tickers:
        print("Failed fetch:")
        for ticker, message in failed_tickers:
            print(f"{ticker} - {message}")
    else:
        print("All tickers fetched successfully!")

    print("=" * 80)
    print(f"Fetched: {success_count}/{len(ticker_list)} tickers")
    print("=" * 80)
