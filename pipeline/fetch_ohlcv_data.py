import argparse
import json
import os
import shutil
from multiprocessing import Pool, cpu_count
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

from fetchOHLCVData.main import fetch_ticker_data

from utils import paths


ORGANIZER_TRADING_WINDOWS = ("10dd", "5dd")


def fetch_organizer_tickers():
    organizer_base_url = os.getenv("ORGANIZER_BASE_URL")
    if not organizer_base_url:
        raise RuntimeError(
            "ORGANIZER_BASE_URL must be set when --process_selected_ticker is used"
        )

    tickers = []
    for trading_window in ORGANIZER_TRADING_WINDOWS:
        query = urlencode({"trading_window": trading_window})
        url = f"{organizer_base_url.rstrip('/')}/stocks?{query}"

        try:
            with urlopen(url, timeout=10) as response:
                response_tickers = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Failed to fetch organizer tickers from {url}: {exc}"
            ) from exc

        if not isinstance(response_tickers, list) or not all(
            isinstance(ticker, str) for ticker in response_tickers
        ):
            raise RuntimeError(
                f"Invalid organizer response from {url}: expected a list of ticker strings"
            )

        tickers.extend(ticker.strip() for ticker in response_tickers if ticker.strip())

    return tickers


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
        default=str(paths.get_ticker_list_path()),
        help="Path to the file containing a list all ticker",
    )

    parser.add_argument(
        "--csv_folder_path",
        type=str,
        default=str(paths.get_ohlcv_dir()),
        help="Directory path where the CSV files will be saved",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers to use (default: CPU count)",
    )

    parser.add_argument(
        "--process_selected_ticker",
        dest='process_selected_ticker', 
        action='store_true',
        help="A boolean enusring that the tickers being processed are just the selected ones",
    )

    parser.set_defaults(process_selected_ticker=False)

    args = parser.parse_args()
    
    if paths.get_ohlcv_dir().exists():
        shutil.rmtree(paths.get_ohlcv_dir())

    paths.get_ohlcv_dir().mkdir(parents=True, exist_ok=True)

    with open(args.file_name, "r") as f:
        ticker_list = f.read().splitlines()
    
    if args.process_selected_ticker:
        selected_ticker_to_process_df = pd.read_csv(
            paths.get_selected_ticker_and_industry_list_path()
        )
        selected_tickers = selected_ticker_to_process_df['Ticker'].values

        organizer_tickers = fetch_organizer_tickers()
        selected_ticker_set = set(selected_tickers)
        ticker_list = [ticker for ticker in ticker_list if ticker in selected_ticker_set]
        ticker_list = list(dict.fromkeys(ticker_list + organizer_tickers))

        
    fetch_args = [(ticker, args.start_date, args.end_date) for ticker in ticker_list]

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

        pool.close()
        pool.join()

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
