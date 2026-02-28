from pathlib import Path
from datetime import datetime, timedelta

from fetchOHLCVData.helper import (
    _fetch_ticker_data
)

def fetch_ticker_data(args_tuple):
    """
    Fetch OHLCV data for a single ticker

    Args:
        args_tuple (tuple): A Tuple containing (ticker, start_date, end_date, csv_folder_path)

    Returns:
        tuple: A Tuple of containing (ticker, success, message)
    """
    ticker, start_date, end_date, csv_folder_path = args_tuple

    try:
        df = _fetch_ticker_data(ticker, start_date=start_date, end_date=end_date)
    
        csv_file_path = (Path(csv_folder_path) / ticker).with_suffix('.csv')
        df.to_csv(csv_file_path, index=False)

        return (
            ticker, 
            True, 
            f"Succesfully fetched {ticker} ticker data"
        )
            
    except Exception as e:
        return (
            ticker, 
            False, 
            f"Failed fetching {ticker}: {str(e)}"
        )