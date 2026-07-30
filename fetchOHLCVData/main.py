import time
from utils import paths

from fetchOHLCVData.helper import (
    _fetch_ticker_data
)

def fetch_ticker_data(args_tuple):
    """
    Fetch OHLCV data for a single ticker

    Args:
        args_tuple (tuple): A Tuple containing (ticker, start_date, end_date)

    Returns:
        tuple: A Tuple of containing (ticker, success, message)
    """
    ticker, start_date, end_date = args_tuple

    try:
        for _ in range(3):
            try:
                df = _fetch_ticker_data(ticker, start_date=start_date, end_date=end_date)
            
                csv_file_path = paths.get_ohlcv_path(ticker)
                df.to_csv(csv_file_path, index=False)

                return (
                    ticker, 
                    True, 
                    f"Succesfully fetched {ticker} ticker data"
                )
            except:
                time.sleep(10)
            
            raise Exception(f"Failed to fetch {ticker} ticker data after 3 attempts")
            
    except Exception as e:
        return (
            ticker, 
            False, 
            f"Failed fetching {ticker}: {str(e)}"
        )