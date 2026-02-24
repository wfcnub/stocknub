import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from curl_cffi import requests
from datetime import datetime, timedelta

def _fetch_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    (Internal Helper) fetch OHLCV ticker data from Yahoo Finance for a given ticker

    Args:
        ticker (str): A ticker for an emiten (e.g., 'BBCA')
        start_date (str): The start date for the data in 'YYYY-MM-DD' format
        end_date (str): The end date for the data in 'YYYY-MM-DD' format. If empty, the download will go up to the most recent date

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned historical stock data, or None if the download fails
    """
    session = requests.Session(impersonate="chrome123")
    ticker_yf = yf.Ticker(f"{ticker}.JK", session=session)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    
    if end_date != "":
        end = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end = datetime.now()

    data = ticker_yf.history(start=start, end=end)

    columns_to_drop = ["Dividends", "Stock Splits", "Capital Gains"]
    for col in columns_to_drop:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    data.reset_index(inplace=True)
    data["Date"] = data["Date"].dt.date
    
    return data