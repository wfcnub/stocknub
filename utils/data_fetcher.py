"""
Data fetching utility functions.

This module handles fetching stock data from external sources (Yahoo Finance).
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from curl_cffi import requests


def download_stock_data(emiten: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance for a given emiten.

    Args:
        emiten (str): The stock emiten symbol (e.g., 'BBCA')
        start_date (str): The start date for the data in 'YYYY-MM-DD' format
                          If empty, the download will start from the earliest available date
        end_date (str): The end date for the data in 'YYYY-MM-DD' format
                        If empty, the download will go up to the most recent date

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned historical stock data,
                      or None if the download fails
    """
    session = requests.Session(impersonate="chrome123")
    ticker = yf.Ticker(f"{emiten}.JK", session=session)

    start = (
        datetime.strptime(start_date, "%Y-%m-%d")
        if start_date != ""
        else datetime.strptime("2021-01-01", "%Y-%m-%d")
    )
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date != "" else datetime.now()
    data = ticker.history(start=start, end=end)

    columns_to_drop = ["Dividends", "Stock Splits", "Capital Gains"]
    for col in columns_to_drop:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    data.reset_index(inplace=True)

    try:
        data["Date"] = data["Date"].dt.date
    except Exception:
        pass

    return data
