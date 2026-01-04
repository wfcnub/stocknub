import os
import pandas as pd
import yfinance as yf
from curl_cffi import requests
from datetime import datetime, timedelta

def _download_stock_data(emiten: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    (Internal Helper) Download historical stock data from Yahoo Finance for a given emiten.

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

def _get_last_date_from_csv(csv_file_path: str) -> str:
    """
    (Internal Helper) Get the exact last date from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        str: Exact last date in 'YYYY-MM-DD' format, or empty string if file doesn't exist
    """
    if not os.path.isfile(csv_file_path):
        return ""

    try:
        df = pd.read_csv(csv_file_path)
        if df.empty:
            return ""
        return str(df.iloc[-1]["Date"])
    except Exception as e:
        print(f"Warning: Could not read last date from {csv_file_path}: {str(e)}")
        return ""


def _append_df_to_csv(df: pd.DataFrame, csv_file_path: str):
    """
    (Internal Helper) Append a DataFrame to a CSV file. If the file does not exist, it creates a new one.

    Args:
        df (pd.DataFrame): The DataFrame to append
        csv_file_path (str): The path to the CSV file
    """
    if not os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, index=False)
    else:
        df.to_csv(csv_file_path, mode="a", header=False, index=False)


def _get_yesterday_date():
    """
    (Internal Helper) Get yesterday's date in YYYY-MM-DD format.

    Returns:
        str: Yesterday's date in 'YYYY-MM-DD' format
    """
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")