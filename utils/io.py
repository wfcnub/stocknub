"""
I/O utility functions for file operations.

This module handles all file input/output operations including CSV reading and writing.
"""

import os
import pandas as pd


def get_last_date_from_csv(csv_file_path: str) -> str:
    """
    Get the exact last date from a CSV file.

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


def append_df_to_csv(df: pd.DataFrame, csv_file_path: str):
    """
    Append a DataFrame to a CSV file. If the file does not exist, it creates a new one.

    Args:
        df (pd.DataFrame): The DataFrame to append
        csv_file_path (str): The path to the CSV file
    """
    if not os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, index=False)
    else:
        df.to_csv(csv_file_path, mode="a", header=False, index=False)
