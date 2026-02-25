import os
import numpy as np
import pandas as pd
from pathlib import Path

from generateLabels.helper import _generate_labels_based_on_label_type

def process_single_ticker(args_tuple):
    """
    Read technical data, generate labels, and save to label folder

    Args:
        args_tuple: Tuple containing (ticker, technical_folder_path, labels_folder_path, target_column, rolling_windows, label_types)

    Returns:
        Tuple of (ticker, success, message, num_new_rows)
    """
    (
        ticker,
        technical_folder_path,
        labels_folder_path,
        target_column,
        rolling_windows,
        label_types
    ) = args_tuple

    try:
        technical_path = (Path(technical_folder_path) / ticker).with_suffix('.csv')
        labels_path = (Path(labels_folder_path) / ticker).with_suffix('.csv')
        
        if not technical_path.is_file():
            return (ticker, False, f"{ticker} - Technical file not found", 0)

        technical_df = pd.read_csv(technical_path)
        if technical_df.empty:
            return (ticker, False, f"{ticker} - Technical data file is empty", 0)

        close_variance = np.var(technical_df["Close"].tail(60).values)
        if close_variance < 1e-10:
            return (
                ticker,
                False,
                f"{ticker} - No price variation (likely suspended/delisted, variance={close_variance:.2e})",
                0,
            )

        technical_df["Date"] = pd.to_datetime(technical_df["Date"]).dt.strftime("%Y-%m-%d")
        labels_df = _generate_labels_based_on_label_type(technical_df, target_column, rolling_windows, label_types)

        if labels_df is None or labels_df.empty:
            return (
                ticker,
                False,
                f"{ticker} - Label generation returned empty dataframe (data quality issue: check for NaN/Inf values)",
                0,
            )

        labels_df.to_csv(labels_path, index=False)
        num_rows = len(labels_df)

        return (
            ticker,
            True,
            f"Generated {num_rows} rows of labels for {ticker}",
            num_rows,
        )

    except Exception as e:
        return (
            ticker, 
            False, 
            f"{ticker} - Exception: {str(e)}", 
            0
        )