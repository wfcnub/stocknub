import os
import numpy as np
import pandas as pd

from generateLabels.helper import _generate_labels_based_on_label_type

def process_single_ticker(args_tuple):
    """
    Read technical data, generate labels, and save to label folder.

    Args:
        args_tuple: Tuple containing (emiten, technical_folder, labels_folder,
                    target_column, rolling_windows, label_types)

    Returns:
        Tuple of (emiten, success, message, num_new_rows)
    """
    (
        emiten,
        technical_folder,
        labels_folder,
        target_column,
        rolling_windows,
        label_types
    ) = args_tuple

    try:
        technical_path = f"{technical_folder}/{emiten}.csv"
        labels_path = f"{labels_folder}/{emiten}.csv"

        if not os.path.exists(technical_path):
            return (emiten, False, f"{emiten} - Technical file not found", 0)

        technical_df = pd.read_csv(technical_path)
        if technical_df.empty:
            return (emiten, False, f"{emiten} - Technical data file is empty", 0)

        close_variance = np.var(technical_df["Close"].values)
        if close_variance < 1e-10:
            return (
                emiten,
                False,
                f"{emiten} - No price variation (likely suspended/delisted, variance={close_variance:.2e})",
                0,
            )

        technical_df["Date"] = pd.to_datetime(technical_df["Date"]).dt.strftime("%Y-%m-%d")
        labels_df = _generate_labels_based_on_label_type(technical_df, target_column, rolling_windows, label_types)

        if labels_df is None or labels_df.empty:
            return (
                emiten,
                False,
                f"{emiten} - Label generation returned empty dataframe (data quality issue: check for NaN/Inf values)",
                0,
            )

        labels_df.to_csv(labels_path, index=False)
        num_rows = len(labels_df)

        return (
            emiten,
            True,
            f"Generated {num_rows} rows of labels for {emiten}",
            num_rows,
        )

    except Exception as e:
        return (
            emiten, 
            False, 
            f"{emiten} - Exception: {str(e)}", 
            0
        )