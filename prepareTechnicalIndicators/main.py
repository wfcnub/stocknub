import os
import numpy as np
import pandas as pd

from prepareTechnicalIndicators.all_technical_indicators import generate_all_technical_indicators

def process_single_ticker(args_tuple):
    """
    Read historical data, generate technical indicators, and save to technical indicators folder.

    Args:
        args_tuple: Tuple containing (emiten, historical_folder, technical_folder)

    Returns:
        Tuple of (emiten, success, message, num_new_rows)
    """
    emiten, historical_folder, technical_folder = args_tuple

    try:
        historical_path = f"{historical_folder}/{emiten}.csv"
        technical_path = f"{technical_folder}/{emiten}.csv"

        if not os.path.exists(historical_path):
            return (emiten, False, f"Historical data not found for {emiten}", 0)

        historical_df = pd.read_csv(historical_path)
        if historical_df.empty:
            return (emiten, False, f"Historical data is empty for {emiten}", 0)

        close_variance = np.var(historical_df["Close"].values)
        if close_variance < 1e-10:
            return (
                emiten,
                False,
                f"{emiten} - No price variation (likely suspended/delisted, variance={close_variance:.2e})",
                0,
            )

        historical_df["Date"] = pd.to_datetime(historical_df["Date"]).dt.strftime("%Y-%m-%d")
        technical_df = generate_all_technical_indicators(historical_df)
        technical_df.reset_index(inplace=True)
        technical_df["Date"] = pd.to_datetime(technical_df["Date"]).dt.strftime("%Y-%m-%d")
        
        technical_df.to_csv(technical_path, index=False)
        num_rows = len(technical_df)
        return (
            emiten,
            True,
            f"Generated {num_rows} rows of technical indicators for {emiten}",
            num_rows,
        )

    except Exception as e:
        return (emiten, False, f"Error processing {emiten}: {str(e)}", 0)