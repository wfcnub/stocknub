import os
import numpy as np
import pandas as pd
from pathlib import Path

from combineForecasts.helper import _combine_multiple_forecast_for_single_ticker

def process_single_ticker(args_tuple):
    """
    A process of combining all forecasting result from all variations of model versions, label types, and rolling windows into a single pandas dataframe

    Args:
        args_tuple: Tuple containing (ticker, label_types, rolling_windows, model_versions, post_process_forecast_pat)

    Returns:
        Tuple of (ticker, success, message, num_new_rows)
    """
    ticker, label_types, rolling_windows, model_versions, post_process_forecast_path = args_tuple

    try:
        forecast_df = _combine_multiple_forecast_for_single_ticker(ticker, label_types, rolling_windows, model_versions)
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.strftime("%Y-%m-%d")

        save_path = (Path(post_process_forecast_path) / ticker).with_suffix('.csv')
        forecast_df.to_csv(save_path, index=False)

        num_rows = len(forecast_df)
        return (
            ticker,
            True,
            f"Generated {num_rows} rows of forecasts probability for {ticker}",
            num_rows,
        )

    except Exception as e:
        return (
            ticker, 
            False, 
            f"Error processing {ticker}: {str(e)}", 
            0
        )