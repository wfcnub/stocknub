import os
import numpy as np
import pandas as pd

from combineForecasts.helper import _combine_multiple_forecast_for_single_emiten

def process_single_ticker(args_tuple):
    """
    """
    emiten, label_types, rolling_windows, model_versions, post_process_forecast_path = args_tuple

    try:
        forecast_df = _combine_multiple_forecast_for_single_emiten(emiten, label_types, rolling_windows, model_versions)
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.strftime("%Y-%m-%d")

        save_path = f'{post_process_forecast_path}/{emiten}.csv'
        forecast_df.to_csv(save_path, index=False)

        num_rows = len(forecast_df)

        return (
            emiten,
            True,
            f"Generated {num_rows} rows of forecasts probability for {emiten}",
            num_rows,
        )

    except Exception as e:
        return (
            emiten, 
            False, 
            f"Error processing {emiten}: {str(e)}", 
            0
        )