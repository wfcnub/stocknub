import numpy as np
import pandas as pd

def _generate_median_loss(data: pd.DataFrame, target_column: str, rolling_window: int) -> (np.array, float):
    """
    (Internal Helper) Calculates the median loss of a target column based on a rolling window

    Args:
        data (pd.DataFrame): A pandas dataframe containing the daily price data
        target_column (str): The column name wished to be used as target data
        rolling_windows (int): The amount of upcoming target data used for creating the label

    Returns:
        np.array: The median loss for all target data
        float: The threshold for the quantile 0.9
    """
    median_close = data[target_column] \
                        [::-1] \
                        .rolling(rolling_window, closed='left') \
                        .quantile(0.4) \
                        [::-1]
    median_loss = (100 * (median_close - data[target_column].values) / data[target_column].values)

    if np.isnan(median_loss).all():
        threshold = np.nan
    else:
        test_median_loss_length = 120
        test_median_loss = median_loss[-1 * (test_median_loss_length + rolling_window): -rolling_window]
        threshold = np.nanquantile(test_median_loss, 0.1)

    return (median_loss, threshold)


def _bin_median_loss(threshold: float, val: float) -> str:
    """
    (Internal Helper) Perform binning for the median loss based on the threshold to create the labels for model development

    Args:
        threshold (float): The threshold for the quantile 0.9
        val (float): The median loss for the upcoming rolling window target variable

    Returns:
        str: Labels for developing the model
    """
    if np.isnan(val):
        return val
    if val <= threshold:
        return "High Loss"
    else:
        return "Low Loss"


def _generate_all_median_loss(data: pd.DataFrame, target_column: str, rolling_window: int) -> pd.DataFrame:
    """
    (Internal Helper) Generates a median loss label for each day based on a rolling window

    Args:
        data (pd.DataFrame): The input DataFrame containing stock data
        target_column (str): The name of the column to analyze
        rolling_window (int): The number of future days to look at for the median loss

    Returns:
        pd.DataFrame: The DataFrame with the new future median loss column added
    """
    column_name = f"Median Loss {rolling_window}dd"
    median_loss, threshold = _generate_median_loss(data, target_column, rolling_window)
    data[column_name] = [_bin_median_loss(threshold, val) for val in median_loss]
    data["Threshold " + column_name] = threshold

    return data
