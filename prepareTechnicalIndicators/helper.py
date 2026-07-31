import pathlib
import tempfile

import numpy as np
import pandas as pd

def identify_historical_trends(data, column, rolling_window, make_bool_up=None, make_bool_down=None):
    """
    (Internal Helper) Identifies the historical trend of a data column using linear regression.

    This function calculates the slope of the data over a rolling window and classifies
    it as 'up', 'down', or 'sideways' based on the slope's value relative to a threshold.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to analyze.
        rolling_window (int): The number of periods to include in the trend calculation.
        make_bool_up (bool, optional): If True, returns 1.0 for positive slopes and 0.0 otherwise.
        make_bool_down (bool, optional): If True, returns 1.0 for negative slopes and 0.0 otherwise.

    Returns:
        np.array: An array of the classified trends for each period.
    """
    linreg_gradients = np.array(
        [
            np.nan
            if i < rolling_window
            else _retrieve_linreg_gradients(data[column].values[i - rolling_window : i])
            for i in range(len(data))
        ]
    )

    if make_bool_up:
        return np.array(
            [np.nan if np.isnan(val) else float(val > 0) for val in linreg_gradients]
        )
    elif make_bool_down:
        return np.array(
            [np.nan if np.isnan(val) else float(val < 0) for val in linreg_gradients]
        )
    return linreg_gradients


def _retrieve_linreg_gradients(target_data):
    """
    Calculates the slope (gradient) of a data series using an analytical least-squares formula.

    This function takes a series of data points, scales them to a 0-1 range,
    and computes the slope of the best-fit line using the closed-form solution.
    This avoids the overhead of instantiating MinMaxScaler and LinearRegression
    objects on every call.

    Args:
        target_data (np.array): An array of numerical data.

    Returns:
        float: The calculated slope of the linear regression line. Returns 0
               if there are not enough data points or if the data has no variation.
    """
    target_data = target_data[~np.isnan(target_data)]

    n = len(target_data)
    if n <= 1:
        return 0

    d_min = target_data.min()
    d_max = target_data.max()
    if d_max == d_min:
        return 0
    y = (target_data - d_min) / (d_max - d_min)
 
    x = np.linspace(0, n, n)

    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    slope = np.dot(x_centered, y - y_mean) / np.dot(x_centered, x_centered)

    return slope


def safe_divide(numerator, denominator):
    """
    Divide two aligned numeric objects without producing infinite values.

    Zero denominators are treated as unavailable observations.  The helper
    intentionally preserves pandas indexes so derived features remain aligned
    by trading date.
    """
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None):
    """Return a causal rolling z-score using only the current and prior rows."""
    min_periods = min_periods or window
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return safe_divide(series - rolling_mean, rolling_std)


def write_technical_indicator_features(feature_columns):
    """Atomically write the generated feature manifest once per pipeline run."""
    from utils import paths

    output_path = paths.get_technical_indicator_features_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_path.parent,
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        for feature_column in sorted(feature_columns):
            tmp_file.write(f"{feature_column}\n")
        tmp_path = tmp_file.name

    pathlib.Path(tmp_path).replace(output_path)


def get_all_technical_indicators():
    """
    Load the saved and generated stock's technical indicators.

    Returns:
        list: A list containing all feature names for the technical indicators.
    """
    from utils import paths

    feature_file = paths.get_technical_indicator_features_path()
    with open(feature_file, "r") as file:
        feature_columns = [line.strip() for line in file]

    return feature_columns
