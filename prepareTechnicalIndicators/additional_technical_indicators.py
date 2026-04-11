import numpy as np
import pandas as pd

WINDOWS = [15, 60, 90]

def _feature_columns_for_window(n_window):
    return [
        f'Foreign to Domestic {n_window} Days Ownership',
        f'Foreign {n_window} Days Average Price Over Current Price',
        f'Domestic {n_window} Days Average Price Over Current Price',
    ]

def _calculate_additional_technical_indicators(data, n_window):
    """
    Calculate foreign/domestic flow indicators for a given rolling window.

    Args:
        data (pd.DataFrame): DataFrame containing OHLCV + Foreign Buy/Sell columns.
        n_window (int): Rolling window size in trading days.

    Returns:
        dict: Mapping of feature column name to its computed values (np.array).
    """
    data['Net Foreign'] = data['Foreign Buy'] - data['Foreign Sell']
    data['Net Domestic'] = data['Volume'] - (data['Foreign Buy'] + data['Foreign Sell'])
    data['Average Price'] = data[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    data['Domestic Accumulation'] = data['Average Price'] * data['Net Domestic']
    data['Foreign Accumulation'] = data['Average Price'] * data['Net Foreign']

    data[f'Foreign {n_window} Days Accumulation'] = data['Foreign Accumulation'].rolling(window=n_window).sum()
    data[f'Domestic {n_window} Days Accumulation'] = data['Domestic Accumulation'].rolling(window=n_window).sum()

    data[f'Foreign {n_window} Days Volume'] = data['Net Foreign'].rolling(window=n_window).sum()
    data[f'Domestic {n_window} Days Volume'] = data['Net Domestic'].rolling(window=n_window).sum()

    foreign_vol = data[f'Foreign {n_window} Days Volume']
    domestic_vol = data[f'Domestic {n_window} Days Volume']
    total_vol = foreign_vol + domestic_vol

    data[f'Foreign {n_window} Days Average Price'] = data[f'Foreign {n_window} Days Accumulation'] / foreign_vol.replace(0, np.nan)
    data[f'Domestic {n_window} Days Average Price'] = data[f'Domestic {n_window} Days Accumulation'] / domestic_vol.replace(0, np.nan)

    data[f'Foreign to Domestic {n_window} Days Ownership'] = foreign_vol / total_vol.replace(0, np.nan)
    data[f'Foreign {n_window} Days Average Price Over Current Price'] = (data[f'Foreign {n_window} Days Average Price'] - data['Average Price']) / data['Average Price'].replace(0, np.nan)
    data[f'Domestic {n_window} Days Average Price Over Current Price'] = (data[f'Domestic {n_window} Days Average Price'] - data['Average Price']) / data['Average Price'].replace(0, np.nan)

    for col in _feature_columns_for_window(n_window):
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)

    return {col: data[col].values for col in _feature_columns_for_window(n_window)}

def calculate_additional_technical_indicators(data):
    """
    Calculate foreign/domestic flow-based technical indicators across multiple windows.

    Computes ownership ratios and average price differentials for foreign vs domestic
    investors over 15, 60, and 90 day rolling windows.

    Args:
        data (pd.DataFrame): DataFrame with OHLCV + Foreign Buy/Sell columns (Date-indexed).

    Returns:
        pd.DataFrame: DataFrame containing only the computed feature columns.
    """
    all_feature_columns = []

    for window in WINDOWS:
        feature_values = _calculate_additional_technical_indicators(data.copy(), window)
        for col_name, values in feature_values.items():
            data[col_name] = values
            all_feature_columns.append(col_name)

    result_df = data[all_feature_columns]

    result_df = result_df.shift(1)

    return result_df