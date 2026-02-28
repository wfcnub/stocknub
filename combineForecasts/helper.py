import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from camel_converter import to_camel

from utils.pipeline import get_label_config

def _get_combined_forecasts_features_target_threshold() -> (str, str, str):
    """
    (Internal Helper) Read the yaml file containing the feature columns, target column, and threshold column for modelling v4

    Returns:
        str: Feature columns consisted of the forecast columns
        str: Target column used in model training
        str: Threshold column used to determine the target column

    """
    columns_information_path = Path('data/combined_forecasts_columns_information.yaml')
    with open(columns_information_path, 'r') as file:
        columns_information = yaml.safe_load(file)
    
    feature_columns = columns_information['feature_columns']
    target_column = columns_information['target_column']
    threhsold_column = columns_information['threhsold_column']

    return feature_columns, target_column, threhsold_column

def _get_ticker_available_on_all_forecasts(label_types: list, rolling_windows: list) -> list:
    """
    (Internal Helper) Get all tickers that are available in all 3 model versions

    Args:
        label_types (list): A list containing all the types of label
        rolling_windows (list): A list of rolling window, the number of future days to look at for the label
    
    Returns:
        list: A list containing all tickers that are available in all 3 model versions
    """
    all_intersected_ticker = {}
    for model_version in [1, 2, 3]:
        for label_type in label_types:
            for window in rolling_windows:
                all_forecast_path = Path(f'data/stock/forecast/model_v{model_version}/{to_camel(label_type)}/{window}dd/').rglob('*.csv')
                all_ticker = set([file.stem for file in all_forecast_path])
                
                if len(all_intersected_ticker) == 0:
                    all_intersected_ticker = all_ticker
                else:
                    all_intersected_ticker = all_intersected_ticker.intersection(all_ticker)
    
    all_intersected_ticker = list(all_intersected_ticker)

    return all_intersected_ticker

def _combine_multiple_forecast_for_single_ticker(ticker: str, label_types: list, rolling_windows: list, model_versions: list) -> pd.DataFrame:
    """
    (Internal Helper) Combine all forecast into a single pandas dataframe

    Args:
        ticker (str): The name of the ticker to process
        label_types (list): A list containing all the types of label
        rolling_windows (list): A list of rolling window, the number of future days to look at for the label
        model_version (list): A list containing all the model versions
    
    Returns:
        pd.DataFrame: A pandas dataframe containing all the forecast columns as features, target column, and threshold column
    """
    forecast_df = pd.DataFrame()
    max_window = np.max(rolling_windows)

    for model_version in model_versions:
        for label_type in label_types:
            for window in rolling_windows:
                target_column, threshold_column, positive_label, _ = get_label_config(label_type, window)
                forecast_column = f'Forecast {positive_label} {window}dd'
                forecast_path = Path(f'data/stock/forecast/model_v{model_version}/{to_camel(label_type)}/{window}dd/{ticker}.csv')

                if (window == max_window) and (label_type == 'median_gain'):
                    temp_forecast_df = pd.read_csv(forecast_path, usecols=['Date', forecast_column, target_column, threshold_column])
                else:
                    temp_forecast_df = pd.read_csv(forecast_path, usecols=['Date', forecast_column])
                
                temp_forecast_df.rename(columns={forecast_column: f'{forecast_column} - V{model_version}'}, inplace=True)
                
                if len(forecast_df) == 0:
                    forecast_df = temp_forecast_df
                else:
                    forecast_df = pd.merge(forecast_df, temp_forecast_df, on='Date', how='inner', suffixes=['', '_drop'])
                    forecast_df.drop(columns=[col for col in forecast_df.columns if col.endswith('_drop')], inplace=True)
    
    feature_columns, target_column, threshold_column = _get_combined_forecasts_features_target_threshold()
    
    columns_order = ['Date'] + feature_columns + [threshold_column] + [target_column]
    forecast_df = forecast_df[columns_order]

    return forecast_df

def _write_combined_forecasts_features_target_threshold(label_types: list, rolling_windows: list, model_versions: list) -> None:
    """
    (Internal Helper) Writes the feature columns, target column, and threshold column available on the combined forecasting data

    Args:
        label_types (list): A list containing all the types of label
        rolling_windows (list): A list of rolling window, the number of future days to look at for the label
        model_version (list): A list containing all the model versions    
    """
    feature_columns = []
    for model_version in model_versions:
        for label_type in label_types:
            for window in rolling_windows:
                _, _, positive_label, _ = get_label_config(label_type, window)
                forecast_column = f'Forecast {positive_label} {window}dd - V{model_version}'
                feature_columns.append(forecast_column)
    
    target_column = f"Median Gain {np.max(rolling_windows)}dd"
    threhsold_column = f"Threshold Median Gain {np.max(rolling_windows)}dd"

    columns_information = {
        'feature_columns': feature_columns,
        'target_column': target_column,
        'threhsold_column': threhsold_column
    }

    columns_information_path = Path('data/combined_forecasts_columns_information.yaml')
    with open(columns_information_path, 'w') as file:
        yaml.dump(columns_information, file, default_flow_style=False, sort_keys=False)

    return