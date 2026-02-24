import os
import glob
import numpy as np
import pandas as pd
from camel_converter import to_camel

from utils.pipeline import get_label_config

def _get_emiten_available_on_all_forecasts(label_types: list, rolling_windows: list) -> list:
    all_intersected_emiten = {}
    for model_version in [1, 2, 3]:
        for label_type in label_types:
            for window in rolling_windows:
                all_forecast_path = glob.glob(os.path.join(f'data/stock/forecast/model_v{model_version}/{to_camel(label_type)}/{window}dd/', "*.csv"))
                all_emiten = set([val.split('/')[-1].split('.')[0] for val in all_forecast_path])
    
                if len(all_intersected_emiten) == 0:
                    all_intersected_emiten = all_emiten
                else:
                    all_intersected_emiten = all_intersected_emiten.intersection(all_emiten)
    
    all_intersected_emiten = list(all_intersected_emiten)

    return all_intersected_emiten

def _combine_multiple_forecast_for_single_emiten(emiten: str, label_types: list, rolling_windows: list, model_versions: list) -> pd.DataFrame:
    forecast_df = pd.DataFrame()
    max_window = np.max(rolling_windows)

    for model_version in model_versions:
        for label_type in label_types:
            for window in rolling_windows:
                target_column, threshold_column, positive_label, _ = get_label_config(label_type, window)
                forecast_column = f'Forecast {positive_label} {window}dd' 
                forecast_path = f'data/stock/forecast/model_v{model_version}/{to_camel(label_type)}/{window}dd/{emiten}.csv'

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
    
    feature_columns = [col for col in forecast_df.columns if 'Forecast' in col]
    threshold_column = [col for col in forecast_df.columns if 'Threshold' in col]
    target_column = [col for col in forecast_df.columns if ('Forecast' not in col) and ('Threshold' not in col) and (col != 'Date')]

    columns_order = ['Date'] + feature_columns + threshold_column + target_column
    forecast_df = forecast_df[columns_order]

    return forecast_df

def _list_combined_forecasts_feature_columns(label_types: list, rolling_windows: list, model_versions: list):
    feature_columns = []
    for model_version in model_versions:
        for label_type in label_types:
            for window in rolling_windows:
                _, _, positive_label, _ = get_label_config(label_type, window)
                forecast_column = f'Forecast {positive_label} {window}dd - V{model_version}'
                feature_columns.append(forecast_column)
    
    return feature_columns

def _get_combined_forecasts_feature_columns():
    """
    Load the saved and generated stock's technical indicators
    Args:
        None

    Returns:
        list: A list containing all feature names for the technical indicators
    """
    feature_file = "data/forecast_features.txt"
    with open(feature_file, "r") as file:
        feature_columns = [line.strip() for line in file]

    return feature_columns
