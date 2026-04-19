import numpy as np
import pandas as pd
import case_conversion
import streamlit as st
from pathlib import Path
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

from utils.pipeline import (
    get_split_dates, 
    get_split_masks
)

def _get_chosen_performance_df(all_df: pd.DataFrame, chosen_model_versions: list, chosen_model_label_types: list, chosed_model_windows: list) -> (list, list):
    """
    (Internal Helper) Get the selected overview of the model performance based on the user's selection

    Args:
        all_df (pd.DataFrame): A pandas dataframe containing the overview of the model performance
        chosen_model_versions (list): A list of chosen model versions
        chosen_model_label_types (list): A list of chosen label types
        chosed_model_windows (list): A list of chosen windows

    Returns:
        (list, list): A tuple containing the selected model identifiers and performance dataframes
    """
    filter_bool = np.all((
        all_df['model_version'].isin(chosen_model_versions),
        all_df['label_type'].isin(chosen_model_label_types),
        all_df['window'].isin(chosed_model_windows)
    ), axis=0)

    selected_model_identifier = all_df.loc[filter_bool, 'model_identifier'].values.tolist()
    selected_performance_df = all_df.loc[filter_bool, 'performance_df'].values.tolist()
    
    return selected_model_identifier, selected_performance_df

def _generate_score_data_on_test_data(rolling_window: str) -> pd.DataFrame:
    """
    (Internal Helper) Generate the score data on the test data

    Returns:
        pd.DataFrame: A pandas dataframe containing the score data on the test data
    """
    score_paths = Path(f'data/stock/score/{rolling_window}').rglob('*.csv')
    all_ticker = [file.stem for file in Path(f'data/stock/score/{rolling_window}').rglob('*.csv')]
    
    test_score_df = pd.DataFrame()
    
    for ticker, file in zip(all_ticker, score_paths):
        splits = get_split_dates(f'Median Gain {rolling_window}')
        temp_score_df = pd.read_csv(file, usecols=['Date', f'Score {rolling_window}'])
        _, _, _, test_mask, _ = get_split_masks(temp_score_df, splits)
        
        temp_test_score_df = temp_score_df.loc[test_mask]
        temp_test_score_df['Ticker'] = ticker
    
        test_score_df = pd.concat((test_score_df, temp_test_score_df))
    
    final_performance = pd.read_csv(Path(f'data/stock/model_v4/performance/medianGain/{rolling_window}.csv'))
    
    test_score_df = pd.merge(
        test_score_df,
        final_performance[['Ticker', 'Test - Gini']],
        on='Ticker',
        how='inner'
    )

    return test_score_df

def _generate_max_daily_performance_metric(rolling_window: str, performance_metric: str) -> pd.DataFrame:
    """
    (Internal Helper) Generate the max daily profit data

    Returns:
        pd.DataFrame: A pandas dataframe containing the max daily profit data
    """
    label_paths = Path('data/stock/label').rglob('*.csv')
    all_ticker = [file.stem for file in Path('data/stock/label').rglob('*.csv')]
    
    label_df = pd.DataFrame()
    
    for ticker, file in zip(all_ticker, label_paths):
        temp_label_df = pd.read_csv(file, usecols=['Date', 'Close'])
        temp_label_df['Ticker'] = ticker

        if performance_metric == 'Profit':
            temp_label_df[f'Max Close {rolling_window}'] = temp_label_df['Close'] \
                                                        [::-1] \
                                                        .rolling(int(rolling_window[:-2]), closed='left') \
                                                        .max() \
                                                        [::-1]
        elif performance_metric == 'Loss':
            temp_label_df[f'Min Close {rolling_window}'] = temp_label_df['Close'] \
                                                    [::-1] \
                                                    .rolling(int(rolling_window[:-2]), closed='left') \
                                                    .min() \
                                                    [::-1]
    
        label_df = pd.concat((label_df, temp_label_df))

    return label_df

def _generate_trading_simulation_df(score_df: pd.DataFrame, max_daily_profit_df: pd.DataFrame, max_daily_loss_df: pd.DataFrame, rolling_window: str) -> pd.DataFrame:
    """
    (Internal Helper) Generate the trading simulation data

    Args:
        score_df (pd.DataFrame): A pandas dataframe containing the score data
        max_daily_profit_df (pd.DataFrame): A pandas dataframe containing the max daily profit data
        max_daily_loss_df (pd.DataFrame): A pandas dataframe containing the max daily loss data

    Returns:
        pd.DataFrame: A pandas dataframe containing the trading simulation data
    """
    trading_simulation_df = pd.merge(
                                        pd.merge(
                                                    score_df,
                                                    max_daily_profit_df,
                                                    on=['Ticker', 'Date'],
                                                    how='inner' 
                                                ),
                                        max_daily_loss_df.drop(columns=['Close']),
                                        on=['Ticker', 'Date'],
                                        how='inner'
                                    )
    
    trading_simulation_df['Profit'] = 100 * (trading_simulation_df[f'Max Close {rolling_window}'] - trading_simulation_df['Close']) / trading_simulation_df['Close']
    trading_simulation_df['Loss'] = 100 * (trading_simulation_df[f'Min Close {rolling_window}'] - trading_simulation_df['Close']) / trading_simulation_df['Close']

    return trading_simulation_df