import numpy as np
import pandas as pd
import case_conversion
import streamlit as st
from pathlib import Path
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

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


def _generate_forecast_data_on_test_data() -> pd.DataFrame:
    """
    (Internal Helper) Generate the forecast data on the test data

    Returns:
        pd.DataFrame: A pandas dataframe containing the forecast data on the test data
    """
    forecast_paths = Path('data/stock/forecast/model_v4/medianGain/10dd').rglob('*.csv')
    all_ticker = [file.stem for file in Path('data/stock/forecast/model_v4/medianGain/10dd').rglob('*.csv')]
    
    forecast_df = pd.DataFrame()
    
    for ticker, file in zip(all_ticker, forecast_paths):
        temp_forecast_df = pd.read_csv(file, usecols=['Date', 'Forecast High Gain 10dd']).dropna().tail(90).head(80)
        temp_forecast_df['Ticker'] = ticker
    
        forecast_df = pd.concat((forecast_df, temp_forecast_df))
    
    final_performance = pd.read_csv(Path('data/stock/model_v4/performance/medianGain/10dd.csv'))
    
    forecast_df = pd.merge(
        forecast_df,
        final_performance[['Ticker', 'Test - Gini']],
        on='Ticker',
        how='inner'
    )

    return forecast_df

def _generate_max_daily_profit() -> pd.DataFrame:
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
        temp_label_df['Max Close 10dd'] = temp_label_df['Close'] \
                                                    [::-1] \
                                                    .rolling(10, closed='left') \
                                                    .max() \
                                                    [::-1]
    
        label_df = pd.concat((label_df, temp_label_df))

    return label_df

def _generate_trading_simulation_df(forecast_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """
    (Internal Helper) Generate the trading simulation data

    Args:
        forecast_df (pd.DataFrame): A pandas dataframe containing the forecast data
        label_df (pd.DataFrame): A pandas dataframe containing the label data

    Returns:
        pd.DataFrame: A pandas dataframe containing the trading simulation data
    """
    trading_simulation_df = pd.merge(
                                forecast_df,
                                label_df,
                                on=['Ticker', 'Date'],
                                how='inner'
                            )
    
    trading_simulation_df['Profit'] = 100 * (trading_simulation_df['Max Close 10dd'] - trading_simulation_df['Close']) / trading_simulation_df['Close']

    return trading_simulation_df

def _get_testing_data_date() -> (str, str):
    """
    (Internal Helper) Get the testing data date

    Returns:
        (str, str): A tuple containing the start and end testing market dates
    """
    active_market_dates = np.sort([datetime.strptime(file.stem, '%Y%m%d').strftime('%Y-%m-%d') for file in Path('data/stock/raw_foreign_flow_non_regular').rglob('*.csv')])
    testing_market_dates = active_market_dates[-90:-10]

    start_testing_market_date = testing_market_dates[0]
    end_testing_market_date = testing_market_dates[-1]

    return start_testing_market_date, end_testing_market_date