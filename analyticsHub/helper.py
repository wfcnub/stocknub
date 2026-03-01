import numpy as np
import pandas as pd
import case_conversion
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def _get_chosen_performance_df(all_df, chosen_model_versions, chosen_model_label_types, chosed_model_windows):
    filter_bool = np.all((
        all_df['model_version'].isin(chosen_model_versions),
        all_df['label_type'].isin(chosen_model_label_types),
        all_df['window'].isin(chosed_model_windows)
    ), axis=0)

    selected_model_identifier = all_df.loc[filter_bool, 'model_identifier'].values
    selected_performance_df = all_df.loc[filter_bool, 'performance_df'].values
    
    return selected_model_identifier, selected_performance_df


def _generate_forecast_data_on_test_data():
    forecast_paths = Path('data/stock/forecast/model_v4/medianGain/10dd').rglob('*.csv')
    all_ticker = [file.stem for file in Path('data/stock/forecast/model_v4/medianGain/10dd').rglob('*.csv')]
    
    forecast_df = pd.DataFrame()
    
    for ticker, file in zip(all_ticker, forecast_paths):
        temp_forecast_df = pd.read_csv(file, usecols=['Date', 'Forecast High Gain 10dd']).dropna().tail(110).head(100)
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

def _generate_max_daily_profit():
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

def _generate_trading_simulation_df(forecast_df, label_df):
    trading_simulation_df = pd.merge(
                                forecast_df,
                                label_df,
                                on=['Ticker', 'Date'],
                                how='inner'
                            )
    
    trading_simulation_df['Profit'] = 100 * (trading_simulation_df['Max Close 10dd'] - trading_simulation_df['Close']) / trading_simulation_df['Close']

    return trading_simulation_df

def generate_trading_simulation_df():
    forecast_df = _generate_forecast_data_on_test_data()
    label_df = _generate_max_daily_profit()

    trading_simulation_df = _generate_trading_simulation_df(forecast_df, label_df)

    return trading_simulation_df

