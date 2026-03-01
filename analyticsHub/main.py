import numpy as np
import pandas as pd
import case_conversion
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from analyticsHub.helper import (
    _get_chosen_performance_df,
    _generate_forecast_data_on_test_data,
    _generate_max_daily_profit,
    _generate_trading_simulation_df
)

@st.cache_data
def get_all_performances() -> pd.DataFrame:
    """
    Get the overview of the model performance

    Returns:
        pd.DataFrame: A pandas dataframe containing the overview of the model performance
    """
    model_versions = [1, 2, 3, 4]
    
    all_performance_paths = []
    for model_version in model_versions:
        model_performance_path = Path(f'data/stock/model_v{model_version}/performance')
        _ = [[all_performance_paths.append(f) for f in file.iterdir()] for file in model_performance_path.iterdir()]  

    model_version_mapping = {
        'model_v1': 'Specific Ticker Model',
        'model_v2': 'Specific Industry Model',
        'model_v3': 'IHSG Model',
        'model_v4': 'Ensemble of Specific Ticker, Specific Industry, and IHSG Model'
        
    }
    all_model_versions = [model_version_mapping[performance_path.parts[2]] for performance_path in all_performance_paths]
    all_label_types = [case_conversion.separate_words(performance_path.parts[-2]).title() for performance_path in all_performance_paths]
    all_windows = [performance_path.stem for performance_path in all_performance_paths]
    all_performance_df = [pd.read_csv(performance_path).describe() for performance_path in all_performance_paths]

    all_df = pd.DataFrame({
        'model_version': all_model_versions,
        'label_type': all_label_types,
        'window': all_windows,
        'performance_df': all_performance_df
    })

    all_df['model_identifier'] = [' - '.join(val) for val in all_df[['model_version', 'label_type', 'window']].values]

    return all_df

@st.cache_data
def get_daily_recommendations() -> (pd.DataFrame, str):
    """
    Get the daily recommendations

    Returns:
        (pd.DataFrame, str): A tuple containing the daily recommendations dataframe and the forecast date
    """
    final_performance = pd.read_csv(Path('data/stock/model_v4/performance/medianGain/10dd.csv'))
    
    forecast_paths = Path('data/stock/forecast/model_v4/medianGain/10dd').rglob('*.csv')
    all_ticker = [file.stem for file in Path('data/stock/forecast/model_v4/medianGain/10dd').rglob('*.csv')]
    
    forecast_df = pd.DataFrame()
    
    for ticker, file in zip(all_ticker, forecast_paths):
        temp_forecast_df = pd.read_csv(file, usecols=['Date', 'Forecast High Gain 10dd']).tail(1)
        temp_forecast_df['Ticker'] = ticker
    
        forecast_df = pd.concat((forecast_df, temp_forecast_df))
    
    forecast_df = pd.merge(
                        forecast_df,
                        final_performance[['Ticker', 'Test - Gini']],
                        on='Ticker',
                        how='inner'
                    ).sort_values('Forecast High Gain 10dd', ascending=False)
    
    assert forecast_df['Date'].nunique() == 1
    forecast_date = forecast_df['Date'].unique()[0]
    
    forecast_df.set_index('Ticker', inplace=True)
    forecast_df.drop(columns=['Date'], inplace=True)

    return forecast_df, forecast_date

@st.cache_data
def generate_trading_simulation_df() -> pd.DataFrame:
    """
    Generate the trading simulation data

    Returns:
        pd.DataFrame: A pandas dataframe containing the trading simulation data
    """
    forecast_df = _generate_forecast_data_on_test_data()
    label_df = _generate_max_daily_profit()

    trading_simulation_df = _generate_trading_simulation_df(forecast_df, label_df)

    return trading_simulation_df

@st.cache_data
def visualize_profit_distribution_for_each_forecast_threshold(trading_simulation_df) -> go.Figure:
    """
    Visualize the profit distribution for each forecast threshold

    Args:
        trading_simulation_df (pd.DataFrame): A pandas dataframe containing the trading simulation data

    Returns:
        go.Figure: A plotly figure containing the profit distribution for each forecast threshold
    """
    boxplot_df = pd.DataFrame()
    all_forecast_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
    
    for lower_forecast_thres, upper_forecast_thres in zip(all_forecast_thresholds[:-1], all_forecast_thresholds[1:]):
        thres_bool = np.all((
            trading_simulation_df['Forecast High Gain 10dd'] >= lower_forecast_thres,
            trading_simulation_df['Forecast High Gain 10dd'] < upper_forecast_thres
        ), axis=0)
    
        temp_boxplot_df = trading_simulation_df.loc[thres_bool, ['Profit']]
        temp_boxplot_df['Forecast Threshold'] = f'{lower_forecast_thres} <= x < {upper_forecast_thres}'
    
        boxplot_df = pd.concat((boxplot_df, temp_boxplot_df))
        
    fig = px.box(
        boxplot_df, 
        y="Forecast Threshold", 
        x="Profit",
        title="Profit's Distribution for Each Forecast Threhshold",
        points=False
    )

    fig.update_layout(
        xaxis=dict(
            title="Profit (%)",
            gridcolor='lightgrey'
        ),
        width=1000, 
        height=500)
    
    fig.add_vrect(x0=boxplot_df['Profit'].min()-0.1, x1=0, fillcolor="red", opacity=0.1, line_width=0)
    
    fig.add_vrect(x0=0, x1=boxplot_df['Profit'].max()+0.1, fillcolor="green", opacity=0.1, line_width=0)

    return fig

@st.cache_data
def visualize_impact_of_threshold_on_profit(trading_simulation_df) -> go.Figure:
    """
    Visualize the impact of threshold on profit

    Args:
        trading_simulation_df (pd.DataFrame): A pandas dataframe containing the trading simulation data

    Returns:
        go.Figure: A plotly figure containing the impact of threshold on profit
    """
    forecast_threshold_10dd = np.arange(0, 1, 0.00075)
    
    average_profit = [trading_simulation_df.loc[trading_simulation_df['Forecast High Gain 10dd'] >= threshold, 'Profit'].mean() for threshold in forecast_threshold_10dd]
    max_profit = [trading_simulation_df.loc[trading_simulation_df['Forecast High Gain 10dd'] >= threshold, 'Profit'].max() for threshold in forecast_threshold_10dd]
    min_profit = [trading_simulation_df.loc[trading_simulation_df['Forecast High Gain 10dd'] >= threshold, 'Profit'].min() for threshold in forecast_threshold_10dd]
    quantile_075_profit = [trading_simulation_df.loc[trading_simulation_df['Forecast High Gain 10dd'] >= threshold, 'Profit'].quantile(0.75) for threshold in forecast_threshold_10dd]
    quantile_05_profit = [trading_simulation_df.loc[trading_simulation_df['Forecast High Gain 10dd'] >= threshold, 'Profit'].quantile(0.5) for threshold in forecast_threshold_10dd]
    quantile_025_profit = [trading_simulation_df.loc[trading_simulation_df['Forecast High Gain 10dd'] >= threshold, 'Profit'].quantile(0.25) for threshold in forecast_threshold_10dd]

    fig = go.Figure()

    fig.update_layout(
        title="Profit's Statistic on Different Forecast Threhsold",
        xaxis=dict(
            title="Forecast Threshold",
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            title="Profit (%)",
            zerolinecolor='black'
        ),
        width=1500,
        height=700,
        template="plotly_white"
    )
    
    fig.add_trace(go.Scatter(x=forecast_threshold_10dd, y=average_profit, name='Average Profit', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_threshold_10dd, y=max_profit, name='Max Profit', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_threshold_10dd, y=min_profit, name='Min Profit', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_threshold_10dd, y=quantile_075_profit, name='Quantile 0.75 Profit', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_threshold_10dd, y=quantile_05_profit, name='Quantile 0.5 Profit', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_threshold_10dd, y=quantile_025_profit, name='Quantile 0.25 Profit', mode='lines'))

    fig.add_hrect(y0=np.min(min_profit)-10, y1=0, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=np.max(max_profit)+10, fillcolor="green", opacity=0.1, line_width=0)
    
    return fig