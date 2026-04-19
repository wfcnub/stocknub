import json
import numpy as np
import pandas as pd
import case_conversion
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from analyticsHub.helper import (
    _get_chosen_performance_df,
    _generate_score_data,
    _generate_close_data,
    _generate_buy_sell_percentage_data,
    _generate_recommendation_data
)

@st.cache_data
def get_pre_market_outlook() -> dict | None:
    """
    Load the most recent pre-market outlook JSON from data/pre_market_outlook/.

    Returns:
        dict | None: The parsed outlook dictionary, or None if no file exists
    """
    json_file_path = Path('data/pre_market_outlook.json')
    with open(json_file_path, "r") as f:
        return json.load(f)

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
def get_daily_recommendations(rolling_window: str) -> (pd.DataFrame, str):
    """
    Get the daily recommendations

    Returns:
        (pd.DataFrame, str): A tuple containing the daily recommendations dataframe and the forecast date
    """
    score_df, score_date = _generate_score_data(rolling_window)
    all_close_df = _generate_close_data()
    buy_percentage, sell_percentage = _generate_buy_sell_percentage_data(rolling_window)
    recommendation_df = _generate_recommendation_data(score_df, all_close_df, buy_percentage, sell_percentage, rolling_window)

    selected_recommendation_df = recommendation_df.loc[:, ['Ticker', f'Score {rolling_window}', 'Target Buy Price', 'Target Sell Price']] \
                                                    .set_index('Ticker') \
                                                    .sort_values(f'Score {rolling_window}', ascending=False)                        
    return selected_recommendation_df, score_date

@st.cache_data
def visualize_performance_metric_distribution_for_each_forecast_threshold(trading_simulation_df: pd.DataFrame, rolling_window: str, performance_metric: str) -> go.Figure:
    """
    Visualize the performance metric distribution for each forecast threshold

    Args:
        trading_simulation_df (pd.DataFrame): A pandas dataframe containing the trading simulation data

    Returns:
        go.Figure: A plotly figure containing the performance metric distribution for each forecast threshold
    """
    boxplot_df = pd.DataFrame()
    all_score_thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1]
    
    for lower_score_thres, upper_score_thres in zip(all_score_thresholds[:-1], all_score_thresholds[1:]):
        thres_bool = np.all((
            trading_simulation_df[f'Score {rolling_window}'] >= lower_score_thres,
            trading_simulation_df[f'Score {rolling_window}'] < upper_score_thres
        ), axis=0)
    
        temp_boxplot_df = trading_simulation_df.loc[thres_bool, [performance_metric]]
        temp_boxplot_df['Score Threshold'] = f'{lower_score_thres} <= x < {upper_score_thres}'
    
        boxplot_df = pd.concat((boxplot_df, temp_boxplot_df))
        
    fig = px.box(
        boxplot_df, 
        y="Score Threshold", 
        x=performance_metric,
        title=f"{performance_metric}'s Distribution for Each Score Threshold",
        points=False
    )

    fig.update_layout(
        xaxis=dict(
            title="Percentage (%)",
            gridcolor='lightgrey'
        ),
        width=1000, 
        height=500)
    
    fig.add_vrect(x0=boxplot_df[performance_metric].min()-0.1, x1=0, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_vrect(x0=0, x1=boxplot_df[performance_metric].max()+0.1, fillcolor="green", opacity=0.1, line_width=0)

    return fig

@st.cache_data
def visualize_impact_of_threshold_on_performance_metric(trading_simulation_df: pd.DataFrame, rolling_window: str, performance_metric: str) -> go.Figure:
    """
    Visualize the impact of threshold on performance metric

    Args:
        trading_simulation_df (pd.DataFrame): A pandas dataframe containing the trading simulation data

    Returns:
        go.Figure: A plotly figure containing the impact of threshold on performance metric
    """
    forecast_threshold = np.arange(0, 1, 0.00075)
    
    average_performance = [trading_simulation_df.loc[trading_simulation_df[f'Score {rolling_window}'] >= threshold, performance_metric].mean() for threshold in forecast_threshold]
    max_performance = [trading_simulation_df.loc[trading_simulation_df[f'Score {rolling_window}'] >= threshold, performance_metric].max() for threshold in forecast_threshold]
    min_performance = [trading_simulation_df.loc[trading_simulation_df[f'Score {rolling_window}'] >= threshold, performance_metric].min() for threshold in forecast_threshold]
    quantile_075_performance = [trading_simulation_df.loc[trading_simulation_df[f'Score {rolling_window}'] >= threshold, performance_metric].quantile(0.75) for threshold in forecast_threshold]
    quantile_05_performance = [trading_simulation_df.loc[trading_simulation_df[f'Score {rolling_window}'] >= threshold, performance_metric].quantile(0.5) for threshold in forecast_threshold]
    quantile_025_performance = [trading_simulation_df.loc[trading_simulation_df[f'Score {rolling_window}'] >= threshold, performance_metric].quantile(0.25) for threshold in forecast_threshold]

    fig = go.Figure()

    fig.update_layout(
        title=f"{performance_metric}'s Statistic on Different Score Threshold",
        xaxis=dict(
            title="Score Threshold",
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            title=f"Percentage (%)",
            zerolinecolor='black'
        ),
        width=1500,
        height=700,
        template="plotly_white"
    )
    
    fig.add_trace(go.Scatter(x=forecast_threshold, y=average_performance, name=f'Average {performance_metric}', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_threshold, y=max_performance, name=f'Max {performance_metric}', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_threshold, y=min_performance, name=f'Min {performance_metric}', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_threshold, y=quantile_075_performance, name=f'Quantile 0.25 {performance_metric}', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_threshold, y=quantile_05_performance, name=f'Quantile 0.5 {performance_metric}', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_threshold, y=quantile_025_performance, name=f'Quantile 0.75 {performance_metric}', mode='lines'))

    fig.add_hrect(y0=min_performance[0]-10, y1=0, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=max_performance[0]+10, fillcolor="green", opacity=0.1, line_width=0)
    
    return fig