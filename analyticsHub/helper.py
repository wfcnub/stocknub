import numpy as np
import pandas as pd
from pathlib import Path
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

def _visualize_micro_outlook_boxplot(mo_data: dict, xaxis_title: str, color: str) -> go.Figure:
    """
    (Internal Helper) Generate a boxplot for the Micro Outlook statistics
    """
    fig = go.Figure(go.Box(
        name="Median Gain",
        q1=[mo_data.get("25%", 0)],
        median=[mo_data.get("50%", 0)],
        q3=[mo_data.get("75%", 0)],
        lowerfence=[mo_data.get("min", 0)],
        upperfence=[mo_data.get("max", 0)],
        mean=[mo_data.get("mean", 0)],
        marker_color=color
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Gain (%)", xaxis_title=xaxis_title)
    
    return fig

def _apply_bin_scores(val):
    """
    (Internal Helper) Apply bin scores for simulation grouping.
    """
    bin_scores = [0.2, 0.4, 0.6, 0.8, 1]
    for i, upper in enumerate(bin_scores):
        if val <= upper:
            return i
    return len(bin_scores)

def _generate_score_data(rolling_window: str) -> (pd.DataFrame, str):
    """
    (Internal Helper) Generate the score data for daily recommendations
    """
    score_paths = Path(f'data/stock/score/{rolling_window}').rglob('*.csv')
    all_ticker = [file.stem for file in Path(f'data/stock/score/{rolling_window}').rglob('*.csv')]
    
    score_df = pd.DataFrame()
    for ticker, file in zip(all_ticker, score_paths):
        temp_score_df = pd.read_csv(file, usecols=['Date', f'Score {rolling_window}']).tail(1)
        temp_score_df['Ticker'] = ticker
        score_df = pd.concat((score_df, temp_score_df))
    
    assert score_df['Date'].nunique() == 1
    score_date = score_df['Date'].unique()[0]
    
    score_df.set_index('Ticker', inplace=True)
    score_df.drop(columns=['Date'], inplace=True)
    
    score_df[f'Score {rolling_window} Bin'] = score_df[f'Score {rolling_window}'].apply(lambda val: _apply_bin_scores(val))
    
    return score_df, score_date

def _generate_close_data() -> pd.DataFrame:
    """
    (Internal Helper) Generate the close price data for daily recommendations
    """
    label_paths = Path('data/stock/label').rglob('*.csv')
    all_tickers = [file.stem for file in Path('data/stock/label').rglob('*.csv')]

    all_close_df = pd.DataFrame()
    for ticker, file in zip(all_tickers, label_paths):
        close_df = pd.read_csv(file, usecols=['Date', 'Close']).tail(1)
        close_df['Ticker'] = ticker
        all_close_df = pd.concat((all_close_df, close_df))
    
    assert all_close_df['Date'].nunique() == 1
    all_close_df.drop(columns=['Date'], inplace=True)
    all_close_df.reset_index(drop=True, inplace=True)
    
    return all_close_df

def _generate_buy_sell_percentage_data(rolling_window: str) -> pd.DataFrame:
    """
    (Internal Helper) Generate the simulation buy/sell percentages
    """
    simulation_df = pd.read_csv(f'data/stock/score/trading_simulation_{rolling_window}.csv')
    simulation_df[f'Score {rolling_window} Bin'] = simulation_df[f'Score {rolling_window}'].apply(lambda val: _apply_bin_scores(val))

    buy_percentage = simulation_df.groupby(f'Score {rolling_window} Bin')['Loss'].quantile(0.25).to_dict()
    sell_percentage = simulation_df.groupby(f'Score {rolling_window} Bin')['Profit'].quantile(0.50).to_dict()
    
    return buy_percentage, sell_percentage

def _generate_recommendation_data(score_df: pd.DataFrame, all_close_df: pd.DataFrame, buy_percentage: dict, sell_percentage: dict, rolling_window: str) -> pd.DataFrame:
    """
    (Internal Helper) Generate final daily recommendation targets
    """
    recommendation_df = pd.merge(
        score_df,
        all_close_df,
        on='Ticker',
        how='inner'
    )
    
    recommendation_df['Buy Percentage'] = recommendation_df[f'Score {rolling_window} Bin'].apply(lambda val: buy_percentage[val])
    recommendation_df['Sell Percentage'] = recommendation_df[f'Score {rolling_window} Bin'].apply(lambda val: sell_percentage[val])

    recommendation_df['Target Buy Price'] = recommendation_df.apply(lambda row: np.floor(row['Close'] * (100 - np.abs(row['Buy Percentage'])) / 100), axis=1)
    recommendation_df['Target Sell Price'] = recommendation_df.apply(lambda row: np.ceil(row['Close'] * (100 + np.abs(row['Sell Percentage'])) / 100), axis=1)
    
    return recommendation_df