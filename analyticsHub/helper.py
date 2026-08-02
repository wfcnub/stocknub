import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import paths


def _window_sort_key(window: str) -> tuple[int, str]:
    numeric_part = window.removesuffix('dd')
    return (
        int(numeric_part) if numeric_part.isdigit() else np.iinfo(np.int32).max,
        window,
    )


def _get_available_score_windows(
    require_simulation: bool = False,
    require_split: bool = False,
) -> list[str]:
    """Return score windows whose artifacts are usable by the analytics hub."""
    score_dir = paths.get_score_dir()
    if not score_dir.is_dir():
        return []

    windows = []
    for window_dir in score_dir.iterdir():
        window = window_dir.name
        if not window_dir.is_dir() or not window.endswith('dd'):
            continue
        if not any(window_dir.glob('*.csv')):
            continue
        if require_simulation and not paths.get_trading_simulation_path(window).is_file():
            continue
        if require_split and not paths.get_split_dates_path(window).is_file():
            continue
        windows.append(window)
    return sorted(windows, key=_window_sort_key)

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
    if pd.isna(val):
        return np.nan

    bin_scores = [0.2, 0.4, 0.6, 0.8, 1]
    for i, upper in enumerate(bin_scores):
        if val <= upper:
            return i
    return len(bin_scores)

def _generate_score_data(rolling_window: str) -> (pd.DataFrame, str):
    """
    (Internal Helper) Generate the score data for daily recommendations
    """
    score_paths = sorted(paths.get_score_window_dir(rolling_window).glob('*.csv'))
    if not score_paths:
        raise FileNotFoundError(
            f"No score files found for the {rolling_window} window"
        )
    all_ticker = [file.stem for file in score_paths]
    
    score_df = pd.DataFrame()
    for ticker, file in zip(all_ticker, score_paths):
        temp_score_df = pd.read_csv(file, usecols=['Date', f'Score {rolling_window}']).tail(1)
        temp_score_df['Ticker'] = ticker
        score_df = pd.concat((score_df, temp_score_df), ignore_index=True)
    
    score_date = score_df['Date'].max()
    
    score_df = score_df[score_df['Date'] == score_date]
    score_df.set_index('Ticker', inplace=True)
    score_df.drop(columns=['Date'], inplace=True)
    
    score_df[f'Score {rolling_window} Bin'] = score_df[f'Score {rolling_window}'].apply(lambda val: _apply_bin_scores(val))
    
    return score_df, score_date

def _generate_close_data(as_of_date: str | None = None) -> pd.DataFrame:
    """
    (Internal Helper) Generate the close price data for daily recommendations
    """
    label_paths = sorted(paths.get_label_dir().glob('*.csv'))
    if not label_paths:
        raise FileNotFoundError("No label files found for recommendation prices")
    all_tickers = [file.stem for file in label_paths]

    all_close_df = pd.DataFrame()
    for ticker, file in zip(all_tickers, label_paths):
        close_df = pd.read_csv(file, usecols=['Date', 'Close'])
        if as_of_date is not None:
            close_df = close_df[close_df['Date'].astype(str) <= str(as_of_date)]
        close_df = close_df.tail(1)
        if close_df.empty:
            continue
        close_df['Ticker'] = ticker
        all_close_df = pd.concat((all_close_df, close_df), ignore_index=True)

    if all_close_df.empty:
        raise ValueError(
            f"No close prices are available on or before {as_of_date}"
        )

    all_close_df.drop(columns=['Date'], inplace=True)
    all_close_df.reset_index(drop=True, inplace=True)
    
    return all_close_df

def _generate_buy_sell_percentage_data(rolling_window: str) -> pd.DataFrame:
    """
    (Internal Helper) Generate the simulation buy/sell percentages
    """
    simulation_df = pd.read_csv(paths.get_trading_simulation_path(rolling_window))
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
    
    def nearest_bin_value(mapping: dict, score_bin: int) -> float:
        if pd.isna(score_bin) or not mapping:
            return np.nan
        if score_bin in mapping:
            return mapping[score_bin]
        nearest_bin = min(mapping, key=lambda candidate: abs(candidate - score_bin))
        return mapping[nearest_bin]

    recommendation_df['Buy Percentage'] = recommendation_df[
        f'Score {rolling_window} Bin'
    ].apply(lambda value: nearest_bin_value(buy_percentage, value))
    recommendation_df['Sell Percentage'] = recommendation_df[
        f'Score {rolling_window} Bin'
    ].apply(lambda value: nearest_bin_value(sell_percentage, value))

    recommendation_df['Target Buy Price'] = recommendation_df.apply(lambda row: np.floor(row['Close'] * (100 - np.abs(row['Buy Percentage'])) / 100), axis=1)
    recommendation_df['Target Sell Price'] = recommendation_df.apply(lambda row: np.ceil(row['Close'] * (100 + np.abs(row['Sell Percentage'])) / 100), axis=1)
    
    return recommendation_df
