import pickle
import pandas as pd
from pathlib import Path
from camel_converter import to_camel

def _ensure_directories_exist(model_version: int, label_types: str, windows: int) -> None:
    """
    (Internal Helper) Ensure all required directories exist before forecasting, if not then it will create the directory

    Args:
        model_version (int): The version of model being developed
        label_types (str): The label used to develop the model
        windows (int): The rolling window used to create the label
    """
    for label_type in label_types:
        camel_label = to_camel(label_type)
        for window in windows:
            Path(f"data/stock/forecast/model_v{model_version}/{camel_label}/{window}dd") \
                .mkdir(parents=True, exist_ok=True)
        
    return

def _clear_forecast_files(model_version: int, label_types: str, windows: int) -> None:
    """
    (Internal Helper) Clear existing forecast files to avoid duplicates

    Args:
        model_version (int): The version of model being developed
        label_types (str): The label used to develop the model
        windows (int): The rolling window used to create the label
    """
    for label_type in label_types:
        camel_label = to_camel(label_type)
        for window in windows:
            filepath = Path(f"data/stock/forecast/model_v{model_version}/{camel_label}/{window}dd.csv")
            if filepath.exists():
                filepath.unlink()
            
    return

def _load_model_performance(model_version: int, label_type: str, window: int, min_test_gini: float = None) -> list:
    """
    (Internal Helper) Load model performance data and filter by minimum Gini.

    Args:
        model_version (int): The version of model being developed
        label_type (str): The label used to develop the model
        window (int): The rolling window used to create the label
        min_test_gini (float): Minimum test Gini threshold (None to include all)

    Returns:
        list: List of ticker codes that meet the criteria
    """
    camel_label = to_camel(label_type)
    performance_path = Path(f"data/stock/model_v{model_version}/performance/{camel_label}/{window}dd.csv")

    if not performance_path.exists():
        print(f"WARNING: Performance file not found: {performance_path}")
        return []

    performance_df = pd.read_csv(performance_path)

    if min_test_gini is not None:
        filtered_df = performance_df[performance_df["Test - Gini"] >= min_test_gini]
        filtered_df = filtered_df.sort_values("Test - Gini", ascending=False)
        return filtered_df["Ticker"].unique().tolist()
    else:
        return performance_df["Ticker"].unique().tolist()


def _get_filtered_ticker_list(model_version: int, label_types: str, windows: int, min_test_gini: float = None) -> list:
    """
    (Internal Helper) Get intersection of ticker codes that meet criteria across all label types and windows.

    Args:
        model_version (int): The version of model being developed
        label_types (str): The label used to develop the model
        windows (int): The rolling window used to create the label
        min_test_gini (float): Minimum test Gini threshold (None to include all)

    Returns:
        list: List of ticker codes that have models meeting criteria for all combinations
    """
    all_ticker_sets = []

    for label_type in label_types:
        for window in windows:
            ticker_list = _load_model_performance(model_version, label_type, window, min_test_gini)
            if ticker_list:
                all_ticker_sets.append(set(ticker_list))

    if not all_ticker_sets:
        return []

    common_ticker = set.intersection(*all_ticker_sets)
    return sorted(list(common_ticker))


def _save_forecast(forecast_df: pd.DataFrame, model_version: int, label_type: str, window: int, ticker: str) -> None:
    """
    (Internal Helper) Save or append forecast results to CSV

    Args:
        forecast_df (pd.DataFrame): A pandas dataframe containing the forecasted value
        model_version (int): The version of model being developed
        label_type (str): The label used to develop the model
        window (int): The rolling window used to create the label
        ticker (str): The name of the ticker inside the forecast_df
    """
    camel_label = to_camel(label_type)
    filepath = Path(f"data/stock/forecast/model_v{model_version}/{camel_label}/{window}dd/{ticker}.csv")
    forecast_df.to_csv(filepath, index=False)

    return