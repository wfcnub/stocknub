import pickle
import pandas as pd
from pathlib import Path
from camel_converter import to_camel

def _ensure_directories_exist(label_types: str, windows: int) -> None:
    """
    (Internal Helper) Ensure all required directories exist before forecasting, if not then it will create the directory

    Args:
        label_types (str): The label used to develop the model
        windows (int): The rolling window used to create the label
    """
    for label_type in label_types:
        camel_label = to_camel(label_type)
        for window in windows:
            Path(f"data/stock/forecast/model_v2/{camel_label}/{window}dd").mkdir(
                parents=True, exist_ok=True
            )
    
    return


def _clear_forecast_files(label_types: str, windows: int) -> None:
    """
    (Internal Helper) Ensure all required directories exist before forecasting, if not then it will create the directory

    Args:
        label_types (str): The label used to develop the model
        windows (int): The rolling window used to create the label
    """
    for label_type in label_types:
        camel_label = to_camel(label_type)
        for window in windows:
            filepath = f"data/stock/forecast/model_v2/{camel_label}/{window}dd.csv"
            if Path(filepath).exists():
                Path(filepath).unlink()

    return

def _load_model_performance(label_type: str, window: int, min_test_gini: float = None) -> list:
    """
    (Internal Helper) Load model performance data and filter by minimum Gini.

    Args:
        label_type (str): The label used to develop the model
        window (int): The rolling window used to create the label
        min_test_gini (float): Minimum test Gini threshold (None to include all)

    Returns:
        list: List of emiten codes that meet the criteria
    """
    camel_label = to_camel(label_type)
    performance_path = f"data/stock/model_v2/performance/{camel_label}/{window}dd.csv"

    if not Path(performance_path).exists():
        print(f"WARNING: Performance file not found: {performance_path}")
        return []

    performance_df = pd.read_csv(performance_path)

    if min_test_gini is not None:
        filtered_df = performance_df[performance_df["Test - Gini"] >= min_test_gini]
        filtered_df = filtered_df.sort_values("Test - Gini", ascending=False)
        return filtered_df["Kode"].unique().tolist()
    else:
        return performance_df["Kode"].unique().tolist()


def _get_filtered_emiten_list(label_types: str, windows: int, min_test_gini: float = None) -> list:
    """
    Get intersection of emiten codes that meet criteria across all label types and windows.

    Args:
        label_types (str): The label used to develop the model
        windows (int): The rolling window used to create the label
        min_test_gini (float): Minimum test Gini threshold (None to include all)

    Returns:
        list: List of emiten codes that have models meeting criteria for all combinations
    """
    all_emiten_sets = []

    for label_type in label_types:
        for window in windows:
            emiten_list = _load_model_performance(label_type, window, min_test_gini)
            if emiten_list:
                all_emiten_sets.append(set(emiten_list))

    if not all_emiten_sets:
        return []

    common_emiten = set.intersection(*all_emiten_sets)
    return sorted(list(common_emiten))


def _save_forecast(forecast_df: pd.DataFrame, label_type: str, window: int, emiten: str) -> None:
    """
    (Internal Helper) Save or append forecast results to CSV

    Args:
        forecast_df (pd.DataFrame): A pandas dataframe containing the forecasted value
        label_type (str): The label used to develop the model
        window (int): The rolling window used to create the label
        emiten (str): The name of the emiten inside the forecast_df
    """
    camel_label = to_camel(label_type)
    filepath = f"data/stock/forecast/model_v2/{camel_label}/{window}dd/{emiten}.csv"
    forecast_df.to_csv(filepath, index=False)

    return