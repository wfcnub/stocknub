import pickle
import pandas as pd
from pathlib import Path
from camel_converter import to_camel

def _ensure_directories_exist(model_version, label_types, windows):
    """Ensure all required directories exist before forecasting."""
    for label_type in label_types:
        camel_label = to_camel(label_type)
        for window in windows:
            Path(f"data/stock/forecast/{model_version}/{camel_label}/{window}dd").mkdir(
                parents=True, exist_ok=True
            )


def _clear_forecast_files(model_version, label_types, windows):
    """Clear existing forecast files to avoid duplicates."""
    for label_type in label_types:
        camel_label = to_camel(label_type)
        for window in windows:
            filepath = f"data/stock/forecast/{model_version}/{camel_label}/{window}dd.csv"
            if Path(filepath).exists():
                Path(filepath).unlink()


def _load_model_performance(model_version, label_type, window, min_test_gini=None):
    """
    Load model performance data and filter by minimum Gini.

    Args:
        label_type: Type of label (median_gain, max_loss, linear_trend)
        window: Forecast window (e.g., 5, 10, 20)
        min_test_gini: Minimum test Gini threshold (None to include all)

    Returns:
        List of emiten codes that meet the criteria
    """
    camel_label = to_camel(label_type)
    performance_path = f"data/stock/{model_version}/performance/{camel_label}/{window}dd.csv"

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


def _get_filtered_emiten_list(model_version, label_types, windows, min_test_gini=None):
    """
    Get intersection of emiten codes that meet criteria across all label types and windows.

    Args:
        label_types: List of label types
        windows: List of forecast windows
        min_test_gini: Minimum test Gini threshold

    Returns:
        List of emiten codes that have models meeting criteria for all combinations
    """
    all_emiten_sets = []

    for label_type in label_types:
        for window in windows:
            emiten_list = _load_model_performance(model_version, label_type, window, min_test_gini)
            if emiten_list:
                all_emiten_sets.append(set(emiten_list))

    if not all_emiten_sets:
        return []

    common_emiten = set.intersection(*all_emiten_sets)
    return sorted(list(common_emiten))


def _save_forecast(forecast_df, model_version, label_type, window, emiten):
    """Save or append forecast results to CSV."""
    camel_label = to_camel(label_type)
    filepath = f"data/stock/forecast/{model_version}/{camel_label}/{window}dd/{emiten}.csv"
    forecast_df.to_csv(filepath, index=False)

    return