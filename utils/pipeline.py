import json
import pandas as pd
from pathlib import Path

def get_label_config(label_type: str, window: int) -> tuple:
    """
    Get configuration for a specific label type and window.

    Args:
        label_type (str): Type of label ('median_gain', 'median_loss')
        window (int): Forecast window in days

    Returns:
        tuple: (target_column, threshold_column, positive_label, negative_label)

    Raises:
        ValueError: If label_type is not recognized
    """
    if label_type == "median_gain":
        return (
            f"Median Gain {window}dd",
            f"Threshold Median Gain {window}dd",
            "High Gain",
            "Low Gain",
        )
    elif label_type == "median_loss":
        return (
            f"Median Loss {window}dd",
            f"Threshold Median Loss {window}dd",
            "High Loss",
            "Low Loss",
        )
    else:
        raise ValueError(f"Unknown label type: {label_type}")

def get_split_dates(target_column: str) -> dict:
    """
    Get the split dates configuration based on the target column.

    Args:
        target_column (str): The name of the target column

    Returns:
        dict: The dictionary containing the train, val, and test split dates
    """
    window_dd = target_column.split(" ")[-1]
    json_path = Path(f"data/split_dates_{window_dd}.json")
    
    with open(json_path, 'r') as f:
        splits = json.load(f)
        
    return splits

def get_split_masks(data: pd.DataFrame, splits: dict) -> tuple:
    """
    Get the split masks for train_val, train, val, test, and forecast datasets.
    
    Args:
        data (pd.DataFrame): The dataframe containing a 'Date' column.
        splits (dict): The dictionary containing the split dates.
        
    Returns:
        tuple: A tuple containing (train_val_mask, train_mask, val_mask, test_mask, forecast_mask)
    """
    train_val_mask = (data['Date'] >= splits['train']['start_date']) & (data['Date'] <= splits['val']['end_date'])
    train_mask = (data['Date'] >= splits['train']['start_date']) & (data['Date'] <= splits['train']['end_date'])
    val_mask = (data['Date'] >= splits['val']['start_date']) & (data['Date'] <= splits['val']['end_date'])
    test_mask = (data['Date'] >= splits['test']['start_date']) & (data['Date'] <= splits['test']['end_date'])
    forecast_mask = data['Date'] > splits['test']['end_date']
    
    return train_val_mask, train_mask, val_mask, test_mask, forecast_mask
