import numpy as np
import pandas as pd

from generateLabels.linear_trend import _generate_all_linreg_gradients
from generateLabels.median_gain import _generate_all_median_gain
from generateLabels.max_loss import _generate_all_max_loss

def _generate_labels_based_on_label_type(data: pd.DataFrame, target_column: str, rolling_windows: list, label_types: list) -> pd.DataFrame:
    """
    (Internal Helper) Generates a label following the requested label type for each day based on a rolling window

    Args:
        data (pd.DataFrame): The input DataFrame containing stock data
        target_column (str): The name of the column to analyze
        rolling_windows (list): A list of rolling window, the number of future days to look at for the label
        label_types (list): A list of label types for model's target variables

    Returns:
        pd.DataFrame: A dataframe with an added column of the generated label
    """
    for label_type in label_types:
        for window in rolling_windows:
            if label_type in 'linear_trend':
                data = _generate_all_linreg_gradients(data, target_column, window)
                
            elif label_type == 'median_gain':
                data = _generate_all_median_gain(data, target_column, window)
        
            elif label_type == 'max_loss':
                data = _generate_all_max_loss(data, target_column, window)
    
    return data