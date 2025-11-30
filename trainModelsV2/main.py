import pandas as pd
from utils.pipeline import get_label_config

from trainModelsV2.modelling import develop_model
from trainModelsV2.helper import _ensure_directories_exist, _save_model, _combine_metrics

def process_single_ticker(args_tuple):
    """
    Utilize label data to create a machine learning model.

    Args:
        args_tuple: Tuple containing (label_file, label_types, rolling_windows, feature_columns)

    Returns:
        Tuple of (failed_stocks, metrics_list)
    """
    industry, label_types, rolling_windows, feature_columns = args_tuple    
        
    failed_industry = []
    metrics_list = []

    try:
        for label_type in label_types:
            for window in rolling_windows:
                target_col, threshold_col, pos_label, neg_label = get_label_config(
                    label_type, window
                )

                try:
                    model, train_metrics, test_metrics = develop_model(
                        industry, target_col, pos_label, neg_label, threshold_col
                    )                        

                    _save_model(model, label_type, industry, window)

                    metrics_df = _combine_metrics(
                        train_metrics, test_metrics
                    )

                    metrics_list.append((label_type, window, metrics_df))

                except Exception as e:
                    failed_industry.append((industry, label_type, window, str(e)))

    except Exception as e:
        failed_industry.append((industry, "all", "all", str(e)))

    return failed_industry, metrics_list


