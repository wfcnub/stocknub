import pandas as pd
from utils.pipeline import get_label_config

from trainModels.modelling_v1 import develop_model
from trainModels.helper import _ensure_directories_exist, _save_model, _combine_metrics

def process_single_ticker(args_tuple):
    """
    Utilize label data to create a machine learning model.

    Args:
        args_tuple: Tuple containing (label_file, label_types, rolling_windows, feature_columns)

    Returns:
        Tuple of (failed_stocks, metrics_list)
    """
    label_file, label_types, rolling_windows, feature_columns = args_tuple

    emiten = label_file.stem
    failed_stocks = []
    metrics_list = []

    try:
        data = pd.read_csv(label_file)

        for label_type in label_types:
            for window in rolling_windows:
                target_col, threshold_col, pos_label, neg_label = get_label_config(
                    label_type, window
                )

                required_cols = feature_columns + [target_col]
                clean_data = data[required_cols].dropna(subset=[target_col])

                try:
                    model, train_metrics, test_metrics = develop_model(
                        clean_data, target_col, pos_label, neg_label
                    )

                    _save_model(model, label_type, emiten, window)

                    threshold_value = data[threshold_col].iloc[0]
                    metrics_df = _combine_metrics(
                        emiten, train_metrics, test_metrics, threshold_value
                    )

                    metrics_list.append((label_type, window, metrics_df))

                except Exception as e:
                    failed_stocks.append((emiten, label_type, window, str(e)))

    except Exception as e:
        failed_stocks.append((emiten, "all", "all", str(e)))

    return failed_stocks, metrics_list


