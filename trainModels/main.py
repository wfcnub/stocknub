import numpy as np
import pandas as pd
from pathlib import Path
from skopt.space import Real, Integer
from utils.pipeline import get_label_config

from trainModels.modelling import (
    _combine_multiple_ticker_in_industry,
    _combine_multiple_ticker,
    _split_data_to_train_val_test_single,
    _split_data_to_train_val_test_multiple, 
    _initializes_fit_tune_catboost_with_bayesian_optimization,
    _initializes_fit_tune_logistic_regression_with_bayesian_optimization,
    _calculate_classification_metrics,
    _calculate_gini,
    _measure_model_performance,
    _measure_model_performance_on_single_ticker,
    _measure_model_performance_for_all_ticker_in_industry,
    _measure_model_performance_for_all_ticker,
    _measure_model_performance_on_forecast_features_for_all_ticker,
)

from trainModels.helper import _save_model, _combine_metrics
from prepareTechnicalIndicators.helper import get_all_technical_indicators
from combineForecasts.helper import _get_combined_forecasts_features_target_threshold

def develop_model_v1(ticker: str, target_column: str, positive_label: str, negative_label: str) -> (any, dict, dict):
    """
    Main orchestration function for the entire model development process

    This function loads the feature names, splits the data, tunes the model,
    and evaluates its final performance

    Args:
        ticker (str): The name of the ticker being worked on
        target_column (str): The name of the target variable column
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label
        
    Returns:
        tuple: A tuple containing:
               - model (CatBoostClassifier): The final, trained model
               - train_metrics (dict): Performance metrics on the training set
               - test_metrics (dict): Performance metrics on the testing set
    """

    feature_columns = get_all_technical_indicators()

    prepared_data = pd.read_csv(Path(f'data/stock/label/{ticker}.csv'))

    cleaned_data = prepared_data[feature_columns + [target_column]].dropna(subset=[target_column])

    train_feature, train_target, test_feature, test_target, cv_split = _split_data_to_train_val_test_single(cleaned_data, feature_columns, target_column)

    search_spaces = {
        'depth': Integer(1, 5),
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'iterations': Integer(150, 300),
        'l2_leaf_reg': Real(0.5, 3.0)
    }

    model = _initializes_fit_tune_catboost_with_bayesian_optimization(train_feature, train_target, cv_split, search_spaces)

    train_metrics = _measure_model_performance(model, train_feature, train_target, positive_label, negative_label)

    test_metrics = _measure_model_performance(model, test_feature, test_target, positive_label, negative_label)
    
    return model, train_metrics, test_metrics

def develop_model_v2(industry: str, target_column: str, positive_label: str, negative_label: str, threshold_col: str) -> (any, dict, dict):
    """
    Main orchestration function for the entire model development process

    This function loads the feature names, splits the data, tunes the model,
    and evaluates its final performance

    Args:
        industry (str): The name of the industry being worked on
        target_column (str): The name of the target variable column
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label
        
    Returns:
        tuple: A tuple containing:
               - model (CatBoostClassifier): The final, trained model
               - train_metrics (dict): Performance metrics on the training set
               - test_metrics (dict): Performance metrics on the testing set
    """
    feature_columns = get_all_technical_indicators()

    prepared_data = _combine_multiple_ticker_in_industry(industry)

    cleaned_data = prepared_data.dropna(subset=[target_column])

    train_feature, train_target, test_feature, test_target, cv_split = _split_data_to_train_val_test_multiple(cleaned_data, feature_columns, target_column)

    search_spaces = {
        'depth': Integer(1, 5),
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'iterations': Integer(750, 1000),
        'l2_leaf_reg': Real(0.5, 3.0)
    }

    model = _initializes_fit_tune_catboost_with_bayesian_optimization(train_feature, train_target, cv_split, search_spaces)

    train_metrics, test_metrics = _measure_model_performance_for_all_ticker_in_industry(industry, model, target_column, positive_label, negative_label, threshold_col)
    
    return model, train_metrics, test_metrics

def develop_model_v3(target_column: str, positive_label: str, negative_label: str, threshold_col: str) -> (any, dict, dict):
    """
    Main orchestration function for the entire model development process

    This function loads the feature names, splits the data, tunes the model,
    and evaluates its final performance

    Args:
        target_column (str): The name of the target variable column
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label
        
    Returns:
        tuple: A tuple containing:
               - model (CatBoostClassifier): The final, trained model
               - train_metrics (dict): Performance metrics on the training set
               - test_metrics (dict): Performance metrics on the testing set
    """
    feature_columns = get_all_technical_indicators()

    prepared_data = _combine_multiple_ticker('data/stock/label')

    cleaned_data = prepared_data.dropna(subset=[target_column])
    
    train_feature, train_target, test_feature, test_target, cv_split = _split_data_to_train_val_test_multiple(cleaned_data, feature_columns, target_column)

    search_spaces = {
        'depth': Integer(1, 5),
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'iterations': Integer(1000, 1250),
        'l2_leaf_reg': Real(0.5, 3.0)
    }

    model = _initializes_fit_tune_catboost_with_bayesian_optimization(train_feature, train_target, cv_split, search_spaces)

    train_metrics, test_metrics = _measure_model_performance_for_all_ticker(model, target_column, positive_label, negative_label, threshold_col)
    
    return model, train_metrics, test_metrics

def develop_model_v4(label_types: list, rolling_windows: list, positive_label: str, negative_label: str) -> (any, dict, dict, str):
    """
    Main orchestration function for the entire model development process

    This function loads the feature names, splits the data, tunes the model,
    and evaluates its final performance

    Args:
        target_column (str): The name of the target variable column
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label
        
    Returns:
        tuple: A tuple containing:
               - model (CatBoostClassifier): The final, trained model
               - train_metrics (dict): Performance metrics on the training set
               - test_metrics (dict): Performance metrics on the testing set
    """
    feature_columns, target_column, threhsold_column = _get_combined_forecasts_features_target_threshold()

    prepared_data = _combine_multiple_ticker('data/stock/combined_forecasts')
    
    cleaned_data = prepared_data.dropna(subset=[target_column])

    train_feature, train_target, test_feature, test_target, cv_split = _split_data_to_train_val_test_multiple(cleaned_data, feature_columns, target_column)

    model = _initializes_fit_tune_logistic_regression_with_bayesian_optimization(train_feature, train_target, cv_split)

    train_metrics, test_metrics = _measure_model_performance_on_forecast_features_for_all_ticker(model, positive_label, negative_label, label_types, rolling_windows)
     
    return model, train_metrics, test_metrics, threhsold_column

def process_single_model(args_tuple):
    """
    Utilize label data to create a machine learning model.

    Args:
        args_tuple: Tuple containing (label_file, label_types, rolling_windows, model_version)

    Returns:
        Tuple of (failed_process, metrics_list)
    """
    identifier, label_types, rolling_windows, model_version = args_tuple
 
    failed_process = []
    metrics_list = []

    try:
        if model_version in [1, 2, 3]:
            for label_type in label_types:
                for window in rolling_windows:
                    target_col, threshold_col, pos_label, neg_label = get_label_config(
                        label_type, window
                    )

                    try:
                        if model_version == 1:
                            model, train_metrics, test_metrics = develop_model_v1(
                                identifier, target_col, pos_label, neg_label
                            )
                        elif model_version == 2:
                            model, train_metrics, test_metrics = develop_model_v2(
                                identifier, target_col, pos_label, neg_label, threshold_col
                            )
                        
                        elif model_version == 3:
                            model, train_metrics, test_metrics = develop_model_v3(
                                target_col, pos_label, neg_label, threshold_col
                            )
                        
                        _save_model(model, model_version, label_type, identifier, window)
                        
                        metrics_df = _combine_metrics(
                            identifier, model_version, train_metrics, test_metrics, threshold_col
                        )

                        metrics_list.append((label_type, window, metrics_df))

                    except Exception as e:
                        failed_process.append((identifier, label_type, window, str(e)))
            
        elif model_version == 4:
            try:
                model, train_metrics, test_metrics, threshold_column = develop_model_v4(
                                label_types, rolling_windows, 'High Gain', 'Low Gain'
                            )

                _save_model(model, model_version, 'median_gain', identifier, np.max(rolling_windows))
                
                metrics_df = _combine_metrics(
                    identifier, model_version, train_metrics, test_metrics, threshold_column
                )

                metrics_list.append(('median_gain', np.max(rolling_windows), metrics_df))
            
            except Exception as e:
                failed_process.append((identifier, 'median_gain', np.max(rolling_windows), str(e)))

    except Exception as e:
        failed_process.append((identifier, "all", "all", str(e)))

    return failed_process, metrics_list