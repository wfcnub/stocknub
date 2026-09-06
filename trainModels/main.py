import hashlib
import numpy as np
import pandas as pd
import traceback
import catboost
import optuna
import sklearn

from utils import paths
from utils.pipeline import get_label_config, get_split_dates

from trainModels.modelling import (
    _combine_multiple_ticker_in_industry,
    _combine_multiple_ticker,
    _split_development_and_locked_test,
    _split_data_to_train_val_test_multiple, 
    _initializes_fit_tune_logistic_regression_with_bayesian_optimization,
    _measure_model_performance,
    _measure_oof_predictions,
    _measure_model_performance_for_all_ticker_in_industry,
    _measure_model_performance_for_all_ticker,
    _measure_model_performance_on_forecast_features_for_all_ticker,
    _initializes_fit_logistic_regression
)

from trainModels.config import CatBoostTrainingConfig
from trainModels.features import (
    add_cross_sectional_features,
    attach_ticker_metadata,
    select_usable_features,
)
from trainModels.helper import _save_model, _save_training_artifacts, _combine_metrics
from trainModels.splits import make_purged_walk_forward_splits
from trainModels.tuning import optimize_catboost
from prepareTechnicalIndicators.helper import get_all_technical_indicators
from combineForecasts.helper import _get_combined_forecasts_features_target_threshold

def _universe_audit_metadata(target_column: str) -> dict:
    selected = pd.read_csv(paths.get_selected_ticker_and_industry_list_path())
    splits = get_split_dates(target_column)
    selection_as_of = None
    if "selection_as_of" in selected and selected["selection_as_of"].notna().any():
        selection_as_of = str(selected["selection_as_of"].dropna().max())
    test_end = str(splits["test"]["end_date"])
    test_start = str(splits["test"]["start_date"])
    return {
        "universe_selection_as_of": selection_as_of,
        "locked_test_start": test_start,
        "locked_test_end": test_end,
        "point_in_time_test_universe_available": bool(
            selection_as_of is not None and selection_as_of <= test_start
        ),
    }


def _catboost_development_data(
    prepared_data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    positive_label: str,
    categorical_features: list[str],
    config: CatBoostTrainingConfig,
):
    cleaned_data = prepared_data.dropna(subset=[target_column]).copy()
    enriched_data, context_features = add_cross_sectional_features(cleaned_data)
    development_data, test_data = _split_development_and_locked_test(
        enriched_data, target_column
    )
    candidate_features = list(
        dict.fromkeys(feature_columns + context_features + categorical_features)
    )
    selected_features, removed_features = select_usable_features(
        development_data,
        candidate_features,
        categorical_features,
        config.max_missing_fraction,
    )
    folds = make_purged_walk_forward_splits(
        development_data,
        target_column,
        positive_label,
        n_splits=config.n_folds,
        validation_dates=config.validation_dates,
        min_train_dates=config.min_train_dates,
        min_class_count=config.min_class_count,
        min_valid_splits=config.min_valid_folds,
    )
    return (
        development_data,
        test_data,
        selected_features,
        removed_features,
        folds,
    )


def _validation_metrics_by_ticker(
    predictions: pd.DataFrame,
    model,
    target_column: str,
    positive_label: str,
    negative_label: str,
    threshold_col: str,
    tickers: list[str],
) -> dict:
    rows = []
    for ticker in tickers:
        ticker_predictions = predictions[predictions["Ticker"] == ticker]
        if ticker_predictions.empty:
            raise ValueError(f"No OOF validation predictions for {ticker}")
        metrics = pd.DataFrame(
            _measure_oof_predictions(
                ticker_predictions,
                target_column,
                positive_label,
                negative_label,
                model.decision_threshold,
            )
        )
        metrics["Ticker"] = ticker
        source = pd.read_csv(
            paths.get_label_path(ticker), usecols=[threshold_col], nrows=1
        )
        metrics["Threshold"] = source[threshold_col].iloc[0]
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True).to_dict(orient="list")


def _tuning_artifacts(
    tuning_result,
    target_column: str,
    development_data: pd.DataFrame,
) -> dict:
    fingerprint_columns = [
        "Date",
        target_column,
        *tuning_result.model.feature_names_,
    ]
    fingerprint_columns = list(dict.fromkeys(fingerprint_columns))
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            development_data[fingerprint_columns], index=False
        ).values.tobytes()
    ).hexdigest()
    return {
        "trials": tuning_result.trial_history,
        "feature-importance": tuning_result.feature_importance,
        "validation-predictions": tuning_result.validation_predictions.reset_index(
            drop=True
        ),
        "metadata": {
            "target_column": target_column,
            "best_parameters": tuning_result.best_parameters,
            "removed_features": tuning_result.removed_features,
            "folds": tuning_result.fold_boundaries,
            "selected_features": tuning_result.model.feature_names_,
            "development_data_sha256": data_hash,
            "library_versions": {
                "catboost": catboost.__version__,
                "optuna": optuna.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "prediction_timing": "after_market_close",
            **_universe_audit_metadata(target_column),
        },
    }


def develop_model_v1(
    ticker: str,
    target_column: str,
    positive_label: str,
    negative_label: str,
    config: CatBoostTrainingConfig | None = None,
) -> tuple:
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

    config = config or CatBoostTrainingConfig()
    prepared_data = attach_ticker_metadata(
        pd.read_csv(paths.get_label_path(ticker)), ticker
    )
    development_data, test_data, selected_features, removed, folds = (
        _catboost_development_data(
            prepared_data,
            feature_columns,
            target_column,
            positive_label,
            [],
            config,
        )
    )
    tuning = optimize_catboost(
        development_data,
        target_column,
        positive_label,
        negative_label,
        selected_features,
        [],
        folds,
        config,
        removed,
        paths.get_tuning_study_path(1, ticker, target_column),
    )
    model = tuning.model
    train_metrics = _measure_model_performance(
        model, development_data, development_data[target_column], positive_label, negative_label
    )
    validation_metrics = _measure_oof_predictions(
        tuning.validation_predictions,
        target_column,
        positive_label,
        negative_label,
        model.decision_threshold,
    )
    test_metrics = _measure_model_performance(
        model, test_data, test_data[target_column], positive_label, negative_label
    )
    return (
        model,
        train_metrics,
        validation_metrics,
        test_metrics,
        _tuning_artifacts(tuning, target_column, development_data),
    )

def develop_model_v2(
    industry: str,
    target_column: str,
    positive_label: str,
    negative_label: str,
    threshold_col: str,
    config: CatBoostTrainingConfig | None = None,
) -> tuple:
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

    config = config or CatBoostTrainingConfig()
    prepared_data = _combine_multiple_ticker_in_industry(industry)
    development_data, _, selected_features, removed, folds = (
        _catboost_development_data(
            prepared_data,
            feature_columns,
            target_column,
            positive_label,
            ["Ticker", "Industry"],
            config,
        )
    )
    tuning = optimize_catboost(
        development_data,
        target_column,
        positive_label,
        negative_label,
        selected_features,
        ["Ticker", "Industry"],
        folds,
        config,
        removed,
        paths.get_tuning_study_path(2, industry, target_column),
    )
    model = tuning.model
    train_metrics, test_metrics = _measure_model_performance_for_all_ticker_in_industry(
        industry, model, target_column, positive_label, negative_label, threshold_col
    )
    tickers = (
        pd.read_csv(paths.get_selected_ticker_and_industry_list_path())
        .loc[lambda frame: frame["Industry"] == industry, "Ticker"]
        .tolist()
    )
    validation_metrics = _validation_metrics_by_ticker(
        tuning.validation_predictions,
        model,
        target_column,
        positive_label,
        negative_label,
        threshold_col,
        tickers,
    )
    return (
        model,
        train_metrics,
        validation_metrics,
        test_metrics,
        _tuning_artifacts(tuning, target_column, development_data),
    )

def develop_model_v3(
    target_column: str,
    positive_label: str,
    negative_label: str,
    threshold_col: str,
    config: CatBoostTrainingConfig | None = None,
) -> tuple:
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

    config = config or CatBoostTrainingConfig()
    prepared_data = _combine_multiple_ticker(paths.get_label_dir())
    development_data, _, selected_features, removed, folds = (
        _catboost_development_data(
            prepared_data,
            feature_columns,
            target_column,
            positive_label,
            ["Ticker", "Industry"],
            config,
        )
    )
    tuning = optimize_catboost(
        development_data,
        target_column,
        positive_label,
        negative_label,
        selected_features,
        ["Ticker", "Industry"],
        folds,
        config,
        removed,
        paths.get_tuning_study_path(3, "IHSG", target_column),
    )
    model = tuning.model
    train_metrics, test_metrics = _measure_model_performance_for_all_ticker(
        model, target_column, positive_label, negative_label, threshold_col
    )
    tickers = [file.stem for file in paths.get_label_dir().rglob("*.csv")]
    validation_metrics = _validation_metrics_by_ticker(
        tuning.validation_predictions,
        model,
        target_column,
        positive_label,
        negative_label,
        threshold_col,
        tickers,
    )
    return (
        model,
        train_metrics,
        validation_metrics,
        test_metrics,
        _tuning_artifacts(tuning, target_column, development_data),
    )

def develop_model_v4(rolling_window: int, positive_label: str, negative_label: str) -> (any, dict, dict, str):
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
    feature_columns, target_column, threshold_column = _get_combined_forecasts_features_target_threshold(rolling_window)

    prepared_data = _combine_multiple_ticker(str(paths.get_combined_forecasts_window_dir(rolling_window)))
    
    cleaned_data = prepared_data.dropna(subset=[target_column])

    train_feature, train_target, test_feature, test_target, cv_split = _split_data_to_train_val_test_multiple(cleaned_data, feature_columns, target_column)

    search = _initializes_fit_tune_logistic_regression_with_bayesian_optimization(
        train_feature, train_target, cv_split, return_search=True
    )
    best_params = search.best_params_
    train_indices = np.where(cv_split.test_fold != 0)[0]
    validation_indices = np.where(cv_split.test_fold == 0)[0]
    validation_model = _initializes_fit_logistic_regression(
        train_feature.iloc[train_indices],
        train_target.iloc[train_indices],
        best_params,
    )
    validation_probability = validation_model.predict_proba(
        train_feature.iloc[validation_indices]
    )[:, list(validation_model.classes_).index(positive_label)]
    validation_rows = cleaned_data.loc[
        train_feature.iloc[validation_indices].index
    ].copy()
    validation_rows["Probability"] = validation_probability

    validation_metric_rows = []
    tickers = [
        file.stem
        for file in paths.get_combined_forecasts_window_dir(rolling_window).rglob("*.csv")
    ]
    for ticker in tickers:
        ticker_predictions = validation_rows[validation_rows["Ticker"] == ticker]
        if ticker_predictions.empty:
            raise ValueError(f"No validation predictions for {ticker}")
        metrics = pd.DataFrame(
            _measure_oof_predictions(
                ticker_predictions,
                target_column,
                positive_label,
                negative_label,
                0.5,
            )
        )
        metrics["Ticker"] = ticker
        metrics["Threshold"] = ticker_predictions[threshold_column].iloc[0]
        validation_metric_rows.append(metrics)

    final_model = _initializes_fit_logistic_regression(
        train_feature, train_target, best_params
    )
    final_train_metrics, final_test_metrics = _measure_model_performance_on_forecast_features_for_all_ticker(final_model, rolling_window, positive_label, negative_label)
    validation_metrics = pd.concat(
        validation_metric_rows, ignore_index=True
    ).to_dict(orient="list")
    artifacts = {
        "trials": pd.DataFrame(search.cv_results_),
        "metadata": {
            "target_column": target_column,
            "best_parameters": search.best_params_,
            "validation_selection_only": True,
        },
    }
    return (
        final_model,
        final_train_metrics,
        validation_metrics,
        final_test_metrics,
        threshold_column,
        artifacts,
    )

def process_single_model(args_tuple):
    """
    Utilize label data to create a machine learning model.

    Args:
        args_tuple: Tuple containing (label_file, label_type, rolling_window, model_version)

    Returns:
        Tuple of (failed_process, metrics_list)
    """
    identifier, label_type, rolling_window, model_version = args_tuple[:4]
    training_config = (
        args_tuple[4]
        if len(args_tuple) > 4 and args_tuple[4] is not None
        else CatBoostTrainingConfig()
    )
 
    target_col, threshold_col, pos_label, neg_label = get_label_config(
            label_type, rolling_window
        )
    failed_process = []
    metrics_list = []

    try:
        if model_version in [1, 2, 3]:
            if model_version == 1:
                model, train_metrics, validation_metrics, test_metrics, artifacts = develop_model_v1(
                    identifier, target_col, pos_label, neg_label, training_config
                )

            elif model_version == 2:
                model, train_metrics, validation_metrics, test_metrics, artifacts = develop_model_v2(
                    identifier, target_col, pos_label, neg_label, threshold_col, training_config
                )
            
            elif model_version == 3:
                model, train_metrics, validation_metrics, test_metrics, artifacts = develop_model_v3(
                    target_col, pos_label, neg_label, threshold_col, training_config
                )
                        
        elif model_version == 4:
            (
                model,
                train_metrics,
                validation_metrics,
                test_metrics,
                threshold_col,
                artifacts,
            ) = develop_model_v4(rolling_window, pos_label, neg_label)

        _save_model(model, model_version, label_type, identifier, rolling_window)
        if artifacts:
            _save_training_artifacts(
                artifacts,
                model_version,
                label_type,
                identifier,
                rolling_window,
            )
        
        metrics_df = _combine_metrics(
            identifier,
            model_version,
            train_metrics,
            test_metrics,
            threshold_col,
            validation_metrics,
        )

        metrics_list.append((label_type, rolling_window, metrics_df))
                
    except Exception as e:
        failed_process.append(
            (
                identifier,
                label_type,
                rolling_window,
                f"{e}\n{traceback.format_exc()}",
            )
        )

    return failed_process, metrics_list
