from pathlib import Path

import numpy as np
import pandas as pd
from skopt.space import Real
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from prepareTechnicalIndicators.helper import get_all_technical_indicators
from combineForecasts.helper import _get_combined_forecasts_features_target_threshold
from trainModels.features import add_cross_sectional_features, attach_ticker_metadata
from trainModels.splits import (
    purge_development_test_boundary,
    purge_overlapping_label_periods,
)
from utils.pipeline import get_split_dates, get_split_masks
from utils import paths


def _combine_multiple_ticker_in_industry(industry: str) -> pd.DataFrame:
    """
    (Internal Helper) Combine all ticker in an industry will be used as training data

    Args:
        industry (str): The name of the industry in which the ticker will be selected

    Returns:
        pd.DataFrame: A pandas dataframe containing all the ticker in an industry
    """
    ticker_industry_df = pd.read_csv(paths.get_selected_ticker_and_industry_list_path())
    
    selected_ticker_industry_df = ticker_industry_df[ticker_industry_df['Industry'] == industry]
    
    selected_ticker = selected_ticker_industry_df['Ticker'].values
    
    selected_ticker_df = pd.concat(
                            (
                                attach_ticker_metadata(
                                    pd.read_csv(paths.get_label_path(ticker)), ticker
                                )
                                for ticker in selected_ticker
                            )
                        ) \
                            .sort_values('Date', ascending=True) \
                            .reset_index(drop=True)

    return selected_ticker_df

def _combine_multiple_ticker(csv_folder_path) -> pd.DataFrame:
    """
    (Internal Helper) Combine data from multiple ticker into a single pandas dataframe

        csv_folder_path (Path): The path to the folder containing the data

    Returns:
        pd.DataFrame: A pandas dataframe containing all the selected ticker
    """
    csv_folder_path = Path(csv_folder_path)
    all_ticker_path = csv_folder_path.rglob("*.csv")
    
    selected_ticker_df = pd.concat(
                            (
                                attach_ticker_metadata(
                                    pd.read_csv(ticker_path), ticker_path.stem
                                )
                                for ticker_path in all_ticker_path
                            )
                        ) \
                            .sort_values('Date', ascending=True) \
                            .reset_index(drop=True)


    return selected_ticker_df


def _split_development_and_locked_test(
    data: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pre-test development data and a test set that tuning never sees."""
    splits = get_split_dates(target_column)
    train_val_mask, _, _, test_mask, _ = get_split_masks(data, splits)
    development_mask = purge_development_test_boundary(
        data, train_val_mask, target_column
    )
    development_data = data.loc[development_mask].copy().reset_index(drop=True)
    test_data = data.loc[test_mask].copy().reset_index(drop=True)
    if development_data.empty or test_data.empty:
        raise ValueError("Development or locked test split is empty")
    return development_data, test_data

def _split_data_to_train_val_test_single(data: pd.DataFrame, feature_columns: list, target_column: str) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, PredefinedSplit):
    """
    (Internal Helper) Splits time-series data into training, validation, and testing sets

    This function implements a time-based split crucial for financial forecasting:
    - Training Set: All data preceding the test set
    - Validation Set (for Hyperparameter Tuning): The last 40 days of the training set
    - Test Set: The last 80 days from current date

    Args:
        data (pd.DataFrame): The complete DataFrame containing features and the target
        feature_columns (list): A list of column names to be used as features
        target_column (str): The name of the column to be used as the target variable

    Returns:
        tuple: A tuple containing:
               - train_feature (pd.DataFrame): Features for the training set
               - train_target (pd.Series): Target for the training set
               - test_feature (pd.DataFrame): Features for the test set
               - test_target (pd.Series): Target for the test set
               - predefined_split_index (PredefinedSplit): An index for cross-validation
                 that designates the last 40 days of the training data as the validation set
    """
    splits = get_split_dates(target_column)
    train_val_mask, train_mask, val_mask, test_mask, _ = get_split_masks(data, splits)
    train_val_mask, val_mask = purge_overlapping_label_periods(
        data,
        train_val_mask,
        train_mask,
        val_mask,
        target_column,
    )

    train_data = data[train_val_mask].copy()
    test_data = data[test_mask].copy()

    train_feature = train_data[feature_columns]
    train_target = train_data[target_column]
    test_feature = test_data[feature_columns]
    test_target = test_data[target_column]
    
    val_mask_train = val_mask[train_val_mask]
    
    split_index = np.full(len(train_feature), -1, dtype=int)
    split_index[val_mask_train.values] = 0    
    predefined_split_index = PredefinedSplit(test_fold=split_index)
    
    return train_feature, train_target, test_feature, test_target, predefined_split_index

def _split_data_to_train_val_test_multiple(data: pd.DataFrame, feature_columns: list, target_column: str) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, PredefinedSplit):
    """
    (Internal Helper) Splits time-series data into training, validation, and testing sets

    This function implements a time-based split crucial for financial forecasting:
    - Training Set: All data preceding the test set
    - Validation Set (for Hyperparameter Tuning): The last 40 days of the training set
    - Test Set: The last 80 days from current date

    Args:
        data (pd.DataFrame): The complete DataFrame containing features and the target
        feature_columns (list): A list of column names to be used as features
        target_column (str): The name of the column to be used as the target variable

    Returns:
        tuple: A tuple containing:
               - train_feature (pd.DataFrame): Features for the training set
               - train_target (pd.Series): Target for the training set
               - test_feature (pd.DataFrame): Features for the test set
               - test_target (pd.Series): Target for the test set
               - predefined_split_index (PredefinedSplit): An index for cross-validation
                 that designates the last 40 days of the training data as the validation set
    """

    splits = get_split_dates(target_column)
    train_val_mask, train_mask, val_mask, test_mask, _ = get_split_masks(data, splits)
    train_val_mask, val_mask = purge_overlapping_label_periods(
        data,
        train_val_mask,
        train_mask,
        val_mask,
        target_column,
    )

    train_data = data[train_val_mask].copy()
    test_data = data[test_mask].copy()
    
    train_feature = train_data[feature_columns]
    train_target = train_data[target_column]
    test_feature = test_data[feature_columns]
    test_target = test_data[target_column]
    
    val_mask_train = val_mask[train_val_mask]
    
    split_index = np.full(len(train_feature), -1, dtype=int)
    split_index[val_mask_train.values] = 0    
    predefined_split_index = PredefinedSplit(test_fold=split_index)
    
    return train_feature, train_target, test_feature, test_target, predefined_split_index

def _initializes_fit_tune_logistic_regression_with_bayesian_optimization(
    train_feature: np.array,
    train_target: np.array,
    predefined_split_index: PredefinedSplit,
    return_search: bool = False,
) -> any:
    """
    (Internal Helper) Initializes, fits, and tunes a Logistic Regression model using Bayesian Optimization.

    This function uses BayesSearchCV to efficiently search for the optimal
    hyperparameters (C and l1_ratio) for a Logistic Regression model.
    It validates performance using a predefined time-series split and fits
    the best-found model on the entire training dataset.

    Args:
        train_feature (np.array): The feature set for training
        train_target (np.array): The target variable for training
        predefined_split_index (PredefinedSplit): The cross-validation strategy

    Returns:
        LogisticRegression: The best-performing model found by the search
    """
    val_indices = np.where(predefined_split_index.test_fold == 0)[0]
    train_indices = np.where(predefined_split_index.test_fold != 0)[0]

    if len(np.unique(train_target[train_indices])) == 1:
        raise ValueError("The train target contains only one unique value.")

    if len(np.unique(train_target[val_indices])) == 1:
        raise ValueError("The validation target contains only one unique value")
    scoring_method = 'roc_auc'

    base_model = LogisticRegression(
        solver='saga',
        penalty='elasticnet',
        max_iter=1500, 
        class_weight='balanced',
        random_state=10120024
    )
    
    model_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('lr', base_model)
    ])

    search_spaces = {
        'lr__C': Real(1e-4, 1e3, prior='log-uniform'), 
        'lr__l1_ratio': Real(0.0, 1.0)
    }
    
    hyper_tune_search = BayesSearchCV(
        estimator=model_pipeline,
        search_spaces=search_spaces,
        n_iter=30,
        cv=predefined_split_index,
        scoring=scoring_method,
        n_jobs=1,
        random_state=10120024,
        verbose=0
    )

    hyper_tune_search.fit(train_feature, train_target)
    best_model = hyper_tune_search.best_estimator_

    return hyper_tune_search if return_search else best_model

def _initializes_fit_logistic_regression(train_feature: np.array, train_target: np.array, best_params: dict) -> any:
    """
    (Internal Helper) Initializes and fits a Logistic Regression pipeline using given parameters

    Args:
        train_feature (np.array): The feature set for training
        train_target (np.array): The target variable for training
        best_params (dict): The hyperparameters identified from previous tuning

    Returns:
        Pipeline: The fitted pipeline model containing scaler and logistics regression
    """
    base_model = LogisticRegression(
        solver='saga',
        penalty='elasticnet',
        max_iter=1500, 
        class_weight='balanced', 
        random_state=10120024
    )
    model = Pipeline([
        ('scaler', RobustScaler()), 
        ('lr', base_model)
    ])
    model.set_params(**best_params)

    model.fit(train_feature, train_target)
    
    return model

def _calculate_classification_metrics(target_true: np.array, target_pred: np.array, positive_label: str, negative_label: str) -> (np.array, np.array, np.array, np.array):
    """
    (Internal Helper) Calculates key classification metrics for a binary prediction task

    Args:
        target_true (np.array): The ground truth labels
        target_pred (np.array): The predicted labels from the model
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label

    Returns:
        tuple: A tuple containing accuracy, precision for both classes, and recall for both classes
    """
    accuracy = accuracy_score(target_true, target_pred)
    precision_positive = precision_score(target_true, target_pred, pos_label=positive_label, zero_division=0)
    precision_negative = precision_score(target_true, target_pred, pos_label=negative_label, zero_division=0)
    recall_positive = recall_score(target_true, target_pred, pos_label=positive_label, zero_division=0)
    recall_negative = recall_score(target_true, target_pred, pos_label=negative_label, zero_division=0)

    return accuracy, precision_positive, precision_negative, recall_positive, recall_negative

def _calculate_gini(model: any, target_true: np.array, target_pred_proba: np.array, positive_label: str) -> float:
    """
    (Internal Helper) Calculates the Gini coefficient from the model's prediction probabilities

    The Gini coefficient is a common metric for evaluating binary classification
    models and is derived from the Area Under the ROC Curve (AUC)
    Formula: Gini = 2 * AUC - 1

    Args:
        model (any): The trianed catboost model for binary classifications
        target_true (np.array): The true labels of the target variable
        target_pred_proba (np.array): The predicted probabilities for each class
        positive_label (str): The positive class of the predicted label

    Returns:
        float: The calculated Gini coefficient, or 0.0 if AUC cannot be calculated
    """
    try:
        positive_class_true = (target_true == positive_label).astype(int)
        
        positive_class_index = np.where(model.classes_ == positive_label)[0][0]
        positive_class_prob = target_pred_proba[:, positive_class_index]
        auc = roc_auc_score(positive_class_true, positive_class_prob)
        gini = 2 * auc - 1

    except (ValueError, IndexError):
        gini = np.nan

    return gini

def _metrics_from_probability(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    positive_label: str,
    negative_label: str,
    decision_threshold: float,
    top_fraction: float = 0.10,
) -> dict:
    target_array = np.asarray(target)
    binary_target = (target_array == positive_label).astype(int)
    target_pred = np.where(
        probability >= decision_threshold, positive_label, negative_label
    )
    accuracy, prec_positive, prec_negative, rec_positive, rec_negative = (
        _calculate_classification_metrics(
            target_array, target_pred, positive_label, negative_label
        )
    )
    if len(np.unique(binary_target)) < 2:
        auc = np.nan
        average_precision = np.nan
        gini = np.nan
    else:
        auc = float(roc_auc_score(binary_target, probability))
        average_precision = float(
            average_precision_score(binary_target, probability)
        )
        gini = 2 * auc - 1

    prevalence = float(np.mean(binary_target))
    top_count = max(1, int(np.ceil(len(probability) * top_fraction)))
    top_indices = np.argsort(probability)[-top_count:]
    top_precision = float(np.mean(binary_target[top_indices]))
    lift = np.nan if prevalence == 0 else top_precision / prevalence
    clipped_probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        'Accuracy': [accuracy],
        f'Precision {positive_label}': [prec_positive],
        f'Precision {negative_label}': [prec_negative],
        f'Recall {positive_label}': [rec_positive],
        f'Recall {negative_label}': [rec_negative],
        'ROC AUC': [auc],
        'Average Precision': [average_precision],
        'Gini': [gini],
        f'Precision Top {int(top_fraction * 100)}%': [top_precision],
        f'Lift Top {int(top_fraction * 100)}%': [lift],
        'Brier Score': [brier_score_loss(binary_target, clipped_probability)],
        'Log Loss': [
            log_loss(binary_target, clipped_probability, labels=[0, 1])
        ],
        'Positive Rate': [prevalence],
        'Predicted Positive Rate': [float(np.mean(target_pred == positive_label))],
        'Decision Threshold': [float(decision_threshold)],
        'Observations': [len(target_array)],
        'Positive Observations': [int(binary_target.sum())],
    }


def _measure_model_performance(model: any, feature: np.array, target: np.array, positive_label: str, negative_label: str) -> dict:
    """
    Measures and reports the performance of the model on a given dataset

    Args:
        model: The trained classifier model
        feature (np.array): The feature set (e.g., train_feature or test_feature)
        target (np.array): The corresponding true target labels
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label

    Returns:
        dict: A dictionary containing all calculated performance metrics.
    """
    target_pred_proba = model.predict_proba(feature)
    positive_class_index = np.where(model.classes_ == positive_label)[0][0]
    decision_threshold = float(getattr(model, "decision_threshold", 0.5))
    return _metrics_from_probability(
        target,
        target_pred_proba[:, positive_class_index],
        positive_label,
        negative_label,
        decision_threshold,
    )


def _measure_oof_predictions(
    predictions: pd.DataFrame,
    target_column: str,
    positive_label: str,
    negative_label: str,
    decision_threshold: float,
) -> dict:
    metrics = _metrics_from_probability(
        predictions[target_column],
        predictions["Probability"].to_numpy(),
        positive_label,
        negative_label,
        decision_threshold,
    )
    unique_dates = predictions["Date"].astype(str).unique()
    bootstrap_gini = []
    if len(unique_dates) >= 20:
        date_values = predictions["Date"].astype(str).to_numpy()
        binary_target = (
            predictions[target_column].to_numpy() == positive_label
        ).astype(int)
        probability = predictions["Probability"].to_numpy()
        random = np.random.default_rng(10120024)
        positions_by_date = {
            date: np.flatnonzero(date_values == date) for date in unique_dates
        }
        for _ in range(200):
            sampled_dates = random.choice(
                unique_dates, size=len(unique_dates), replace=True
            )
            sampled_positions = np.concatenate(
                [positions_by_date[date] for date in sampled_dates]
            )
            sampled_target = binary_target[sampled_positions]
            if len(np.unique(sampled_target)) < 2:
                continue
            auc = roc_auc_score(sampled_target, probability[sampled_positions])
            bootstrap_gini.append(2 * auc - 1)
    metrics["Gini CI 95% Lower"] = [
        float(np.quantile(bootstrap_gini, 0.025)) if bootstrap_gini else np.nan
    ]
    metrics["Gini CI 95% Upper"] = [
        float(np.quantile(bootstrap_gini, 0.975)) if bootstrap_gini else np.nan
    ]
    return metrics


def _prepare_ticker_data_for_model(
    prepared_data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    enriched = attach_ticker_metadata(prepared_data, ticker)
    enriched, _ = add_cross_sectional_features(enriched)
    return enriched

def _measure_model_performance_on_single_ticker(prepared_data: pd.DataFrame, model: any, feature_columns: str, target_column: str, positive_label: str, negative_label: str) -> (pd.DataFrame, pd.DataFrame):
    """
    (Internal Helper) Measures and reports the performance of the model on a given ticker

    Args:
        prepared_data (pd.DataFrame): A pandas dataframe containing the features and target
        model (any): The trained classifier model
        feature (np.array): The feature set (e.g., train_feature or test_feature)
        target (np.array): The corresponding true target labels
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label

    Returns:
        Tuple: A tuple containing the model's performance on trainings and testing data, stored as a pandas dataframe
    """
    model_features = list(getattr(model, "feature_names_", feature_columns))
    train_feature, train_target, test_feature, test_target, cv_split = _split_data_to_train_val_test_multiple(
        prepared_data.dropna(subset=[target_column]), model_features, target_column
    )

    train_metrics = _measure_model_performance(model, train_feature, train_target, positive_label, negative_label)
    test_metrics = _measure_model_performance(model, test_feature, test_target, positive_label, negative_label)

    train_metrics_df = pd.DataFrame(train_metrics)
    test_metrics_df = pd.DataFrame(test_metrics)
    
    return train_metrics_df, test_metrics_df

def _measure_model_performance_for_all_ticker_in_industry(industry: str, model: any, target_column: str, positive_label: str, negative_label: str, threshold_col: str) -> (pd.DataFrame, pd.DataFrame):
    """
    (Internal Helper) Measures and reports the performance of the model on a given industry

    Args:
        industry (str): The name of the industry being worked on
        model (any): The trained classifier model
        target_column (str): The name of the target variable column
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label
        threshold_col (str): The name of the columns used as a threshold during the creation of the label

    Returns:
        Tuple: A tuple containing the model's performance on trainings and testing data, stored as a pandas dataframe
    """
    ticker_industry_df = pd.read_csv(paths.get_selected_ticker_and_industry_list_path())
    all_tickers = ticker_industry_df.loc[ticker_industry_df['Industry'] == industry, 'Ticker'].values

    all_ticker_train_metrics_df = pd.DataFrame()
    all_ticker_test_metrics_df = pd.DataFrame()

    feature_columns = get_all_technical_indicators()
    failures = []

    for ticker in all_tickers:
        try:
            prepared_data = _prepare_ticker_data_for_model(
                pd.read_csv(paths.get_label_path(ticker)), ticker
            )
            ticker_train_metrics_df, ticker_test_metrics_df = _measure_model_performance_on_single_ticker(prepared_data, model, feature_columns, target_column, positive_label, negative_label)

            ticker_train_metrics_df['Ticker'] = ticker
            ticker_test_metrics_df['Ticker'] = ticker

            ticker_train_metrics_df['Threshold'] = prepared_data[threshold_col].iloc[0]
            ticker_test_metrics_df['Threshold'] = prepared_data[threshold_col].iloc[0]

            all_ticker_train_metrics_df = pd.concat((all_ticker_train_metrics_df, ticker_train_metrics_df))
            all_ticker_test_metrics_df = pd.concat((all_ticker_test_metrics_df, ticker_test_metrics_df))
        except Exception as e:
            failures.append(f"{ticker}: {e}")

    if failures:
        raise RuntimeError(
            "Industry evaluation failed for ticker(s): " + "; ".join(failures[:10])
        )

    all_ticker_train_metrics = all_ticker_train_metrics_df.to_dict(orient='list')
    all_ticker_test_metrics = all_ticker_test_metrics_df.to_dict(orient='list')

    return all_ticker_train_metrics, all_ticker_test_metrics

def _measure_model_performance_for_all_ticker(model: any, target_column: str, positive_label: str, negative_label: str, threshold_col: str) -> (pd.DataFrame, pd.DataFrame):
    """
    (Internal Helper) Measures and reports the performance of the model on a given top IHSG valuation

    Args:s
        model (any): The trained classifier model
        feature (np.array): The feature set (e.g., train_feature or test_feature)
        target (np.array): The corresponding true target labels
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label
        threshold_col (str): The name of the columns used as a threshold during the creation of the label

    Returns:
        Tuple: A tuple containing the model's performance on trainings and testing data, stored as a pandas dataframe
    """
    all_tickers = [file.stem for file in paths.get_label_dir().rglob('*.csv')]
    
    all_ticker_train_metrics_df = pd.DataFrame()
    all_ticker_test_metrics_df = pd.DataFrame()

    feature_columns = get_all_technical_indicators()
    failures = []

    for ticker in all_tickers:
        try:
            prepared_data = _prepare_ticker_data_for_model(
                pd.read_csv(paths.get_label_path(ticker)), ticker
            )
            ticker_train_metrics_df, ticker_test_metrics_df = _measure_model_performance_on_single_ticker(prepared_data, model, feature_columns, target_column, positive_label, negative_label)
    
            ticker_train_metrics_df['Ticker'] = ticker
            ticker_test_metrics_df['Ticker'] = ticker

            ticker_train_metrics_df['Threshold'] = prepared_data[threshold_col].iloc[0]
            ticker_test_metrics_df['Threshold'] = prepared_data[threshold_col].iloc[0]
    
            all_ticker_train_metrics_df = pd.concat((all_ticker_train_metrics_df, ticker_train_metrics_df))
            all_ticker_test_metrics_df = pd.concat((all_ticker_test_metrics_df, ticker_test_metrics_df))
        except Exception as e:
            failures.append(f"{ticker}: {e}")

    if failures:
        raise RuntimeError(
            "Market evaluation failed for ticker(s): " + "; ".join(failures[:10])
        )

    all_ticker_train_metrics = all_ticker_train_metrics_df.to_dict(orient='list')
    all_ticker_test_metrics = all_ticker_test_metrics_df.to_dict(orient='list')

    return all_ticker_train_metrics, all_ticker_test_metrics

def _measure_model_performance_on_forecast_features_for_all_ticker(model: any, rolling_window: int, positive_label: str, negative_label: str) -> (pd.DataFrame, pd.DataFrame):
    """
    (Internal Helper) Measures and reports the performance of the model on a given top IHSG valuation, with the forecasts as the features

    Args:s
        model (any): The trained classifier model
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label

    Returns:
        Tuple: A tuple containing the model's performance on trainings and testing data, stored as a pandas dataframe
    """
    all_tickers = [file.stem for file in paths.get_combined_forecasts_window_dir(rolling_window).rglob('*.csv')]
    
    all_ticker_train_metrics_df = pd.DataFrame()
    all_ticker_test_metrics_df = pd.DataFrame()

    for ticker in all_tickers:
        try:
            prepared_data = pd.read_csv(paths.get_combined_forecasts_path(rolling_window, ticker))
            
            feature_columns, target_column, threshold_column = _get_combined_forecasts_features_target_threshold(rolling_window)
            
            cleaned_data = prepared_data.dropna(subset=[target_column])

            ticker_train_metrics_df, ticker_test_metrics_df = _measure_model_performance_on_single_ticker(cleaned_data, model, feature_columns, target_column, positive_label, negative_label)
    
            ticker_train_metrics_df['Ticker'] = ticker
            ticker_test_metrics_df['Ticker'] = ticker

            ticker_train_metrics_df['Threshold'] = prepared_data[threshold_column].iloc[0]
            ticker_test_metrics_df['Threshold'] = prepared_data[threshold_column].iloc[0]
    
            all_ticker_train_metrics_df = pd.concat((all_ticker_train_metrics_df, ticker_train_metrics_df))
            all_ticker_test_metrics_df = pd.concat((all_ticker_test_metrics_df, ticker_test_metrics_df))
        except Exception as e:
            print(e)
            pass

    all_ticker_train_metrics = all_ticker_train_metrics_df.to_dict(orient='list')
    all_ticker_test_metrics = all_ticker_test_metrics_df.to_dict(orient='list')

    return all_ticker_train_metrics, all_ticker_test_metrics
