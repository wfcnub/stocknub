import os
import json
import pickle
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from utils.pipeline import (
    get_split_dates, 
    get_split_masks
)

def _prepare_data(window: str):
    """
    (Internal Helper) Prepares and merges the forecast data, splitting dates, and model performance metrics.
    
    Args:
        window (str): The rolling window configuration (e.g., '10dd').
        
    Returns:
        tuple: A tuple containing:
            - joined_train_data (pd.DataFrame): The merged training dataset.
            - joined_test_data (pd.DataFrame): The merged testing dataset.
            - joined_forecast_data (pd.DataFrame): The merged forecasting dataset.
            - feature_col (str): The column name representing the forecast feature.
            - target_col (str): The column name representing the median gain target.
            - score_col (str): The column name where the resulting score will be stored.
    """
    target_col = f'Median Gain {window}'
    feature_col = f'Forecast High Gain {window}'
    score_col = f'Score {window}'
    selected_columns = ['Date', 'Ticker', feature_col, target_col]

    forecast_dir = Path(f'data/stock/forecast/model_v4/medianGain/{window}')
    all_file_paths = list(forecast_dir.rglob('*.csv'))

    all_data = pd.DataFrame()
    for file_path in all_file_paths:
        data = pd.read_csv(file_path)
        data['Ticker'] = file_path.stem
        all_data = pd.concat((all_data, data), ignore_index=True)

    splits = get_split_dates(target_col)
    train_val_mask, train_mask, val_mask, test_mask, forecast_mask = get_split_masks(all_data, splits)
    
    train_data = all_data.loc[train_mask, selected_columns].copy()
    test_data = all_data.loc[test_mask, selected_columns].copy()
    forecast_data = all_data.loc[forecast_mask, selected_columns].copy()

    perf_path = Path(f'data/stock/model_v4/performance/medianGain/{window}.csv')
    model_performance = pd.read_csv(perf_path)
    
    joined_train_data = pd.merge(
        train_data,
        model_performance[['Ticker', 'Train - Gini']],
        on='Ticker',
        how='inner'
    )

    joined_test_data = pd.merge(
        test_data,
        model_performance[['Ticker', 'Test - Gini']],
        on='Ticker',
        how='inner'
    )

    joined_forecast_data = pd.merge(
        forecast_data,
        model_performance[['Ticker', 'Test - Gini']],
        on='Ticker',
        how='inner'
    )

    return joined_train_data, joined_test_data, joined_forecast_data, feature_col, target_col, score_col

def _train_model(joined_train_data: pd.DataFrame, feature_col: str, target_col: str, window: str):
    """
    (Internal Helper) Trains a Logistic Regression model to calculate composite scores based on forecast features and performance metrics (Gini).
    
    Args:
        joined_train_data (pd.DataFrame): The training dataset containing features and target.
        feature_col (str): The column name of the primary forecast feature.
        target_col (str): The column name of the target variable.
        window (str): The rolling window configuration used, which dictates the output filename.
        
    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    model = LogisticRegression()
    
    train_feature = joined_train_data[[feature_col, 'Train - Gini']].values
    train_target = joined_train_data[target_col]

    model.fit(train_feature, train_target)

    filepath = Path(f"data/stock/score/{window}.pkl")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    return model

def _infer_and_export(model: LogisticRegression, joined_test_data: pd.DataFrame, joined_forecast_data: pd.DataFrame, feature_col: str, target_col: str, score_col: str, window: str):
    """
    (Internal Helper) Conducts inference on testing and forecasting datasets using the trained scorecard model, 
    then exports the final scores per ticker.
    
    Args:
        model (LogisticRegression): The trained model used to predict probability scores.
        joined_test_data (pd.DataFrame): The testing dataset to infer scores on.
        joined_forecast_data (pd.DataFrame): The active forecast dataset to infer scores on.
        feature_col (str): The column name of the primary forecast feature.
        target_col (str): The column name of the target variable.
        score_col (str): The resulting predicted score column name.
        window (str): The rolling window configuration, dictating the saving path.
    """
    test_feature = joined_test_data[[feature_col, 'Test - Gini']].values
    joined_test_data[score_col] = model.predict_proba(test_feature)[:, 0]

    forecast_feature = joined_forecast_data[[feature_col, 'Test - Gini']].values
    joined_forecast_data[score_col] = model.predict_proba(forecast_feature)[:, 0]

    joined_test_data.drop(columns=[feature_col, target_col, 'Test - Gini'], inplace=True)
    joined_forecast_data.drop(columns=[feature_col, target_col, 'Test - Gini'], inplace=True)

    joined_test_forecast_data = pd.concat([joined_test_data, joined_forecast_data], ignore_index=True)

    save_dir = Path(f'data/stock/score/{window}')
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for ticker in joined_test_forecast_data['Ticker'].unique():
        ticker_test_data = joined_test_forecast_data[joined_test_forecast_data['Ticker'] == ticker]
        ticker_test_data.to_csv(save_dir / f'{ticker}.csv', index=False)

    return