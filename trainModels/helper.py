import shutil
import pickle
import json
import numpy as np
import pandas as pd

from utils import paths

def _ensure_directories_exist(model_version: int, label_types: list) -> None:
    """
    (Internal Helper) Ensure all required directories exist before training.

    Args:
    model_version (int): The version of model currently being developed
    label_types (list): A list containing all the types of label
    """
    for label_type in label_types:
        model_pkl_folder_path = paths.get_model_dir(model_version, label_type)
        model_performance_folder_path = paths.get_model_performance_dir(model_version, label_type)

        if model_pkl_folder_path.exists():
            shutil.rmtree(model_pkl_folder_path)

        model_pkl_folder_path.mkdir(parents=True, exist_ok=True)
        paths.get_model_artifact_dir(model_version, label_type).mkdir(
            parents=True, exist_ok=True
        )
        
        if model_performance_folder_path.exists():
            shutil.rmtree(model_performance_folder_path)

        model_performance_folder_path.mkdir(parents=True, exist_ok=True)
    
    return

def _save_model(model: any, model_version: int, label_type: str, identifier: str, window: int) -> None:
    """
    (Internal Helper) Save a trained model to file

    Args:
    model (any): The machine learning model that has been developed
    model_version (int): The version of machine learning model being developed
    label_type (str): The label used for developing the model
    identifier (str): An identifier for saving the model, could be ticker or industry
    window (int): The window used for generating the label
    """
    filepath = paths.get_model_path(model_version, label_type, identifier, window)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    
    return


def _save_training_artifacts(
    artifacts: dict,
    model_version: int,
    label_type: str,
    identifier: str,
    window: int,
) -> None:
    artifact_dir = paths.get_model_artifact_dir(model_version, label_type)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        if isinstance(value, pd.DataFrame):
            filepath = paths.get_model_artifact_path(
                model_version, label_type, identifier, window, name, "csv"
            )
            value.to_csv(filepath, index=False)
        else:
            filepath = paths.get_model_artifact_path(
                model_version, label_type, identifier, window, name, "json"
            )
            with open(filepath, "w") as file:
                json.dump(value, file, indent=2, default=str)

def _combine_metrics(
    ticker: str,
    model_version: int,
    train_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    threshold_col: str,
    validation_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    (Internal Helper) Combine train and test metrics into a single DataFrame row.

    Args:
    ticker (str): The name of the ticker being worked on
    model_version (int): The version of machine learning model being developed
    train_metrics (pd.DataFrame): The model's performance metrics on training data
    test_metrics (pd.DataFrame): The model's performance metrics on testing data
    threshold_col (str): The name of the threshold column

    Returns:
    pd.DataFrame: A pandas dataframe containing all the training and testing metrics of the model
    """
    train_df = pd.DataFrame(train_metrics)
    train_df.columns = [f"Train - {col}" for col in train_df.columns]

    test_df = pd.DataFrame(test_metrics)
    test_df.columns = [f"Test - {col}" for col in test_df.columns]

    frames = [train_df]
    if validation_metrics is not None:
        validation_df = pd.DataFrame(validation_metrics)
        validation_df.columns = [f"Validation - {col}" for col in validation_df.columns]
        frames.append(validation_df)
    frames.append(test_df)
    result = pd.concat(frames, axis=1)

    if model_version == 1:
        result.insert(0, "Ticker", ticker)

        threshold_value = pd.read_csv(
            paths.get_label_path(ticker), usecols=[threshold_col], nrows=1
        )[threshold_col].iloc[0]
        result["Threshold"] = threshold_value

    elif model_version in [2, 3, 4]:
        ticker_columns = [
            col for col in result.columns if col.split(" - ", 1)[-1] == "Ticker"
        ]
        threshold_columns = [
            col for col in result.columns if col.split(" - ", 1)[-1] == "Threshold"
        ]
        if not ticker_columns or not threshold_columns:
            raise ValueError("Pooled metrics must include Ticker and Threshold")
        canonical_ticker = result[ticker_columns[0]].values
        canonical_threshold = result[threshold_columns[0]].values
        for column in ticker_columns[1:]:
            if not np.array_equal(canonical_ticker, result[column].values):
                raise ValueError("Metric rows are not aligned by ticker")
        for column in threshold_columns[1:]:
            if not np.allclose(
                canonical_threshold.astype(float),
                result[column].values.astype(float),
                equal_nan=True,
            ):
                raise ValueError("Metric thresholds are not aligned")
        result.insert(0, "Ticker", canonical_ticker)
        result["Threshold"] = canonical_threshold
        result.drop(columns=ticker_columns + threshold_columns, inplace=True)
    
    return result
