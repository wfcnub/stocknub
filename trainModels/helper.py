import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from camel_converter import to_camel
from utils.pipeline import get_label_config

def _ensure_directories_exist(model_version: int, label_types: list) -> None:
    """
    (Internal Helper) Ensure all required directories exist before training.

    Args:
    model_version (int): The version of model currently being developed
    label_types (list): A list containing all the types of label
    """
    for label_type in label_types:
        camel_label = to_camel(label_type)
        Path(f"data/stock/model_v{model_version}/{camel_label}").mkdir(
            parents=True, exist_ok=True
        )
        Path(f"data/stock/model_v{model_version}/performance/{camel_label}").mkdir(
            parents=True, exist_ok=True
        )
    
    return

def _save_model(model: any, model_version: int, label_type: str, identifier: str, window: int) -> None:
    """
    (Internal Helper) Save a trained model to file

    Args:
    model (any): The machine learning model that has been developed
    model_version (int): The version of machine learning model being developed
    label_type (str): The label used for developing the model
    identifier (str): An identifier for saving the model, could be emiten or industry
    window (int): The window used for generating the label
    """
    camel_label = to_camel(label_type)
    filepath = f"data/stock/model_v{model_version}/{camel_label}/{identifier}-{window}dd.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    
    return

def _combine_metrics(emiten: str, model_version: int, train_metrics: pd.DataFrame, test_metrics: pd.DataFrame, threshold_col: str) -> pd.DataFrame:
    """
    (Internal Helper) Combine train and test metrics into a single DataFrame row.

    Args:
    emiten (str): The name of the emiten being worked on
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

    result = pd.concat([train_df, test_df], axis=1)

    if model_version == 1:
        result = pd.concat([train_df, test_df], axis=1)
        result.insert(0, "Kode", emiten)

        threshold_value = pd.read_csv(f'data/stock/label/{emiten}.csv')[threshold_col].iloc[0]
        result["Threshold"] = threshold_value

    elif model_version in [2, 3, 4]:
        kode_column = [col for col in result.columns if 'Kode' in col]
        threshold_column = [col for col in result.columns if 'Threshold' in col]

        assert len(kode_column) == 2
        assert len(threshold_column) == 2

        assert np.all(result[kode_column[0]].values == result[kode_column[1]].values, axis=0)
        assert np.all(result[threshold_column[0]].values == result[threshold_column[1]].values, axis=0)

        result.insert(0, "Kode", result[kode_column[0]].values)
        result["Threshold"] = result[threshold_column[0]].values
        result.drop(columns=kode_column + threshold_column, inplace=True)
    
    return result