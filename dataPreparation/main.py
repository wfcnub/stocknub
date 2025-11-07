import logging
import pandas as pd

from technicalIndicators.main import generate_all_technical_indicators
from dataPreparation.helper import (
    _download_stock_data,
    _generate_labels_based_on_label_type,
)


def prepare_data_for_modelling_emiten(
    emiten: str,
    start_date: str,
    end_date: str,
    target_column: str,
    label_types: list,
    rolling_windows: list,
) -> pd.DataFrame:
    """
    Orchestrates the full data preparation pipeline for a machine learning model

    This function serves as the main controller, executing a sequence of steps:
    1. Downloads historical stock data
    2. Generates a comprehensive set of technical indicators to be used as model features
    3. Creates the target variable(s)
    4. Cleans the final dataset by removing any rows with missing values (NaNs) that result from the indicator and label generation

    Args:
        emiten (str): The stock emiten symbol
        start_date (str): The start date for the data ('YYYY-MM-DD')
        end_date (str): The end date for the data ('YYYY-MM-DD')
        target_column (str): The target column to use for label generation
        label_types (list): A list of label types for model's target variables
        rolling_windows (list): A list of integers for the future statistic windows

    Returns:
        pd.DataFrame: A clean, feature-rich DataFrame ready for model training and evaluation
    """
    data = _download_stock_data(emiten, start_date, end_date)

    data = generate_all_technical_indicators(data)

    data = _generate_labels_based_on_label_type(
        data, target_column, rolling_windows, label_types
    )

    return data


def prepare_data_for_forecasting(
    emiten: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Orchestrates the full data preparation pipeline for making forecasts using the developed machine learning model

    This function reads pre-calculated technical indicators from CSV files instead of regenerating them.

    Args:
        emiten (str): The stock emiten symbol.
        start_date (str): The start date for the data ('YYYY-MM-DD') - not used, kept for compatibility
        end_date (str): The end date for the data ('YYYY-MM-DD') - not used, kept for compatibility

    Returns:
        pd.DataFrame: A feature-rich DataFrame ready for forecasting
    """
    technical_path = f"data/stock/01_technical/{emiten}.csv"
    data = pd.read_csv(technical_path)

    forecasting_data = data.tail(1).copy()
    forecasting_data["Kode"] = emiten

    return forecasting_data
