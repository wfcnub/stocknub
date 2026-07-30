import os
from pathlib import Path
from camel_converter import to_camel
from typing import Union

env = os.getenv("APP_ENV", "prod")

if env not in ["dev", "prod"]:
    env = "prod"

BASE_DIR = Path("data") / env
STOCK_DIR = BASE_DIR / "stock"

def _format_window(window: Union[int, str]) -> str:
    window_str = str(window)
    if not window_str.endswith("dd"):
        return f"{window_str}dd"
    return window_str

# Common base directories
def get_label_dir() -> Path:
    return STOCK_DIR / "label"

def get_label_path(ticker: str) -> Path:
    return get_label_dir() / f"{ticker}.csv"

def get_ohlcv_dir() -> Path:
    return STOCK_DIR / "OHLCV"

def get_ohlcv_path(ticker: str) -> Path:
    return get_ohlcv_dir() / f"{ticker}.csv"

def get_raw_foreign_flow_non_regular_dir() -> Path:
    return STOCK_DIR / "raw_foreign_flow_non_regular"

def get_foreign_flow_non_regular_dir() -> Path:
    return STOCK_DIR / "foreign_flow_non_regular"

def get_foreign_flow_non_regular_path(ticker: str) -> Path:
    return get_foreign_flow_non_regular_dir() / f"{ticker}.csv"

def get_technical_dir() -> Path:
    return STOCK_DIR / "technical"

def get_technical_path(ticker: str) -> Path:
    return get_technical_dir() / f"{ticker}.csv"

def get_technical_indicator_features_path() -> Path:
    return BASE_DIR / "technical_indicator_features.txt"

def get_ticker_list_path() -> Path:
    return BASE_DIR / "ticker_list.txt"

def get_ticker_and_industry_list_path() -> Path:
    return BASE_DIR / "ticker_and_industry_list.csv"

def get_selected_ticker_and_industry_list_path() -> Path:
    return BASE_DIR / "selected_ticker_and_industry_list.csv"

def get_split_dates_path(window: Union[int, str]) -> Path:
    window_str = str(window).replace("dd", "")
    return BASE_DIR / f"split_dates_{window_str}.json"

def get_pre_market_outlook_path() -> Path:
    return BASE_DIR / "pre_market_outlook.json"

def get_combined_forecasts_columns_information_path(window: Union[int, str]) -> Path:
    return BASE_DIR / f"combined_forecasts_columns_information_{_format_window(window)}.yaml"

# Models
def get_model_dir(version: int, label_type: str) -> Path:
    camel_label = to_camel(label_type)
    return STOCK_DIR / f"model_v{version}" / camel_label

def get_model_path(version: int, label_type: str, identifier: str, window: Union[int, str]) -> Path:
    return get_model_dir(version, label_type) / f"{identifier}-{_format_window(window)}.pkl"

def get_model_performance_base_dir(version: int) -> Path:
    return STOCK_DIR / f"model_v{version}" / "performance"

def get_model_performance_dir(version: int, label_type: str) -> Path:
    camel_label = to_camel(label_type)
    return get_model_performance_base_dir(version) / camel_label

def get_model_performance_path(version: int, label_type: str, window: Union[int, str]) -> Path:
    return get_model_performance_dir(version, label_type) / f"{_format_window(window)}.csv"

# Score
def get_score_dir() -> Path:
    return STOCK_DIR / "score"

def get_score_window_dir(window: Union[int, str]) -> Path:
    return get_score_dir() / _format_window(window)

def get_score_path(window: Union[int, str]) -> Path:
    return get_score_dir() / f"{str(window).replace('dd', '')}.pkl"

def get_trading_simulation_path(window: Union[int, str]) -> Path:
    return get_score_dir() / f"trading_simulation_{_format_window(window)}.csv"

# Forecasts
def get_forecast_base_dir(version: int, label_type: str) -> Path:
    camel_label = to_camel(label_type)
    return STOCK_DIR / "forecast" / f"model_v{version}" / camel_label

def get_forecast_dir(version: int, label_type: str, window: Union[int, str]) -> Path:
    return get_forecast_base_dir(version, label_type) / _format_window(window)

def get_forecast_path(version: int, label_type: str, window: Union[int, str], ticker: str) -> Path:
    return get_forecast_dir(version, label_type, window) / f"{ticker}.csv"

def get_combined_forecasts_dir() -> Path:
    return STOCK_DIR / "combined_forecasts"

def get_combined_forecasts_base_dir() -> Path:
    return STOCK_DIR / "combined_forecasts"

def get_combined_forecasts_window_dir(window: Union[int, str]) -> Path:
    return STOCK_DIR / f"combined_forecasts_{_format_window(window)}"

def get_combined_forecasts_path(window: Union[int, str], ticker: str) -> Path:
    return get_combined_forecasts_window_dir(window) / f"{ticker}.csv"

