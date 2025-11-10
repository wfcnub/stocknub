import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from camel_converter import to_camel
from utils.pipeline import get_label_config

from forecastStocks.helper import _ensure_directories_exist, _clear_forecast_files, _get_filtered_emiten_list, _save_forecast
from prepareTechnicalIndicators.helper import get_all_technical_indicators

def process_single_ticker(args_tuple):
    """
    Process forecast for a single emiten, label_type, and window combination.

    Args:
        args_tuple: Tuple containing (emiten, label_type, window, feature_columns)

    Returns:
        Tuple of (emiten, label_type, window, success, message, forecast_data_dict)
    """
    emiten, label_type, window, feature_columns = args_tuple

    try:
        target_col, threshold_col, positive_label, negative_label = get_label_config(
            label_type, window
        )

        camel_label = to_camel(label_type)
        model_path = f"data/stock/03_model/{camel_label}/{emiten}-{window}dd.pkl"

        if not Path(model_path).exists():
            return (
                emiten,
                label_type,
                window,
                False,
                f"Model not found: {model_path}",
                None,
            )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        try:
            technical_path = f"data/stock/01_technical/{emiten}.csv"
            if not Path(technical_path).exists():
                return (
                    emiten,
                    label_type,
                    window,
                    False,
                    f"Technical data not found: {technical_path}",
                    None,
                )

            forecasting_data = pd.read_csv(technical_path)

            if forecasting_data.empty:
                return (
                    emiten,
                    label_type,
                    window,
                    False,
                    "Technical data is empty",
                    None,
                )

            latest_data = forecasting_data.tail(1).copy()
            latest_data["Kode"] = emiten

        except Exception as e:
            return (
                emiten,
                label_type,
                window,
                False,
                f"Failed to read data: {str(e)}",
                None,
            )

        missing_features = [
            col for col in feature_columns if col not in latest_data.columns
        ]
        if missing_features:
            return (
                emiten,
                label_type,
                window,
                False,
                f"Missing features: {missing_features[:5]}...",
                None,
            )

        forecast_column_name = f"Forecast {positive_label} {window}dd"
        positive_label_index = list(model.classes_).index(positive_label)

        forecast_proba = model.predict_proba(
            latest_data[feature_columns].values.reshape(1, -1)
        )[0, positive_label_index]

        latest_date = (
            latest_data["Date"].iloc[0] if "Date" in latest_data.columns else None
        )

        forecast_data = {
            "Kode": emiten,
            "Date": latest_date,
            "forecast_column": forecast_column_name,
            "forecast_value": forecast_proba,
        }

        return (
            emiten,
            label_type,
            window,
            True,
            f"Forecast: {forecast_proba:.4f}",
            forecast_data,
        )

    except Exception as e:
        return (emiten, label_type, window, False, f"Error: {str(e)}", None)

def forecast_stocks(label_types, windows, min_test_gini, workers=None, tickers=""):
    valid_label_types = ["linear_trend", "median_gain", "max_loss"]
    for label_type in label_types:
        if label_type not in valid_label_types:
            print(f"ERROR: Invalid label type: {label_type}")
            print(f"   Valid types: {', '.join(valid_label_types)}")
            return

    _ensure_directories_exist(label_types, windows)

    print("\nClearing old forecast files...")
    _clear_forecast_files(label_types, windows)

    feature_columns = get_all_technical_indicators()
    print(f"Using {len(feature_columns)} technical indicators as features")

    if tickers:
        emiten_list = [ticker.strip() for ticker in tickers.split(",")]
        print(f"\nForecasting specified tickers: {', '.join(emiten_list)}")
    else:
        print("\nFinding emiten with models meeting criteria...")
        if min_test_gini is not None:
            print(f"   Min Test Gini: {min_test_gini}")
        else:
            print("   Min Test Gini: None (using all available models)")

        emiten_list = _get_filtered_emiten_list(label_types, windows, min_test_gini)

        if not emiten_list:
            print("ERROR: No emiten found meeting the criteria")
            return

        print(f"Found {len(emiten_list)} emiten meeting criteria")

    forecast_tasks = []
    for emiten in emiten_list:
        for label_type in label_types:
            for window in windows:
                forecast_tasks.append((emiten, label_type, window, feature_columns))

    total_tasks = len(forecast_tasks)
    print(
        f"\nStarting forecasts for {len(emiten_list)} emiten × {len(label_types)} label types × {len(windows)} windows = {total_tasks} tasks"
    )
    print(f"Using {workers} parallel workers\n")

    successful = 0
    failed = 0

    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_ticker, forecast_tasks),
                total=total_tasks,
                desc="Generating forecasts",
            )
        )

    return (results, successful, failed, total_tasks)