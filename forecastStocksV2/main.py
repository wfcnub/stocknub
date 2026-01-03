import pickle
import pandas as pd
from pathlib import Path
from camel_converter import to_camel
from utils.pipeline import get_label_config

def process_single_ticker(args_tuple):
    """
    Forecast an emiten based on the label_type and window combination using the developed model.

    Args:
        args_tuple: Tuple containing (emiten, label_type, window, feature_columns)

    Returns:
        Tuple of (emiten, label_type, window, success, message, forecast_data_dict)
    """
    technical_folder, industry, emiten, label_type, window, feature_columns = args_tuple

    try:
        target_col, threshold_col, positive_label, negative_label = get_label_config(
            label_type, window
        )

        camel_label = to_camel(label_type)
        model_path = f"data/stock/model_v2/{camel_label}/{industry}-{window}dd.pkl"

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
            technical_path = f"{technical_folder}/{emiten}.csv"
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
            col for col in feature_columns if col not in forecasting_data.columns
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
            forecasting_data[feature_columns].values
        )[:, positive_label_index]

        forecasting_data[forecast_column_name] = forecast_proba

        return (
            emiten,
            label_type,
            window,
            True,
            "Forecast Succeeded",
            forecasting_data,
        )

    except Exception as e:
        return (
            emiten, 
            label_type, 
            window, 
            False, 
            f"Error: {str(e)}", 
            None
        )