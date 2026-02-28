import pickle
import pandas as pd
from pathlib import Path
from camel_converter import to_camel
from utils.pipeline import get_label_config

def process_single_ticker(args_tuple):
    """
    Forecast an ticker based on the label_type and window combination using the developed model.

    Args:
        args_tuple: Tuple containing (ticker, label_type, window, feature_columns)

    Returns:
        Tuple of (ticker, label_type, window, success, message, forecast_data_dict)
    """
    model_version, csv_folder_path, model_identifier, ticker, label_type, window, feature_columns = args_tuple

    try:
        target_col, threshold_col, positive_label, negative_label = get_label_config(
            label_type, window
        )

        camel_label = to_camel(label_type)
        model_path = Path(f"data/stock/model_v{model_version}/{camel_label}/{model_identifier}-{window}dd.pkl")

        if not Path(model_path).exists():
            return (
                identifier,
                label_type,
                window,
                False,
                f"Model not found: {model_path}",
                None,
            )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        try:
            csv_file_path = Path(f"{csv_folder_path}/{ticker}.csv")
            if not Path(csv_file_path).exists():
                return (
                    ticker,
                    label_type,
                    window,
                    False,
                    f"CSV file data not found: {csv_file_path}",
                    None,
                )

            csv_data = pd.read_csv(csv_file_path)
            if csv_data.empty:
                return (
                    ticker,
                    label_type,
                    window,
                    False,
                    "CSV file data is empty",
                    None,
                )

        except Exception as e:
            return (
                ticker,
                label_type,
                window,
                False,
                f"Failed to read data: {str(e)}",
                None,
            )

        missing_features = [col for col in feature_columns if col not in csv_data.columns]
        if missing_features:
            return (
                ticker,
                label_type,
                window,
                False,
                f"Missing features: {missing_features[:5]}...",
                None,
            )

        forecast_column_name = f"Forecast {positive_label} {window}dd"
        positive_label_index = list(model.classes_).index(positive_label)

        forecast_proba = model.predict_proba(csv_data[feature_columns].values) \
                                [:, positive_label_index]

        csv_data[forecast_column_name] = forecast_proba

        return (
            ticker,
            label_type,
            window,
            True,
            "Forecast Succeeded",
            csv_data,
        )

    except Exception as e:
        return (
            ticker, 
            label_type, 
            window, 
            False, 
            f"Error: {str(e)}", 
            None
        )