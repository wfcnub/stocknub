import pickle
import pandas as pd
from utils.pipeline import get_label_config
from trainModels.features import add_cross_sectional_features, attach_ticker_metadata

def process_single_ticker(args_tuple):
    """
    Forecast an ticker based on the label_type and window combination using the developed model.

    Args:
        args_tuple: Tuple containing (ticker, label_type, window, feature_columns)

    Returns:
        Tuple of (ticker, label_type, window, success, message, forecast_data_dict)
    """
    model_version, model_identifier, ticker, label_type, window, feature_columns = args_tuple

    try:
        target_col, threshold_col, positive_label, negative_label = get_label_config(
            label_type, window
        )

        from utils import paths
        model_path = paths.get_model_path(model_version, label_type, model_identifier, window)

        if not model_path.exists():
            return (
                ticker,
                label_type,
                window,
                False,
                f"Model not found: {model_path}",
                None,
            )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        try:
            if model_version in [1, 2, 3]:
                csv_file_path = paths.get_label_path(ticker)
            elif model_version == 4:
                csv_file_path = paths.get_combined_forecasts_path(window, ticker)
            
            if not csv_file_path.exists():
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

        prediction_data = csv_data
        if model_version in [1, 2, 3]:
            prediction_data = attach_ticker_metadata(csv_data, ticker)
            prediction_data, _ = add_cross_sectional_features(prediction_data)
        expected_features = list(getattr(model, "feature_names_", feature_columns))
        missing_features = [
            col for col in expected_features if col not in prediction_data.columns
        ]
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

        forecast_proba = model.predict_proba(prediction_data[expected_features]) \
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
