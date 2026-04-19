from generateScore.helper import (
    _prepare_data,
    _train_model,
    _infer_and_export
)

def process_generate_score(window: str):
    """
    Train a Logistic Regression tracking model to calculate composite scores 
    combining forecast probabilities and model performance metrics (Gini), then 
    perform inference to score the tested and active forecasting periods.

    Args:
        window (str): The evaluated rolling window configuration (e.g., '5dd', '10dd').
    """
    joined_train_data, joined_test_data, joined_forecast_data, feature_col, target_col, score_col = _prepare_data(window)
    
    model = _train_model(joined_train_data, feature_col, target_col, window)
    
    _infer_and_export(model, joined_test_data, joined_forecast_data, feature_col, target_col, score_col, window)

    return
