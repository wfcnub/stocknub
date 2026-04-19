from pathlib import Path

from generateScore.helper import (
    _prepare_data,
    _train_model,
    _infer_and_export,
    _generate_score_data_on_test_data,
    _generate_max_daily_performance_metric,
    _generate_trading_simulation_df
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

def process_generate_trading_simulation(rolling_window: str):
    """
    Generate the trading simulation data
    """
    score_df = _generate_score_data_on_test_data(rolling_window)
    max_daily_profit_df = _generate_max_daily_performance_metric(rolling_window, 'Profit')
    max_daily_loss_df = _generate_max_daily_performance_metric(rolling_window, 'Loss')

    trading_simulation_df = _generate_trading_simulation_df(score_df, max_daily_profit_df, max_daily_loss_df, rolling_window)
    trading_simulation_df.to_csv(Path(f'data/stock/score/trading_simulation_{rolling_window}.csv'), index=False)

    return