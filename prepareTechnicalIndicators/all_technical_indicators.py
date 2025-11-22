import numpy as np
import pandas as pd
from stock_indicators import Quote

from prepareTechnicalIndicators.price_trends import calculate_atr_trailing_stop, calculate_aroon, calculate_average_directional_index, calculate_elder_ray_index, calculate_moving_average_convergence_divergence
from prepareTechnicalIndicators.price_channels import calculate_keltner, calculate_donchian, calculate_bollinger_bands
from prepareTechnicalIndicators.oscillators import calculate_relative_strength_index, calculate_stochastic_oscillator
from prepareTechnicalIndicators.volume_based import calculate_on_balance_volume, calculate_money_flow_index, calculate_chaikin_money_flow, calculate_accumulation_distribution_line
from prepareTechnicalIndicators.price_transformations import calculate_ehler_fisher_transform, calculate_zig_zag


def _prepare_data_for_generating_stock_indicators(data: pd.DataFrame) -> list:
    """
    Converts pandas dataframe into a type that is suitable for generating technical indicators using stock_indicators.

    Args:
        data (pd.DataFrame): Stock data containing the column Date, Open, High, Low, Close, and Volume

    Returns:
        list: Converted pandas dataframe ready to be used for generating the technical indicators
    """
    data['Date'] = pd.to_datetime(data['Date'])
    prepared_data = [
        Quote(d,o,h,l,c,v) 
        for d,o,h,l,c,v 
        in zip(data['Date'], data['Open'], data['High'], data['Low'], data['Close'], data['Volume'])
    ]
    
    return prepared_data

def generate_all_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all the technical indicators, each having different unique characteristics of the stock to be measures

    Each of the stocks characteristics to be measured are as follows:
    1. Price Trends
    2. Price Channels
    3. Oscillators
    4. Volume Based
    5. Price Transformations

    Args:
        data (pd.DataFrame): Stock data containing the column Date, Open, High, Low, Close, and Volume

    Returns:
        list: Converted pandas dataframe ready to be used for generating the technical indicators
    """
    prepared_data = _prepare_data_for_generating_stock_indicators(data) 

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    original_columns = set(data.columns)

    price_trends = [
        calculate_atr_trailing_stop(data, prepared_data),
        calculate_aroon(prepared_data),
        calculate_average_directional_index(prepared_data), 
        calculate_elder_ray_index(prepared_data), 
        calculate_moving_average_convergence_divergence(prepared_data)
    ]

    price_channels = [
        calculate_keltner(data, prepared_data), 
        calculate_donchian(data, prepared_data),
        calculate_bollinger_bands(data, prepared_data)
    ]

    oscillators = [
        calculate_relative_strength_index(prepared_data), 
        calculate_stochastic_oscillator(prepared_data)
    ]

    volume_based = [
        calculate_on_balance_volume(prepared_data), 
        calculate_money_flow_index(prepared_data), 
        calculate_chaikin_money_flow(prepared_data),
        calculate_accumulation_distribution_line(prepared_data)
    ]

    price_transformations = [
        calculate_ehler_fisher_transform(prepared_data), 
        calculate_zig_zag(prepared_data)
    ]

    all_stock_indicators = price_trends + price_channels + oscillators + volume_based + price_transformations
    all_stock_indicators_data = data.join(all_stock_indicators)
    all_stock_indicators_data.dropna(inplace=True)

    updated_columns = set(all_stock_indicators_data.columns)
    feature_columns = sorted(list(updated_columns - original_columns))
    for col in feature_columns:
        all_stock_indicators_data[col] = all_stock_indicators_data[col].astype(int)

    output_path = 'data/technical_indicator_features.txt'
    with open(output_path, "w") as file:
        for fea_col in feature_columns:
            file.write(fea_col + "\n")

    return all_stock_indicators_data