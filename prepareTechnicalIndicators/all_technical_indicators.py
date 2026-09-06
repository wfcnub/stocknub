import numpy as np
import pandas as pd
from stock_indicators import Quote

from prepareTechnicalIndicators.momentum import calculate_price_momentum
from prepareTechnicalIndicators.price_trends import (
    calculate_aroon,
    calculate_atr_trailing_stop,
    calculate_average_directional_index,
    calculate_elder_ray_index,
    calculate_moving_average_convergence_divergence,
    calculate_supertrend,
    calculate_vortex,
)
from prepareTechnicalIndicators.price_channels import calculate_keltner, calculate_donchian, calculate_bollinger_bands
from prepareTechnicalIndicators.oscillators import calculate_relative_strength_index, calculate_stochastic_oscillator, calculate_multi_timeframe_rsi
from prepareTechnicalIndicators.volume_based import (
    calculate_accumulation_distribution_line,
    calculate_chaikin_money_flow,
    calculate_money_flow_index,
    calculate_on_balance_volume,
    calculate_percentage_volume_oscillator,
)
from prepareTechnicalIndicators.price_transformations import calculate_ehler_fisher_transform
from prepareTechnicalIndicators.additional_technical_indicators import calculate_additional_technical_indicators
from prepareTechnicalIndicators.market_regime import calculate_market_regime
from prepareTechnicalIndicators.price_characteristics import calculate_price_characteristics


OHLCV_COLUMNS = ('Open', 'High', 'Low', 'Close', 'Volume')
# ADX(14) has the longest documented convergence recommendation currently used:
# 2 * 14 + 250 observations.
MINIMUM_WARMUP_PERIODS = 278


def _normalize_input_frame(data: pd.DataFrame, required_columns, frame_name):
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required columns: {', '.join(missing_columns)}"
        )

    normalized = data.copy()
    normalized['Date'] = pd.to_datetime(normalized['Date'], errors='raise')
    if normalized['Date'].duplicated().any():
        duplicate_dates = normalized.loc[
            normalized['Date'].duplicated(),
            'Date',
        ].dt.strftime('%Y-%m-%d').tolist()
        raise ValueError(
            f"{frame_name} contains duplicate dates: {', '.join(duplicate_dates[:5])}"
        )

    return normalized.sort_values('Date').reset_index(drop=True)

def _prepare_data_for_generating_stock_indicators(data: pd.DataFrame) -> list:
    """
    Converts pandas dataframe into a type that is suitable for generating technical indicators using stock_indicators.

    Args:
        data (pd.DataFrame): Stock data containing the column Date, Open, High, Low, Close, and Volume

    Returns:
        list: Converted pandas dataframe ready to be used for generating the technical indicators
    """
    prepared_data = [
        Quote(d,o,h,l,c,v) 
        for d,o,h,l,c,v 
        in zip(data['Date'], data['Open'], data['High'], data['Low'], data['Close'], data['Volume'])
    ]
    
    return prepared_data

def _generate_all_technical_indicators(data, additional_data, prepared_data, technical_indicator):
    if technical_indicator == 'price_trends':
        technical_indicator_data = [
            calculate_atr_trailing_stop(data, prepared_data),
            calculate_aroon(prepared_data),
            calculate_average_directional_index(prepared_data), 
            calculate_elder_ray_index(data, prepared_data),
            calculate_moving_average_convergence_divergence(prepared_data),
            calculate_supertrend(prepared_data),
            calculate_vortex(prepared_data),
        ]

    elif technical_indicator == 'price_channels':
        technical_indicator_data = [
            calculate_keltner(data, prepared_data), 
            calculate_donchian(data, prepared_data),
            calculate_bollinger_bands(data, prepared_data)
        ]

    elif technical_indicator == 'oscillators':
        technical_indicator_data = [
            calculate_relative_strength_index(prepared_data), 
            calculate_stochastic_oscillator(prepared_data),
            calculate_multi_timeframe_rsi(prepared_data)
        ]

    elif technical_indicator == 'volume_based':
        technical_indicator_data = [
            calculate_on_balance_volume(prepared_data),
            calculate_money_flow_index(prepared_data),
            calculate_chaikin_money_flow(prepared_data),
            calculate_accumulation_distribution_line(prepared_data),
            calculate_percentage_volume_oscillator(prepared_data),
        ]

    elif technical_indicator == 'price_transformations':
        technical_indicator_data = [
            calculate_ehler_fisher_transform(prepared_data),
        ]
    
    elif technical_indicator == 'additional_technical_indicators':
        technical_indicator_data = [
            calculate_additional_technical_indicators(additional_data)
        ]

    elif technical_indicator == 'momentum':
        technical_indicator_data = [
            calculate_price_momentum(data)
        ]

    elif technical_indicator == 'price_characteristics':
        technical_indicator_data = [
            calculate_price_characteristics(data)
        ]

    elif technical_indicator == 'market_regime':
        technical_indicator_data = [
            calculate_market_regime(prepared_data)
        ]

    return technical_indicator_data
    
def generate_all_technical_indicators(data: pd.DataFrame, additional_data: pd.DataFrame) -> pd.DataFrame:
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
    data = _normalize_input_frame(
        data,
        ('Date', *OHLCV_COLUMNS),
        'OHLCV data',
    )
    additional_data = _normalize_input_frame(
        additional_data,
        (
            'Date',
            *OHLCV_COLUMNS,
            'Foreign Buy',
            'Foreign Sell',
            'Non Regular Volume',
            'Non Regular Value',
            'Non Regular Frequency',
        ),
        'Foreign-flow data',
    )

    if len(data) <= MINIMUM_WARMUP_PERIODS:
        raise ValueError(
            f"At least {MINIMUM_WARMUP_PERIODS + 1} OHLCV rows are required; "
            f"received {len(data)}"
        )

    prepared_data = _prepare_data_for_generating_stock_indicators(data)
    data = data.copy()
    data.set_index('Date', inplace=True)

    additional_data = additional_data.copy()
    additional_data.set_index('Date', inplace=True)

    original_columns = set(data.columns)

    all_stock_indicators_data = data.copy()

    selected_technical_indicators = [
        'price_trends',
        'price_channels',
        'oscillators',
        'volume_based',
        'price_transformations',
        'additional_technical_indicators',
        'momentum',
        'price_characteristics',
        'market_regime',
    ]
    
    for technical_indicator in selected_technical_indicators:
        technical_indicator_data = _generate_all_technical_indicators(data, additional_data, prepared_data, technical_indicator)
        for d in technical_indicator_data:
            cleaned_d = d.reset_index().drop_duplicates('Date').set_index('Date')
            overlapping_columns = set(all_stock_indicators_data.columns) & set(
                cleaned_d.columns
            )
            if overlapping_columns:
                raise ValueError(
                    "Duplicate technical feature names: "
                    + ", ".join(sorted(overlapping_columns))
                )
            all_stock_indicators_data = all_stock_indicators_data.join(
                cleaned_d,
                how='left',
            )

    all_stock_indicators_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_stock_indicators_data = all_stock_indicators_data.iloc[
        MINIMUM_WARMUP_PERIODS:
    ].copy()

    feature_columns = sorted(set(all_stock_indicators_data.columns) - original_columns)
    if not feature_columns:
        raise ValueError("Technical indicator generation produced no feature columns")

    return all_stock_indicators_data
