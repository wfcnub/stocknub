import numpy as np
import pandas as pd
from stock_indicators import indicators

from prepareTechnicalIndicators.helper import identify_historical_trends, safe_divide

def calculate_atr_trailing_stop(data, prepared_data):
    result = indicators.get_atr_stop(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'ATR Stop': [float(val.atr_stop) if val.atr_stop is not None else np.nan for val in result],
        'ATR Buy Stop': [float(val.buy_stop) if val.buy_stop is not None else np.nan for val in result],
        'ATR Sell Stop': [float(val.sell_stop) if val.sell_stop is not None else np.nan for val in result],
    })

    close = pd.Series(data['Close'].values, index=result_df.index)
    result_df['ATR Bullish'] = result_df['ATR Sell Stop'].notna().astype(int)
    result_df['ATR Bearish'] = result_df['ATR Buy Stop'].notna().astype(int)
    result_df['ATR Stop Distance Percent'] = safe_divide(
        close - result_df['ATR Stop'],
        close,
    )

    result_df.drop(columns=['ATR Stop', 'ATR Buy Stop', 'ATR Sell Stop'], inplace=True)

    return result_df.set_index('Date')

def calculate_aroon(prepared_data):
    result = indicators.get_aroon(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Aroon Up': [val.aroon_up for val in result],
        'Aroon Down': [val.aroon_down for val in result],
        'Aroon Oscillator': [val.oscillator for val in result],
    })

    aroon_position = result_df['Aroon Up'].values >= result_df['Aroon Down'].values
    result_df['Aroon Change Position'] = [np.nan] + (aroon_position[:-1] != aroon_position[1:]).astype(int).tolist()
    result_df['Aroon Up Trend'] = (result_df['Aroon Up'].values >= 70).astype(int)
    result_df['Aroon Down Trend'] = (result_df['Aroon Down'].values >= 70).astype(int)

    return result_df.set_index('Date')

def calculate_average_directional_index(prepared_data):
    result = indicators.get_adx(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Plus Directional Index': [val.pdi for val in result],
        'Minus Directional Index': [val.mdi for val in result],
        'ADX Value': [val.adx for val in result],
        'ADXR Value': [val.adxr for val in result],
    })

    result_df['Directional Index Spread'] = (
        result_df['Plus Directional Index'] - result_df['Minus Directional Index']
    )
    result_df['Positive Direction'] = (
        result_df['Plus Directional Index'] > result_df['Minus Directional Index']
    ).astype(int)
    result_df['ADX Strong Trend'] = (result_df['ADX Value'] >= 25).astype(int)
    result_df['ADX Change 5D'] = result_df['ADX Value'].diff(5)

    return result_df.set_index('Date')

def calculate_elder_ray_index(data, prepared_data):
    result = indicators.get_elder_ray(prepared_data)

    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Bull Power': [val.bull_power for val in result],
        'Bear Power': [val.bear_power for val in result]
    })

    result_df['Bull Power Up Trend'] = identify_historical_trends(result_df, 'Bull Power', 10, make_bool_up=True)
    result_df['Bull Power 80% Positives'] = np.array([np.nan if i < 10 else np.sum(result_df['Bull Power'].values[i-10:i] > 0) >= 8 for i in range(len(result_df))])

    result_df['Bear Power Trend'] = identify_historical_trends(result_df, 'Bear Power', 10, make_bool_down=True)
    result_df['Bear Power 80% Negatives'] = np.array([np.nan if i < 10 else np.sum(result_df['Bear Power'].values[i-10:i] < 0) >= 8 for i in range(len(result_df))])

    close = pd.Series(data['Close'].values, index=result_df.index)
    result_df['Bull Power Percent'] = safe_divide(result_df['Bull Power'], close)
    result_df['Bear Power Percent'] = safe_divide(result_df['Bear Power'], close)
    result_df.drop(columns=['Bull Power', 'Bear Power'], inplace=True)

    return result_df.set_index('Date')

def calculate_moving_average_convergence_divergence(prepared_data):
    result = indicators.get_macd(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'MACD Value': [val.macd for val in result],
        'MACD Signal': [val.signal for val in result],
        'MACD Histogram': [val.histogram for val in result],
        'MACD Slow EMA': [val.slow_ema for val in result],
    })

    macd_threshold = result_df['MACD Histogram'].abs().rolling(window=50, min_periods=10).median()
    macd_threshold = macd_threshold.fillna(result_df['MACD Histogram'].abs().expanding(min_periods=1).median())

    result_df['MACD Near 0'] = (result_df['MACD Histogram'].abs() < macd_threshold).astype(int)
    result_df['MACD Non-Near 0 Negative'] = ((result_df['MACD Histogram'] <= -macd_threshold)).astype(int)
    result_df['MACD Non-Near 0 Positive'] = ((result_df['MACD Histogram'] >= macd_threshold)).astype(int)
    result_df['MACD Percent'] = safe_divide(result_df['MACD Value'], result_df['MACD Slow EMA'])
    result_df['MACD Signal Percent'] = safe_divide(result_df['MACD Signal'], result_df['MACD Slow EMA'])
    result_df['MACD Histogram Percent'] = safe_divide(result_df['MACD Histogram'], result_df['MACD Slow EMA'])
    result_df['MACD Histogram Change 5D'] = result_df['MACD Histogram Percent'].diff(5)

    result_df.drop(
        columns=['MACD Value', 'MACD Signal', 'MACD Histogram', 'MACD Slow EMA'],
        inplace=True,
    )

    return result_df.set_index('Date')


def calculate_supertrend(prepared_data):
    """Calculate a causal ATR-based trend state and normalized stop distance."""
    result = indicators.get_super_trend(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'SuperTrend': [
            float(val.super_trend) if val.super_trend is not None else np.nan
            for val in result
        ],
        'SuperTrend Upper Band': [
            float(val.upper_band) if val.upper_band is not None else np.nan
            for val in result
        ],
        'SuperTrend Lower Band': [
            float(val.lower_band) if val.lower_band is not None else np.nan
            for val in result
        ],
    })
    result_df['SuperTrend Bullish'] = result_df['SuperTrend Lower Band'].notna().astype(int)
    result_df['SuperTrend Bearish'] = result_df['SuperTrend Upper Band'].notna().astype(int)
    result_df['SuperTrend Change'] = (
        result_df['SuperTrend Bullish'].diff().abs().fillna(0).astype(int)
    )
    result_df.drop(
        columns=['SuperTrend', 'SuperTrend Upper Band', 'SuperTrend Lower Band'],
        inplace=True,
    )
    return result_df.set_index('Date')


def calculate_vortex(prepared_data):
    """Calculate continuous Vortex direction and spread features."""
    result = indicators.get_vortex(prepared_data, lookback_periods=14)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Vortex Positive': [val.pvi for val in result],
        'Vortex Negative': [val.nvi for val in result],
    })
    result_df['Vortex Spread'] = (
        result_df['Vortex Positive'] - result_df['Vortex Negative']
    )
    result_df['Vortex Positive Direction'] = (
        result_df['Vortex Spread'] > 0
    ).astype(int)
    return result_df.set_index('Date')
