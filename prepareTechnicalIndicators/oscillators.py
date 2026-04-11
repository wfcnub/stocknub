import numpy as np
import pandas as pd
from stock_indicators import indicators

def calculate_relative_strength_index(prepared_data):
    """Calculate RSI with continuous values, binary flags, multi-timeframe, and rate-of-change features."""
    result = indicators.get_rsi(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Relative Strength Index': [val.rsi for val in result]
    })
    
    result_df.dropna(subset=['Relative Strength Index'], inplace=True)

    result_df['RSI Value'] = result_df['Relative Strength Index']

    result_df['RSI Up Trend'] = (result_df['Relative Strength Index'] >= 65).astype(int)
    result_df['RSI Down Trend'] = (result_df['Relative Strength Index'] <= 35).astype(int)

    result_df['RSI Rate of Change 5'] = result_df['Relative Strength Index'] - result_df['Relative Strength Index'].shift(5)
    result_df['RSI Rate of Change 10'] = result_df['Relative Strength Index'] - result_df['Relative Strength Index'].shift(10)

    result_df.drop(columns=['Relative Strength Index'], inplace=True)

    return result_df.set_index('Date')

def calculate_multi_timeframe_rsi(prepared_data):
    """
    Calculate RSI at multiple timeframes (7, 21) to complement the default 14-period RSI.

    This lets the model capture short-term vs long-term momentum divergences.
    """
    rsi_periods = [7, 21]
    result_dfs = []

    for period in rsi_periods:
        result = indicators.get_rsi(prepared_data, lookback_periods=period)
        rsi_df = pd.DataFrame({
            'Date': [val.date for val in result],
            f'RSI {period}': [val.rsi for val in result]
        })
        rsi_df.dropna(subset=[f'RSI {period}'], inplace=True)
        rsi_df = rsi_df.set_index('Date')
        result_dfs.append(rsi_df)

    combined = result_dfs[0].join(result_dfs[1], how='outer')

    return combined


def calculate_stochastic_oscillator(prepared_data):
    """Calculate Stochastic Oscillator with continuous value alongside binary flags."""
    result = indicators.get_stoch(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Stochastic Oscillator': [val.oscillator for val in result]
    })

    result_df.dropna(subset=['Stochastic Oscillator'], inplace=True)

    result_df['Stochastic K Value'] = result_df['Stochastic Oscillator']

    result_df['Stochastic Oscillator Overbought'] = (result_df['Stochastic Oscillator'] >= 80).astype(float)
    result_df['Stochastic Oscillator Oversold'] = (result_df['Stochastic Oscillator'] <= 20).astype(float)

    result_df.drop(columns=['Stochastic Oscillator'], inplace=True)

    return result_df.set_index('Date')