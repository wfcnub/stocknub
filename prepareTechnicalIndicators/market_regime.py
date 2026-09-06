import pandas as pd
from stock_indicators import indicators


def calculate_market_regime(prepared_data):
    """Calculate causal regime features for trendiness and persistence."""
    choppiness = indicators.get_chop(prepared_data, lookback_periods=14)
    hurst = indicators.get_hurst(prepared_data, lookback_periods=100)

    choppiness_df = pd.DataFrame({
        'Date': [value.date for value in choppiness],
        'Choppiness Index 14D': [value.chop for value in choppiness],
    }).set_index('Date')
    hurst_df = pd.DataFrame({
        'Date': [value.date for value in hurst],
        'Hurst Exponent 100D': [value.hurst_exponent for value in hurst],
    }).set_index('Date')

    result = choppiness_df.join(hurst_df, how='outer')
    result['Regime Trending'] = (result['Choppiness Index 14D'] < 38.2).astype(int)
    result['Regime Choppy'] = (result['Choppiness Index 14D'] > 61.8).astype(int)
    result['Regime Persistent'] = (result['Hurst Exponent 100D'] > 0.55).astype(int)
    result['Regime Mean Reverting'] = (
        result['Hurst Exponent 100D'] < 0.45
    ).astype(int)
    return result
