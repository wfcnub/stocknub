import numpy as np
import pandas as pd

def calculate_price_momentum(data):
    """
    Calculate price momentum and volume ratio features from raw OHLCV data.

    These features capture velocity (how fast price is moving) rather than just state,
    giving the model richer signal about trend strength.
    """
    result_df = pd.DataFrame({'Date': data.index})
    result_df = result_df.set_index('Date')

    close = data['Close']
    result_df['Price Momentum 5D'] = close.pct_change(periods=5)
    result_df['Price Momentum 10D'] = close.pct_change(periods=10)
    result_df['Price Momentum 20D'] = close.pct_change(periods=20)

    vol_ma20 = data['Volume'].rolling(window=20).mean()
    result_df['Volume Ratio 20D'] = data['Volume'] / vol_ma20.replace(0, np.nan)
    result_df['Volume Ratio 20D'] = result_df['Volume Ratio 20D'].replace([np.inf, -np.inf], np.nan)

    return result_df