import numpy as np
import pandas as pd

from prepareTechnicalIndicators.helper import rolling_zscore, safe_divide


RETURN_WINDOWS = (1, 3, 5, 10, 20)
TREND_WINDOWS = (5, 10, 20, 50)
VOLATILITY_WINDOWS = (5, 20, 60)
RANGE_WINDOWS = (20, 60)


def calculate_price_characteristics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate causal, normalized OHLCV features that are comparable across stocks.

    The input is expected to be Date-indexed and sorted in ascending order.  All
    rolling calculations include only the current and preceding observations.
    """
    result = pd.DataFrame(index=data.index)
    open_price = data['Open'].astype(float)
    high = data['High'].astype(float)
    low = data['Low'].astype(float)
    close = data['Close'].astype(float)
    volume = data['Volume'].astype(float)

    log_close = np.log(close.where(close > 0))
    log_return_1d = log_close.diff()

    for window in RETURN_WINDOWS:
        result[f'Log Return {window}D'] = log_close.diff(window)

    previous_close = close.shift(1)
    result['Open Gap Percent'] = safe_divide(open_price - previous_close, previous_close)
    result['Intraday Return Percent'] = safe_divide(close - open_price, open_price)
    result['High Low Range Percent'] = safe_divide(high - low, close)
    result['Close Location Value'] = safe_divide(
        (close - low) - (high - close),
        high - low,
    )

    for window in TREND_WINDOWS:
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        result[f'Close to EMA {window}D'] = safe_divide(close - ema, ema)

    ema5 = close.ewm(span=5, adjust=False, min_periods=5).mean()
    ema10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    result['EMA 5D to 20D Spread'] = safe_divide(ema5 - ema20, ema20)
    result['EMA 10D to 50D Spread'] = safe_divide(ema10 - ema50, ema50)

    for window in VOLATILITY_WINDOWS:
        result[f'Realized Volatility {window}D'] = (
            log_return_1d.rolling(window=window, min_periods=window).std(ddof=0)
            * np.sqrt(252)
        )

    result['Volatility 5D to 20D Ratio'] = safe_divide(
        result['Realized Volatility 5D'],
        result['Realized Volatility 20D'],
    )
    result['Volatility 20D to 60D Ratio'] = safe_divide(
        result['Realized Volatility 20D'],
        result['Realized Volatility 60D'],
    )

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    result['ATR Percent 14D'] = safe_divide(atr14, close)

    for window in RANGE_WINDOWS:
        rolling_high = high.rolling(window=window, min_periods=window).max()
        rolling_low = low.rolling(window=window, min_periods=window).min()
        rolling_mean = close.rolling(window=window, min_periods=window).mean()
        rolling_std = close.rolling(window=window, min_periods=window).std(ddof=0)

        result[f'Distance From {window}D High'] = safe_divide(
            close - rolling_high,
            rolling_high,
        )
        result[f'Distance From {window}D Low'] = safe_divide(
            close - rolling_low,
            rolling_low,
        )
        result[f'Price ZScore {window}D'] = safe_divide(
            close - rolling_mean,
            rolling_std,
        )

    result['Volume Change 1D'] = volume.pct_change(1)
    result['Volume Change 5D'] = volume.pct_change(5)
    result['Volume ZScore 20D'] = rolling_zscore(volume, 20)
    result['Volume ZScore 60D'] = rolling_zscore(volume, 60)

    dollar_volume = close * volume
    result['Dollar Volume Ratio 20D'] = safe_divide(
        dollar_volume,
        dollar_volume.rolling(window=20, min_periods=20).mean(),
    )
    up_volume = volume.where(close > previous_close, 0.0)
    result['Up Volume Share 20D'] = safe_divide(
        up_volume.rolling(window=20, min_periods=20).sum(),
        volume.rolling(window=20, min_periods=20).sum(),
    )

    result['Return Volume Correlation 20D'] = (
        log_return_1d.rolling(window=20, min_periods=20).corr(volume.pct_change())
    )
    result['Return Volume Correlation 60D'] = (
        log_return_1d.rolling(window=60, min_periods=60).corr(volume.pct_change())
    )
    result['Price Volume Divergence 20D'] = (
        result['Log Return 5D'] * result['Volume ZScore 20D']
    )

    return result.replace([np.inf, -np.inf], np.nan)
