import numpy as np
import pandas as pd

from prepareTechnicalIndicators.helper import rolling_zscore, safe_divide


FLOW_WINDOWS = (5, 10, 20, 60)
NON_REGULAR_WINDOWS = (5, 20, 60)
FLOW_PUBLICATION_LAG = 1


def _require_columns(data: pd.DataFrame, required_columns):
    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError(
            "Foreign-flow data is missing required columns: " + ", ".join(missing)
        )


def _calculate_foreign_flow_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate foreign participation and net-flow features.

    IDX daily ``Volume`` is treated as matched market volume on one side of each
    transaction.  Therefore domestic buy/sell volume is ``Volume - Foreign
    Buy/Sell`` and domestic net flow is the exact inverse of foreign net flow.
    No feature is described as ownership because holdings/free-float data is not
    available in this dataset.
    """
    _require_columns(
        data,
        ('Open', 'High', 'Low', 'Close', 'Volume', 'Foreign Buy', 'Foreign Sell'),
    )

    close = data['Close'].astype(float)
    total_volume = data['Volume'].astype(float)
    foreign_buy = data['Foreign Buy'].astype(float)
    foreign_sell = data['Foreign Sell'].astype(float)
    net_foreign = foreign_buy - foreign_sell
    foreign_turnover = foreign_buy + foreign_sell
    typical_price = data[['High', 'Low', 'Close']].astype(float).mean(axis=1)
    log_return = np.log(close.where(close > 0)).diff()

    result = pd.DataFrame(index=data.index)
    result['Foreign Net Intensity 1D'] = safe_divide(net_foreign, total_volume)
    result['Foreign Buy Share 1D'] = safe_divide(foreign_buy, total_volume)
    result['Foreign Sell Share 1D'] = safe_divide(foreign_sell, total_volume)
    result['Foreign Participation 1D'] = safe_divide(
        foreign_turnover,
        2 * total_volume,
    )
    result['Foreign Net Flow Signed Log'] = (
        np.sign(net_foreign) * np.log1p(net_foreign.abs())
    )

    for window in FLOW_WINDOWS:
        rolling_volume = total_volume.rolling(window, min_periods=window).sum()
        rolling_buy = foreign_buy.rolling(window, min_periods=window).sum()
        rolling_sell = foreign_sell.rolling(window, min_periods=window).sum()
        rolling_net = net_foreign.rolling(window, min_periods=window).sum()
        rolling_turnover = foreign_turnover.rolling(
            window,
            min_periods=window,
        ).sum()

        result[f'Foreign Net Intensity {window}D'] = safe_divide(
            rolling_net,
            rolling_volume,
        )
        result[f'Foreign Buy Share {window}D'] = safe_divide(
            rolling_buy,
            rolling_volume,
        )
        result[f'Foreign Sell Share {window}D'] = safe_divide(
            rolling_sell,
            rolling_volume,
        )
        result[f'Foreign Participation {window}D'] = safe_divide(
            rolling_turnover,
            2 * rolling_volume,
        )
        result[f'Foreign Net Flow ZScore {window}D'] = rolling_zscore(
            net_foreign,
            window,
        )
        result[f'Foreign Accumulation Day Share {window}D'] = (
            (net_foreign > 0)
            .astype(float)
            .rolling(window, min_periods=window)
            .mean()
        )

        rolling_net_value = (typical_price * net_foreign).rolling(
            window,
            min_periods=window,
        ).sum()
        rolling_market_value = (typical_price * total_volume).rolling(
            window,
            min_periods=window,
        ).sum()
        result[f'Foreign Net Value Intensity {window}D'] = safe_divide(
            rolling_net_value,
            rolling_market_value,
        )

    net_intensity = result['Foreign Net Intensity 1D']
    net_ema5 = net_intensity.ewm(span=5, adjust=False, min_periods=5).mean()
    net_ema20 = net_intensity.ewm(span=20, adjust=False, min_periods=20).mean()
    result['Foreign Net Intensity EMA 5D Minus 20D'] = net_ema5 - net_ema20
    result['Foreign Net Intensity Acceleration 5D'] = (
        net_intensity.diff(5) - net_intensity.diff(5).shift(5)
    )

    for window in (20, 60):
        result[f'Foreign Flow Return Correlation {window}D'] = (
            net_intensity.rolling(window, min_periods=window).corr(log_return)
        )
        result[f'Foreign Flow Price Divergence {window}D'] = (
            rolling_zscore(net_intensity, window)
            - rolling_zscore(log_return, window)
        )

    return result


def _calculate_non_regular_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate block/non-regular activity features."""
    required = (
        'Non Regular Volume',
        'Non Regular Value',
        'Non Regular Frequency',
    )
    _require_columns(data, required)

    result = pd.DataFrame(index=data.index)
    total_volume = data['Volume'].astype(float)
    typical_price = data[['High', 'Low', 'Close']].astype(float).mean(axis=1)
    market_value_proxy = typical_price * total_volume
    non_regular_volume = data['Non Regular Volume'].astype(float)
    non_regular_value = data['Non Regular Value'].astype(float)
    non_regular_frequency = data['Non Regular Frequency'].astype(float)

    result['Non Regular Volume Share 1D'] = safe_divide(
        non_regular_volume,
        total_volume,
    )
    result['Non Regular Value Share 1D'] = safe_divide(
        non_regular_value,
        market_value_proxy,
    )
    result['Non Regular Average Trade Value 1D'] = safe_divide(
        non_regular_value,
        non_regular_frequency,
    )
    result['Non Regular Average Trade Volume 1D'] = safe_divide(
        non_regular_volume,
        non_regular_frequency,
    )

    for window in NON_REGULAR_WINDOWS:
        rolling_market_volume = total_volume.rolling(
            window,
            min_periods=window,
        ).sum()
        rolling_market_value = market_value_proxy.rolling(
            window,
            min_periods=window,
        ).sum()
        rolling_non_regular_volume = non_regular_volume.rolling(
            window,
            min_periods=window,
        ).sum()
        rolling_non_regular_value = non_regular_value.rolling(
            window,
            min_periods=window,
        ).sum()

        result[f'Non Regular Volume Share {window}D'] = safe_divide(
            rolling_non_regular_volume,
            rolling_market_volume,
        )
        result[f'Non Regular Value Share {window}D'] = safe_divide(
            rolling_non_regular_value,
            rolling_market_value,
        )
        result[f'Non Regular Volume ZScore {window}D'] = rolling_zscore(
            non_regular_volume,
            window,
        )
        result[f'Non Regular Value ZScore {window}D'] = rolling_zscore(
            non_regular_value,
            window,
        )
        result[f'Non Regular Frequency ZScore {window}D'] = rolling_zscore(
            non_regular_frequency,
            window,
        )

    result['Non Regular Block Activity'] = (
        result['Non Regular Volume ZScore 20D'] > 2
    ).astype(int)
    return result


def calculate_additional_technical_indicators(
    data: pd.DataFrame,
    publication_lag: int = FLOW_PUBLICATION_LAG,
) -> pd.DataFrame:
    """
    Generate stable foreign-flow and non-regular-market features.

    The complete result is shifted by the configured publication lag so a row
    never uses transaction data that was unavailable at prediction time.
    """
    foreign_flow = _calculate_foreign_flow_features(data)
    non_regular = _calculate_non_regular_features(data)
    result = foreign_flow.join(non_regular, how='left')
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.shift(publication_lag)
