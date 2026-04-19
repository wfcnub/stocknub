from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from preMarketOutlook.helper import (
    _create_session,
    _fetch_indicator_history,
    _fetch_intraday_price,
    _get_latest_value,
    _compute_percentile_rank,
    _classify_vix,
    _classify_usdidr_change,
    _classify_index_change,
    _derive_overall_outlook,
    _save_outlook_to_json,
    VIX_TICKER,
    USDIDR_TICKER,
    SP500_TICKER,
    NIKKEI_TICKER,
    LOOKBACK_DAYS,
)

def _fetch_ihsg_data(period="1y"):
    ticker_symbol = "^JKSE"
    ihsg_data = yf.download(ticker_symbol, period=period, interval="1d")
    return ihsg_data

def _calculate_micro_outlook(rolling_window: int) -> dict:
    ihsg_data = _fetch_ihsg_data(period="1y")
    if isinstance(ihsg_data.columns, pd.MultiIndex) or (len(ihsg_data.columns) > 0 and isinstance(ihsg_data.columns[0], tuple)):
        ihsg_data.columns = [col[0] for col in ihsg_data.columns]

    median_close = ihsg_data['Close'] \
                        [::-1] \
                        .rolling(rolling_window, closed='left') \
                        .quantile(0.4) \
                        [::-1]

    median_gain = (100 * (median_close - ihsg_data['Close'].values) / ihsg_data['Close'].values)

    ihsg_data[f'Median Gain {rolling_window}dd'] = median_gain
    ihsg_data.index = pd.to_datetime(ihsg_data.index).strftime('%Y-%m-%d')

    all_score_paths = list(Path(f'data/stock/score/{rolling_window}dd').rglob('*.csv'))
    if not all_score_paths:
        return {}

    score_test_data = pd.concat((pd.read_csv(file) for file in all_score_paths)).groupby('Date')[f'Score {rolling_window}dd'].mean().to_frame(f'Average Score {rolling_window}dd')

    joined_ihsg_score_data = pd.merge(ihsg_data.dropna(), score_test_data, left_index=True, right_index=True, how='inner')[[f'Median Gain {rolling_window}dd', f'Average Score {rolling_window}dd']]
    
    if joined_ihsg_score_data.empty:
        return {}

    latest_score_data = score_test_data.tail(1)
    latest_avg_score = latest_score_data[f'Average Score {rolling_window}dd'].values[0]
    ihsg_micro_outlook = joined_ihsg_score_data.loc[
        joined_ihsg_score_data[f'Average Score {rolling_window}dd'] >= latest_avg_score, 
        f'Median Gain {rolling_window}dd'
    ].describe()
    
    return ihsg_micro_outlook.to_dict()

def generate_pre_market_outlook() -> dict:
    """
    Main orchestration function for the Pre-Market Macro Analysis feature.

    Fetches the latest VIX, USD/IDR, S&P 500, and Nikkei 225 data from
    Yahoo Finance, analyses them against their historical distributions,
    and produces a consolidated market outlook for the Indonesian equity market.

    Nikkei 225 uses intraday data when available (market opens 2 hours before IDX),
    while S&P 500 uses the previous session's close.

    Returns:
        dict: A nested dictionary containing analysis for each indicator and an
              overall composite outlook.
    """
    session = _create_session()

    vix_history = _fetch_indicator_history(VIX_TICKER, LOOKBACK_DAYS, session)
    usdidr_history = _fetch_indicator_history(USDIDR_TICKER, LOOKBACK_DAYS, session)
    sp500_history = _fetch_indicator_history(SP500_TICKER, LOOKBACK_DAYS, session)
    nikkei_history = _fetch_indicator_history(NIKKEI_TICKER, LOOKBACK_DAYS, session)

    vix_date, vix_value = _get_latest_value(vix_history)
    usdidr_date, usdidr_value = _get_latest_value(usdidr_history)
    sp500_date, sp500_value = _get_latest_value(sp500_history)

    nikkei_timestamp, nikkei_value, nikkei_is_live = _fetch_intraday_price(NIKKEI_TICKER, session)

    vix_percentile = _compute_percentile_rank(vix_history, vix_value)
    usdidr_percentile = _compute_percentile_rank(usdidr_history, usdidr_value)
    sp500_percentile = _compute_percentile_rank(sp500_history, sp500_value)
    nikkei_percentile = _compute_percentile_rank(nikkei_history, nikkei_value)

    vix_classification = _classify_vix(vix_value)
    usdidr_classification = _classify_usdidr_change(usdidr_value, usdidr_history)
    sp500_classification = _classify_index_change("S&P 500", sp500_value, sp500_history)
    nikkei_classification = _classify_index_change("Nikkei 225", nikkei_value, nikkei_history)

    overall_outlook = _derive_overall_outlook({
        "VIX": vix_classification["sentiment"],
        "USD/IDR": usdidr_classification["sentiment"],
        "S&P 500": sp500_classification["sentiment"],
        "Nikkei 225": nikkei_classification["sentiment"],
    })

    micro_outlook_5dd = _calculate_micro_outlook(5)
    micro_outlook_10dd = _calculate_micro_outlook(10)

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vix": {
            "date": vix_date,
            "value": vix_value,
            "percentile_rank": vix_percentile,
            "classification": vix_classification,
        },
        "usdidr": {
            "date": usdidr_date,
            "value": usdidr_value,
            "percentile_rank": usdidr_percentile,
            "classification": usdidr_classification,
        },
        "sp500": {
            "date": sp500_date,
            "value": sp500_value,
            "percentile_rank": sp500_percentile,
            "classification": sp500_classification,
        },
        "nikkei": {
            "timestamp": nikkei_timestamp,
            "value": nikkei_value,
            "is_live": nikkei_is_live,
            "percentile_rank": nikkei_percentile,
            "classification": nikkei_classification,
        },
        "overall_outlook": overall_outlook,
        "micro_outlook_5dd": micro_outlook_5dd,
        "micro_outlook_10dd": micro_outlook_10dd,
    }

    _save_outlook_to_json(result)

    return result