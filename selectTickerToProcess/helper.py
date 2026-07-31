import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests

FUNDAMENTAL_FIELDS = [
    "trailingPE",
    "forwardPE",
    "pegRatio",
    "priceToBook",
    "returnOnEquity",
    "returnOnAssets",
    "profitMargins",
    "operatingMargins",
    "debtToEquity",
    "currentRatio",
    "quickRatio",
    "freeCashflow",
    "operatingCashflow",
    "revenueGrowth",
    "earningsGrowth",
    "averageVolume",
    "volume",
    "marketCap",
    "regularMarketPrice",
]

_thread_state = threading.local()


def _get_session():
    if not hasattr(_thread_state, "session"):
        _thread_state.session = requests.Session(impersonate="chrome123")
    return _thread_state.session


def _fetch_one_fundamental(ticker: str) -> dict:
    ticker_symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
    error = ""
    for attempt in range(3):
        try:
            ticker_yf = yf.Ticker(ticker_symbol, session=_get_session())
            info = ticker_yf.info
            record = {"Ticker": ticker}
            record.update({field: info.get(field) for field in FUNDAMENTAL_FIELDS})
            available = sum(pd.notna(record[field]) for field in FUNDAMENTAL_FIELDS)
            if available == 0:
                raise ValueError("provider returned no fundamental fields")
            record.update(
                {
                    "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "fetch_status": "success",
                    "fetch_error": "",
                }
            )
            return record
        except Exception as exc:
            error = str(exc)
            if attempt < 2:
                time.sleep((2**attempt) + random.uniform(0, 1))

    return {
        "Ticker": ticker,
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "fetch_status": "failed",
        "fetch_error": error,
        **{field: np.nan for field in FUNDAMENTAL_FIELDS},
    }


def _load_fundamental_history(cache_path: Path | None) -> pd.DataFrame:
    if cache_path is None or not cache_path.is_file():
        return pd.DataFrame()
    try:
        history = pd.read_csv(cache_path)
        if "Ticker" not in history or "fetched_at" not in history:
            return pd.DataFrame()
        return history
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def _latest_successful_fundamentals(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    successful = history[
        history.get("fetch_status", pd.Series("success", index=history.index)).eq("success")
    ].copy()
    if successful.empty:
        return successful
    successful["_fetched_at"] = pd.to_datetime(
        successful["fetched_at"], errors="coerce", utc=True
    )
    return (
        successful.sort_values("_fetched_at")
        .drop_duplicates("Ticker", keep="last")
        .drop(columns="_fetched_at")
    )


def _fetch_fundamentals(
    tickers: list,
    cache_path: str | Path | None = None,
    ttl_days: int = 30,
    max_workers: int = 4,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    (Internal Helper) Queries yfinance to acquire robust fundamentals for a list of tickers.
    
    Args:
        tickers (list): A list of ticker symbols (e.g. ['BBCA', 'AAPL'])
        
    Returns:
        pd.DataFrame: A dataframe containing fundamental metrics for each ticker
    """
    requested = list(dict.fromkeys(tickers))
    cache = Path(cache_path) if cache_path is not None else None
    history = _load_fundamental_history(cache)
    latest = _latest_successful_fundamentals(history)
    latest_by_ticker = latest.set_index("Ticker") if not latest.empty else pd.DataFrame()

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=ttl_days)
    to_fetch = []
    for ticker in requested:
        if force_refresh or latest_by_ticker.empty or ticker not in latest_by_ticker.index:
            to_fetch.append(ticker)
            continue
        fetched_at = pd.to_datetime(
            latest_by_ticker.loc[ticker, "fetched_at"], errors="coerce", utc=True
        )
        if pd.isna(fetched_at) or fetched_at < cutoff:
            to_fetch.append(ticker)

    fetched_records = []
    if to_fetch:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(to_fetch)))) as pool:
            futures = {
                pool.submit(_fetch_one_fundamental, ticker): ticker for ticker in to_fetch
            }
            for future in as_completed(futures):
                fetched_records.append(future.result())

    if fetched_records:
        fetched = pd.DataFrame(fetched_records)
        history = pd.concat([history, fetched], ignore_index=True, sort=False)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache.with_suffix(cache.suffix + ".tmp")
            history.to_csv(temporary_path, index=False)
            temporary_path.replace(cache)

    latest = _latest_successful_fundamentals(history)
    if latest.empty:
        return pd.DataFrame(columns=["Ticker", *FUNDAMENTAL_FIELDS])
    result = latest[latest["Ticker"].isin(requested)].copy()
    result["fundamental_cache_stale"] = (
        pd.to_datetime(result["fetched_at"], errors="coerce", utc=True) < cutoff
    )
    return result.reset_index(drop=True)

def _num(df: pd.DataFrame, col: str, fallback: float = 0) -> pd.Series:
    """Helper to safely convert a column to numeric, filling NaNs."""
    return pd.to_numeric(df[col], errors='coerce').fillna(fallback)

def _calc_profitability_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the profitability score based on Return on Equity (ROE), Return on Assets (ROA),
    Profit Margin, and Operating Margin. Higher values for these metrics result in a higher score.
    
    Args:
        df (pd.DataFrame): DataFrame containing fundamental metrics.
        
    Returns:
        pd.Series: A series containing the computed profitability score.
    """
    roe             = _num(df, 'returnOnEquity', -1)
    roa             = _num(df, 'returnOnAssets', -1)
    profit_margin   = _num(df, 'profitMargins', -1)
    operating_margin = _num(df, 'operatingMargins', -1)
    
    prof_score = (roe * 100).clip(lower=-50, upper=80) + \
                 (roa * 100).clip(lower=-50, upper=80) + \
                 (profit_margin * 100).clip(lower=-50, upper=70) + \
                 (operating_margin * 100).clip(lower=-50, upper=70)
    return prof_score

def _calc_growth_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the growth score based on Revenue Growth and Earnings Growth.
    Companies with higher year-over-year growth metrics receive a higher score.
    
    Args:
        df (pd.DataFrame): DataFrame containing fundamental metrics.
        
    Returns:
        pd.Series: A series containing the computed growth score.
    """
    rev_growth  = _num(df, 'revenueGrowth', -1)
    earn_growth = _num(df, 'earningsGrowth', -1)
    
    growth_score = (rev_growth * 100).clip(lower=-50, upper=150) + \
                   (earn_growth * 100).clip(lower=-50, upper=150)
    return growth_score

def _calc_valuation_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the valuation score inversely correlated with traditional valuation multiples.
    Lower PEG, Forward PE, Trailing PE, and Price-to-Book ratios yield higher scores, indicating better value.
    
    Args:
        df (pd.DataFrame): DataFrame containing fundamental metrics.
        
    Returns:
        pd.Series: A series containing the computed valuation score.
    """
    peg = _num(df, 'pegRatio', np.nan).where(lambda value: value > 0)
    fwd_pe = _num(df, 'forwardPE', np.nan).where(lambda value: value > 0)
    trailing_pe = _num(df, 'trailingPE', np.nan).where(lambda value: value > 0)
    pb = _num(df, 'priceToBook', np.nan).where(lambda value: value > 0)
    
    val_score = (50 / peg).clip(upper=100).fillna(0) + \
                (200 / fwd_pe).clip(upper=75).fillna(0) + \
                (200 / trailing_pe).clip(upper=75).fillna(0) + \
                (25 / pb).clip(upper=50).fillna(0)
    return val_score

def _calc_health_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the financial health score by evaluating short-term liquidity and leverage.
    Higher Current Ratio and Quick Ratio enhance the score, while a higher Debt-to-Equity ratio penalizes it.
    
    Args:
        df (pd.DataFrame): DataFrame containing fundamental metrics.
        
    Returns:
        pd.Series: A series containing the computed financial health score.
    """
    current_ratio  = _num(df, 'currentRatio', 0)
    quick_ratio    = _num(df, 'quickRatio', 0)
    debt_to_equity = _num(df, 'debtToEquity', 999)
    
    health_score = (current_ratio * 20).clip(upper=50) + \
                   (quick_ratio * 25).clip(upper=50) + \
                   (200 - debt_to_equity).clip(lower=-50, upper=100)
    return health_score

def _calc_cashflow_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the cash flow quality score based on Free Cash Flow Yield and Operating Cash Flow Yield.
    Yields are computed by comparing the cash flows against the company's Market Cap.
    
    Args:
        df (pd.DataFrame): DataFrame containing fundamental metrics.
        
    Returns:
        pd.Series: A series containing the computed cash flow score.
    """
    fcf        = _num(df, 'freeCashflow', 0)
    ocf        = _num(df, 'operatingCashflow', 0)
    market_cap = _num(df, 'marketCap', 0)
    
    safe_mcap = market_cap.where(market_cap > 0, np.nan)
    fcf_yield = (fcf / safe_mcap * 100).fillna(-50)
    ocf_yield = (ocf / safe_mcap * 100).fillna(-50)
    
    cashflow_score = (fcf_yield * 10).clip(lower=-50, upper=100) + \
                     (ocf_yield * 10).clip(lower=-50, upper=100)
    return cashflow_score

def _calc_liquidity_score(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates the liquidity score using a graduated logarithmic scale based on the average daily value traded.
    Also issues a severe penalty gate if the traded value falls below the 5 Billion IDR threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing fundamental metrics.
        
    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: A tuple containing the liquidity score, the liquidity penalty gate,
                                                and the raw average value traded.
    """
    avg_vol = _num(df, 'averageVolume', 0)
    price   = _num(df, 'regularMarketPrice', 0)
    
    avg_value_traded = avg_vol * price
    safe_value = avg_value_traded.clip(lower=1)
    
    threshold = 5 * 1e9
    
    liquidity_score = ((np.log10(safe_value) - np.log10(threshold)) * 50).clip(lower=-100, upper=200)
    liquidity_gate = (avg_value_traded < threshold).astype(int) * -10000
    
    return liquidity_score, liquidity_gate, avg_value_traded

def _calculate_fundamental_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Internal Helper) Calculates a comprehensive fundamental score to rank tickers.
    
    Combines six factor categories — profitability, growth, valuation, financial health,
    cash flow quality, and liquidity — to identify stocks with strong upside potential
    that are also liquid and actively traded.
    
    Score budget (approximate max per category):
        1. Profitability  : +300  (ROE, ROA, Profit Margin, Operating Margin)
        2. Growth         : +300  (Revenue Growth, Earnings Growth)
        3. Valuation      : +300  (PEG, Forward PE, Trailing PE, P/B)
        4. Financial Health: +200  (Current Ratio, Quick Ratio, Debt-to-Equity)
        5. Cash Flow      : +200  (FCF Yield, OCF Yield)
        6. Liquidity      : +200  (graduated, based on avg daily value traded)
        Hard gate         : -10000 if avg daily value traded < 5B IDR
    
    Args:
        df (pd.DataFrame): DataFrame with fundamental metrics.
    
    Returns:
        pd.DataFrame: DataFrame with a new 'fundamental_score' column.
    """
    scored_df = df.copy()
    
    prof_score = _calc_profitability_score(scored_df)
    growth_score = _calc_growth_score(scored_df)
    val_score = _calc_valuation_score(scored_df)
    health_score = _calc_health_score(scored_df)
    cashflow_score = _calc_cashflow_score(scored_df)
    liquidity_score, liquidity_gate, avg_value_traded = _calc_liquidity_score(scored_df)
    
    scored_df['avg_value_traded'] = avg_value_traded
    
    scored_df['fundamental_score'] = (
        prof_score
        + growth_score
        + val_score
        + health_score
        + cashflow_score
        + liquidity_score
        + liquidity_gate
    )
    
    return scored_df
