import time
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests

def _fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """
    (Internal Helper) Queries yfinance to acquire robust fundamentals for a list of tickers.
    
    Args:
        tickers (list): A list of ticker symbols (e.g. ['BBCA', 'AAPL'])
        
    Returns:
        pd.DataFrame: A dataframe containing fundamental metrics for each ticker
    """
    session = requests.Session(impersonate="chrome123")
    data = []
    
    for ticker in tickers:
        retries = 3
        while retries > 0:
            try:
                ticker_symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
                ticker_yf = yf.Ticker(ticker_symbol, session=session)
                
                info = ticker_yf.info
                
                data.append({
                    "Ticker": ticker,
                    "trailingPE": info.get("trailingPE"),
                    "forwardPE": info.get("forwardPE"),
                    "pegRatio": info.get("pegRatio"),
                    "priceToBook": info.get("priceToBook"),
                    "returnOnEquity": info.get("returnOnEquity"),
                    "returnOnAssets": info.get("returnOnAssets"),
                    "profitMargins": info.get("profitMargins"),
                    "operatingMargins": info.get("operatingMargins"),
                    "debtToEquity": info.get("debtToEquity"),
                    "currentRatio": info.get("currentRatio"),
                    "quickRatio": info.get("quickRatio"),
                    "freeCashflow": info.get("freeCashflow"),
                    "operatingCashflow": info.get("operatingCashflow"),
                    "revenueGrowth": info.get("revenueGrowth"),
                    "earningsGrowth": info.get("earningsGrowth"),
                    "averageVolume": info.get("averageVolume"),
                    "volume": info.get("volume"),
                    "marketCap": info.get("marketCap"),
                    "regularMarketPrice": info.get("regularMarketPrice"),
                })
                break

            except Exception as e:
                error_msg = str(e).lower()
                if '429' in error_msg or 'rate limit' in error_msg or 'too many requests' in error_msg:
                    print(f"Rate limit hit for {ticker}. Waiting 3 minutes before continuing...")
                    time.sleep(180)
                    retries -= 1
                else:
                    print(f"Failed to fetch {ticker}: {e}")
                    break
            
    df = pd.DataFrame(data)
    return df

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
    peg        = _num(df, 'pegRatio', 999).clip(lower=0.1)
    fwd_pe     = _num(df, 'forwardPE', 999).clip(lower=1)
    trailing_pe = _num(df, 'trailingPE', 999).clip(lower=1)
    pb         = _num(df, 'priceToBook', 999).clip(lower=0.1)
    
    val_score = (50  / peg).clip(upper=100) + \
                (200 / fwd_pe).clip(upper=75) + \
                (200 / trailing_pe).clip(upper=75) + \
                (25  / pb).clip(upper=50)
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
        Hard gate         : -10000 if avg daily value traded < 1B IDR
    
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