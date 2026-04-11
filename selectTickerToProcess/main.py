import pandas as pd

from selectTickerToProcess.helper import (
    _fetch_fundamentals,
    _calculate_fundamental_score,
)

def select_ticker_to_process(ticker: str) -> pd.DataFrame | None:
    """
    A process of selecting a ticker based on fundamental analysis.

    This function evaluates a single ticker to check its metrics and compute its fundamental score.

    Args:
        ticker (str): The ticker symbol to process
    
    Returns:
        pd.DataFrame | None: A pandas dataframe containing the ticker's data, or None if it fails.
    """
    
    fundamental_df = _fetch_fundamentals([ticker])
    
    if fundamental_df.empty:
        return None
        
    scored_ticker_df = _calculate_fundamental_score(fundamental_df)

    if scored_ticker_df.empty:
        return None

    return scored_ticker_df