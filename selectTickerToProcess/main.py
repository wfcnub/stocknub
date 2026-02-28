import pandas as pd

from selectTickerToProcess.helper import (
    _generate_ticker_avg_valuation_df, 
    _select_percentile_ticker_industry_df, 
    _select_percentile_ticker_df, 
    _combine_both_ticker_df
)

def select_ticker_to_process(ohlcv_folder_path: str, perc_ticker_in_industry: float, perc_ticker: float) -> pd.DataFrame:
    """
    A process of selecting an ticker based on the ticker's latest average valuation

    This process combines two appraoch of ticker selection process
        1. Select ticker with the top certain percent of average valuation compared to the corresponding industry's average valuation
        2. Select ticker with the top certain percent of average valuation compared to the entire ticker's average valuation

    Args:
        perc_ticker_in_industry (float): The percentile value for the threshold of the average valuation for the first approach
        perc_ticker (float): The percentile value for the threshold of the average valuation for the second approach
    
    Returns:
        pd.DataFrame: A pandas dataframe that combines the result of the selected ticker from the two approaches
    """
    ticker_industry_valuation_df = _generate_ticker_avg_valuation_df(ohlcv_folder_path)
    
    perc_ticker_industry = _select_percentile_ticker_industry_df(ticker_industry_valuation_df, perc_ticker_in_industry)

    perc_ticker = _select_percentile_ticker_df(ticker_industry_valuation_df, perc_ticker)

    selected_ticker_df = _combine_both_ticker_df(perc_ticker_industry, perc_ticker)

    return selected_ticker_df