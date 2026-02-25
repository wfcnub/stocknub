import os
import numpy as np
import pandas as pd
from pathlib import Path

def _generate_ticker_avg_valuation_df(ohlcv_folder_path: str) -> pd.DataFrame:
    """
    (Internal Helper) Calculates each ticker's average valuation on the past 60 days of active market date

    Args:
        ohlcv_folder_path (str): The directory path where the OHLCV data is stored
    
    Returns:
        pd.DataFrame: A pandas dataframe containing the ticker with its correspoding industry and average valuation
    """
    files = list(Path(ohlcv_folder_path).rglob("*.csv"))
    all_tickers = [file.stem for file in Path(ohlcv_folder_path).rglob("*.csv")]

    all_avg_valuation = []

    for file in files:
        ticker_df = pd.read_csv(file).tail(60)
        avg_valuation = np.mean(ticker_df['Volume'] * ticker_df[['Open', 'High', 'Close', 'Low']].mean(axis=1))
        all_avg_valuation.append(avg_valuation)

    all_ticker_valuation_df = pd.DataFrame({
        'Ticker': all_tickers,
        'Average Valuation': all_avg_valuation
    })

    ticker_industry_df = pd.read_csv('data/ticker_and_industry_list.csv')
    ticker_industry_valuation_df = pd.merge(
        ticker_industry_df,
        all_ticker_valuation_df,
        on='Ticker',
        how='inner'
    )
    ticker_industry_valuation_df = ticker_industry_valuation_df[ticker_industry_valuation_df['Average Valuation'] > 0]

    return ticker_industry_valuation_df

def _select_percentile_ticker_industry_df(ticker_industry_valuation_df: pd.DataFrame, perc_ticker_in_industry: float) -> pd.DataFrame:
    """
    (Internal Helper) Select ticker from each industry that has a average valuation higher than a certain percentile of average valuation on that industry
    
    Args:
        ticker_industry_valuation_df (pd.DataFrame): A pandas dataframe containing the emiten with its correspoding industry and average valuation
        perc_ticker_in_industry (float): The percentile value for the threshold of the average valuation

    Returns:
        pd.DataFrame: A pandas dataframe containing the selected ticker
    """
    industry_quantile_df = ticker_industry_valuation_df.groupby('Industry') \
                                                        ['Average Valuation'] \
                                                        .quantile(perc_ticker_in_industry) \
                                                        .to_frame('Threshold')

    perc_ticker_industry = pd.merge(
        ticker_industry_valuation_df,
        industry_quantile_df,
        on='Industry',
        how='inner'
    )

    perc_ticker_industry = perc_ticker_industry[perc_ticker_industry['Average Valuation'] > perc_ticker_industry['Threshold']]
    perc_ticker_industry.drop(columns=['Average Valuation', 'Threshold'], inplace=True)

    return perc_ticker_industry

def _select_percentile_ticker_df(ticker_industry_valuation_df: pd.DataFrame, perc_ticker: float) -> pd.DataFrame:
    """
    (Internal Helper) Select ticker that has an average valuation higher than a certain percentile from the entire ticker's average valuation
    
    Args:
        ticker_industry_valuation_df (pd.DataFrame): A pandas dataframe containing the ticker with its correspoding industry and average valuation
        perc_ticker (float): The percentile value for the threshold of the average valuation

    Returns:
        pd.DataFrame: A pandas dataframe containing the selected ticker
    """
    n_ticker = np.ceil(len(ticker_industry_valuation_df) * (1 - perc_ticker)).astype(int)
    perc_ticker = ticker_industry_valuation_df.sort_values('Average Valuation', ascending=False) \
                                                    .head(n_ticker) \
                                                    .drop(columns=['Average Valuation'])
                                                
    return perc_ticker


def _combine_both_ticker_df(perc_ticker_industry: pd.DataFrame, perc_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    (Internal Helper) Combining the selected ticker from two different approach of selecting an ticker based on the average valuation

    Args:
        perc_ticker_industry (pd.DataFrame): A pandas dataframe containing the selected ticker based on the industry's threshold
        perc_ticker (pd.DataFrame): A pandas dataframe containing the selected ticker based on the all ticker's threshold
    
    Returns:
        pd.DataFrame: A pandas dataframe containing the combined result from the two different ticker selection approach
    """
    selected_emiten_df = pd.concat((perc_ticker_industry, perc_ticker)) \
                            .drop_duplicates('Ticker') \
                            .reset_index(drop=True)
                        
    return selected_emiten_df