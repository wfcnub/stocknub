import os
import glob
import numpy as np
import pandas as pd

def _generate_emiten_avg_valuation_df() -> pd.DataFrame:
    """
    (Internal Helper) Calculates each emiten's average valuation on the past 10 days of active market date
    
    Returns:
        pd.DataFrame: A pandas dataframe containing the emiten with its correspoding industry and average valuation
    """
    files = glob.glob(os.path.join('data/stock/historical/', "*.csv"))
    all_emiten = [file.split('/')[-1].split('.')[0] for file in files]

    all_avg_valuation = []

    for file in files:
        emiten_df = pd.read_csv(file).tail(10)
        avg_valuation = np.mean(emiten_df['Volume'] * emiten_df[['Open', 'High', 'Close', 'Low']].mean(axis=1))
        all_avg_valuation.append(avg_valuation)

    all_emiten_valuation_df = pd.DataFrame({
        'Kode': all_emiten,
        'Average Valuation': all_avg_valuation
    })

    emiten_industry_df = pd.read_csv('data/emiten_and_industry_list.csv')
    emiten_industry_valuation_df = pd.merge(
        emiten_industry_df,
        all_emiten_valuation_df,
        on='Kode',
        how='inner'
    )
    emiten_industry_valuation_df = emiten_industry_valuation_df[emiten_industry_valuation_df['Average Valuation'] > 0]

    return emiten_industry_valuation_df

def _select_percentile_emiten_industry_df(emiten_industry_valuation_df: pd.DataFrame, perc_emiten_in_industry: float) -> pd.DataFrame:
    """
    (Internal Helper) Select emiten from each industry that has a average valuation higher than a certain percentile of average valuation on that industry
    
    Args:
        emiten_industry_valuation_df (pd.DataFrame): A pandas dataframe containing the emiten with its correspoding industry and average valuation
        perc_emiten_in_industry (float): The percentile value for the threshold of the average valuation

    Returns:
        pd.DataFrame: A pandas dataframe containing the selected emiten
    """
    industry_quantile_df = emiten_industry_valuation_df.groupby('Industri') \
                                                        ['Average Valuation'] \
                                                        .quantile(perc_emiten_in_industry) \
                                                        .to_frame('Threshold')

    perc_emiten_industry = pd.merge(
        emiten_industry_valuation_df,
        industry_quantile_df,
        on='Industri',
        how='inner'
    )

    perc_emiten_industry = perc_emiten_industry[perc_emiten_industry['Average Valuation'] > perc_emiten_industry['Threshold']]
    perc_emiten_industry.drop(columns=['Average Valuation', 'Threshold'], inplace=True)

    return perc_emiten_industry

def _select_percentile_emiten_df(emiten_industry_valuation_df: pd.DataFrame, perc_emiten: float) -> pd.DataFrame:
    """
    (Internal Helper) Select emiten that has a average valuation higher than a certain percentile from the entire emiten's average valuation
    
    Args:
        emiten_industry_valuation_df (pd.DataFrame): A pandas dataframe containing the emiten with its correspoding industry and average valuation
        perc_emiten (float): The percentile value for the threshold of the average valuation

    Returns:
        pd.DataFrame: A pandas dataframe containing the selected emiten
    """
    n_emiten = np.ceil(len(emiten_industry_valuation_df) * (1 - perc_emiten)).astype(int)
    perc_emiten = emiten_industry_valuation_df.sort_values('Average Valuation', ascending=False) \
                                                    .head(n_emiten) \
                                                    .drop(columns=['Average Valuation'])
                                                
    return perc_emiten


def _combine_both_emiten_df(perc_emiten_industry: pd.DataFrame, perc_emiten: pd.DataFrame) -> pd.DataFrame:
    """
    (Internal Helper) Combining the selected emiten from two different approach of selecting an emiten based on the average valuation

    Args:
        perc_emiten_industry (pd.DataFrame): A pandas dataframe containing the selected emiten based on the industry's threshold
        perc_emiten (pd.DataFrame): A pandas dataframe containing the selected emiten based on the all emiten's threshold
    
    Returns:
        pd.DataFrame: A pandas dataframe containing the combined result from the two different emiten selection approach
    """
    selected_emiten_df = pd.concat((perc_emiten_industry, perc_emiten)) \
                            .drop_duplicates('Kode') \
                            .reset_index(drop=True)
                        
    return selected_emiten_df