from selectEmitenToProcess.helper import _generate_emiten_avg_valuation_df, _select_percentile_emiten_industry_df, _select_percentile_emiten_df, _combine_both_emiten_df

def select_emiten_to_process(perc_emiten_in_industry, perc_emiten):
    """
    A process of selecting an emiten based on the emiten's latest average valuation

    This process combines two appraoch of emiten selection process
        1. Select emiten with the top certain percent of average valuation compared to the corresponding industry's average valuation
        2. Select emiten with the top certain percent of average valuation compared to the entire emiten's average valuation

    Args:
        perc_emiten_in_industry (float): The percentile value for the threshold of the average valuation for the first approach
        perc_emiten (float): The percentile value for the threshold of the average valuation for the second approach
    
    Returns:
        pd.DataFrame: A pandas dataframe that combines the result of the selected emiten from the two approaches
    """
    emiten_industry_valuation_df = _generate_emiten_avg_valuation_df()
    
    perc_emiten_industry = _select_percentile_emiten_industry_df(emiten_industry_valuation_df, perc_emiten_in_industry)

    perc_emiten = _select_percentile_emiten_df(emiten_industry_valuation_df, perc_emiten)

    selected_emiten_df = _combine_both_emiten_df(perc_emiten_industry, perc_emiten)

    return selected_emiten_df