import os
import pandas as pd
from pathlib import Path
from datetime import datetime

from fetchForeignFlowAndNonRegularData.helper import (
    _initialize_driver, 
    _select_year_month_on_web, 
    _select_and_download_specific_date_on_web, 
    _clean_downloaded_data
)

def fetch_foreign_flow_and_non_regular_ticker_data(active_market_dates: list, raw_csv_folder_path: str) -> list:
    """
    Fetch an additional historical data containing foreign flow and non regular market from the IDX website using web scraping

    Args:
        active_market_dates (list): A list containing all active market dates
        raw_csv_folder_path (str): The directory where the downloaded data will be stored
    
    Returns:
        list: A list containing the downloaded data, status of the downloaded data, and status message
    """
    results = []

    driver = _initialize_driver(raw_csv_folder_path)

    for active_market_date in active_market_dates:
        try:
            year = datetime.strptime(active_market_date, '%Y-%m-%d').year
            month = datetime.strptime(active_market_date, '%Y-%m-%d').month
                
            _select_year_month_on_web(driver, year, month)
                    
            process_data_bool = _select_and_download_specific_date_on_web(driver, raw_csv_folder_path, active_market_date)
            
            if process_data_bool:
                _clean_downloaded_data(raw_csv_folder_path, active_market_date)

            results.append((active_market_date, True, f"Succesfully fetch data on {active_market_date}."))

        except Exception as e:
            results.append((active_market_date, False, f"No data found on {active_market_date}: {e}"))
            
    driver.close()

    return results

def process_foreign_flow_and_non_regular_ticker_data(args_tuple) -> None:
    """
    Clean and reformat the downloaded additional historical data

    Args:
        args_tuple (tuple): A Tuple containing (ticker, combined_df, csv_folder_path)
    """
    ticker, combined_df, csv_folder_path = args_tuple

    ticker_df = combined_df.loc[combined_df['Stock Code'] == ticker] \
                                .sort_values('Last Trading Date', ascending=True) \
                                .rename(columns={'Last Trading Date': 'Date'}) \
                                .drop(columns=['Stock Code']) \
                                .drop_duplicates('Date') \
                                .reset_index(drop=True)
    
    save_path = (Path(csv_folder_path) / ticker).with_suffix('.csv')
    ticker_df.to_csv(save_path, index=False)

    return