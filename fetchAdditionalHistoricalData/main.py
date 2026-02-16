import os
import glob
import pandas as pd
from datetime import datetime

from fetchAdditionalHistoricalData.helper import _initialize_driver, _select_year_month_on_web, _select_and_download_specific_date_on_web, _clean_downloaded_data

def fetch_additional_emiten_data(active_market_dates: list, download_dir: str) -> list:
    """
    Perform the process of fetching the additional historical data from the IDX website using web scraping

    Args:
        active_market_dates (list): A list containing all active market dates
        download_dir (str): The directory where the downloaded data will be stored
    
    Returns:
        list: A list containing the downloaded data, status of the downloaded data, and status message
    """
    results = []

    driver = _initialize_driver(download_dir)

    for active_market_date in active_market_dates:
        try:
            year = datetime.strptime(active_market_date, '%Y-%m-%d').year
            month = datetime.strptime(active_market_date, '%Y-%m-%d').month
                
            _select_year_month_on_web(driver, year, month)
                    
            process_data_bool = _select_and_download_specific_date_on_web(driver, download_dir, active_market_date)

            if process_data_bool:
                _clean_downloaded_data(download_dir, active_market_date)

            results.append((active_market_date, True, f"Succesfully fetch data on {active_market_date}."))

        except:
            results.append((active_market_date, False, f"No data found on {active_market_date}."))
            
    driver.close()

    return results

def process_additional_historical_data(csv_folder_path: str, processed_folder_path: str) -> None:
    """
    Clean and reformat the downloaded additional historical data

    Args:
        csv_folder_path (str): The directory where raw additional historical data is being stored
        processed_folder_path (str): The directory where the processed additional historical data will be stored
    """
    csv_files = glob.glob(os.path.join(csv_folder_path, "*.csv"))
    df_list = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)

    all_emitens = combined_df['Stock Code'].unique()
    
    for emiten in all_emitens:
        emiten_df = combined_df.loc[combined_df['Stock Code'] == emiten] \
                                    .sort_values('Last Trading Date', ascending=True) \
                                    .rename(columns={'Last Trading Date': 'Date'}) \
                                    .drop(columns=['Stock Code']) \
                                    .drop_duplicates('Date') \
                                    .reset_index(drop=True)
        
        save_full_path = os.path.join(processed_folder_path, f'{emiten}.csv')
        emiten_df.to_csv(save_full_path, index=False)

    return