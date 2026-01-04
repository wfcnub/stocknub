import os
import glob
import pandas as pd
from datetime import datetime

from fetchAdditionalHistoricalData.helper import _initialize_driver, _select_year_month_on_web, _select_and_download_specific_date_on_web, _clean_downloaded_data

def fetch_additional_emiten_data(weekday_dates, download_dir):
    results = []

    driver = _initialize_driver(download_dir)

    for weekday_dt in weekday_dates:
        try:
            year = datetime.strptime(weekday_dt, '%Y-%m-%d').year
            month = datetime.strptime(weekday_dt, '%Y-%m-%d').month
                
            _select_year_month_on_web(driver, year, month)
                    
            process_data_bool = _select_and_download_specific_date_on_web(driver, download_dir, weekday_dt)

            if process_data_bool:
                _clean_downloaded_data(download_dir, weekday_dt)

            results.append((weekday_dt, True, f"Succesfully fetch data on {weekday_dt}."))

        except:
            results.append((weekday_dt, False, f"No data found on {weekday_dt}."))
            
    driver.close()

    return results

def process_additional_historical_data(csv_folder_path, processed_folder_path):
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