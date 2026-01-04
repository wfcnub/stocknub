import os
import glob
import time
import calendar
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC

def _get_latest_date(download_dir):
    files = glob.glob(os.path.join(download_dir, "*.csv"))
    latest_file = max(files)
    latest_date = datetime.strptime(os.path.basename(latest_file).split('.')[0], '%Y%m%d').strftime('%Y-%m-%d')

    return latest_date

def _get_all_weekdays_from_selected_date(selected_date):
    start_date = datetime.strptime(selected_date, "%Y-%m-%d")
    
    today = datetime.now()
    
    weekday_dates = []
    
    current_date = start_date
    while current_date <= today:
        if current_date.weekday() < 5:
            weekday_dates.append(current_date.strftime("%Y-%m-%d"))
        
        current_date += timedelta(days=1)
        
    return weekday_dates

def _get_all_weekstart_to_backfill(csv_folder_path, weekday_dates):
    fetched_weekstart_dates = set([datetime.strptime(f.stem, '%Y%m%d').strftime('%Y-%m-%d') for f in Path(csv_folder_path).iterdir() if f.is_file() and f.suffix == '.csv'])

    backfill_weekstart_dates = list(set(weekday_dates) - fetched_weekstart_dates)

    return backfill_weekstart_dates
    
def _wait_before_click(driver, tag, attr_name, attr_val):
    wait = WebDriverWait(driver, 5)
    element = f"//{tag}[@{attr_name}='{attr_val}']"
    
    try:
        _ = wait.until(EC.element_to_be_clickable((By.XPATH, element)))
        if not driver.find_element(By.XPATH, element).is_enabled():
            error_msg = "The button is being disabled"
            print(error_msg)

            return False

        driver.find_element(By.XPATH, element).click()
        
    except TimeoutException:
        error_msg = "Error: Operation timed out. The button was not ready within 10 seconds."
        print(error_msg)

        return False
    
    except Exception as e:
        error_msg = f"Error: An unexpected issue occurred while clicking: {e}"
        print(error_msg)

        return False
    
    return True

def _initialize_driver(download_dir):     
    chrome_options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=chrome_options)

    target_url = 'https://www.idx.co.id/en/market-data/trading-summary/stock-summary'
    driver.get(target_url)
    
    driver.maximize_window()

    return driver

def _check_if_download_completion(download_dir, weekday_dt, timeout=10):
    filename = f"Stock Summary-{weekday_dt.replace('-', '')}.xlsx"
    end_time = time.time() + timeout

    while time.time() < end_time:
        full_path = os.path.join(download_dir, filename)
        
        if os.path.isfile(full_path):
            return
        else:
            time.sleep(1)
    
    return

def _get_latest_date(download_dir):
    files = glob.glob(os.path.join(download_dir, "*.csv"))
    latest_file = max(files)
    latest_date = datetime.strptime(os.path.basename(latest_file).split('.')[0], '%Y%m%d').strftime('%Y-%m-%d')

    return latest_date

def wait_for_page_stability(driver, timeout=30, check_interval=0.5):
    end_time = time.time() + timeout
    previous_source = ""
    
    while time.time() < end_time:
        current_source = driver.page_source
        
        if current_source == previous_source:
            return True
            
        previous_source = current_source
        time.sleep(check_interval)

    print("Warning: Page kept changing until timeout.")
    return False

def _select_year_month_on_web(driver, year, month):
    # div for changing the date
    _ = _wait_before_click(driver, 'div', 'class', 'mx-input-wrapper')
    
    # button for changing the year
    _ = _wait_before_click(driver, 'button', 'class', 'mx-btn mx-btn-text mx-btn-current-year')
    
    # button for selecting the year
    _ = _wait_before_click(driver, 'td', 'data-year', year)
    
    # button for selecting the month
    _ = _wait_before_click(driver, 'td', 'data-month', month-1)

    return

def _select_and_download_specific_date_on_web(driver, download_dir, weekday_dt):
    # div for changing the date
    _ = _wait_before_click(driver, 'div', 'class', 'mx-input-wrapper')
    
    # button for selecting the day
    _ = _wait_before_click(driver, 'td', 'title', weekday_dt)
    
    # wait before everything finished being loaded
    _ = wait_for_page_stability(driver)

    # button for downloading the day
    check_data_bool = _wait_before_click(driver, 'button', 'class', 'btn-filter-input mb-8 btn-download text-center')

    if check_data_bool:
        _check_if_download_completion(download_dir, weekday_dt)

        return True
    
    return False

def _parse_indo_date(date_str):
    indo_months = {
        'Mei': 'May',
        'Agt': 'Aug',
        'Okt': 'Oct',
        'Des': 'Dec'
    }
    
    for indo, eng in indo_months.items():
        if indo in date_str:
            date_str = date_str.replace(indo, eng)
            break
            
    return datetime.strptime(date_str, '%d %b %Y').strftime('%Y-%m-%d')

def _clean_downloaded_data(download_dir, weekday_dt):    
    filename = f"Stock Summary-{weekday_dt.replace('-', '')}.xlsx"
    full_path = os.path.join(download_dir, filename)

    selected_columns = [
        'Stock Code',
        'Last Trading Date',
        'Foreign Sell',
        'Foreign Buy',
        'Non Regular Volume',
        'Non Regular Value',
        'Non Regular Frequency'
    ]
    
    data = pd.read_excel(full_path, usecols=selected_columns)
    data['Last Trading Date'] = data['Last Trading Date'].apply(lambda val: _parse_indo_date(val))

    os.remove(full_path)
    
    save_filename = f"{data['Last Trading Date'].unique()[0].replace('-', '')}.csv"
    save_full_path = os.path.join(download_dir, save_filename)
    
    data.to_csv(save_full_path, index=False)

    return