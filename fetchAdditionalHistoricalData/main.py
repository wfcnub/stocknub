import os
import glob
import time
import calendar
import pandas as pd
from datetime import datetime, date, timedelta

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC

def get_current_month_weekdays():
    today_date = date.today()
    year = today_date.year
    month = today_date.month

    weekday_dates = []
    cal = calendar.Calendar(firstweekday=0)
    
    for day, weekday_code in cal.itermonthdays2(year, month):
        if day > today_date.day:
            break
            
        elif day != 0 and weekday_code < 5:
            dt = date(year, month, day).strftime('%Y-%m-%d')
            weekday_dates.append(dt)
            
    return weekday_dates
    
def get_past_month_weekdays(year, month):
    weekday_dates = []
    cal = calendar.Calendar(firstweekday=0)
    
    for day, weekday_code in cal.itermonthdays2(year, month):
        if day != 0 and weekday_code < 5:
            dt = date(year, month, day).strftime('%Y-%m-%d')
            weekday_dates.append(dt)
            
    return weekday_dates

def get_all_weekdays_from_selected_date(selected_date):
    start_date = datetime.strptime(selected_date, "%Y-%m-%d") + timedelta(days=1)
    
    today = datetime.now()
    
    weekday_dates = []
    
    current_date = start_date
    while current_date <= today:
        if current_date.weekday() < 5:
            weekday_dates.append(current_date.strftime("%Y-%m-%d"))
        
        current_date += timedelta(days=1)
        
    return weekday_dates

def create_list_of_historical_year_month(start_year=2021, start_month=1):
    current_date = date.today()
    current_year = current_date.year
    current_month = current_date.month

    date_list = []

    for year in range(start_year, current_year + 1):
        first_m = start_month if year == start_year else 1
        
        last_m = current_month if year == current_year else 12
        
        for month in range(first_m, last_m + 1):
            date_list.append((year, month))
            
    return date_list[:-1]
    
def wait_before_click(driver, tag, attr_name, attr_val):
    wait = WebDriverWait(driver, 5)
    element = f"//{tag}[@{attr_name}='{attr_val}']"
    
    try:
        _ = wait.until(EC.element_to_be_clickable((By.XPATH, element)))
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

def initialize_driver():
    download_dir = os.path.join(os.getcwd(), "downloads")
     
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

    return driver, download_dir

def check_if_download_completion(download_dir, weekday_dt, timeout=10):
    filename = f"Stock Summary-{weekday_dt.replace('-', '')}.xlsx"
    end_time = time.time() + timeout

    while time.time() < end_time:
        full_path = os.path.join(download_dir, filename)
        
        if os.path.isfile(full_path):
            return
        else:
            time.sleep(1)
    
    return

def get_latest_date(download_dir):
    files = glob.glob(os.path.join(download_dir, "*.csv"))
    latest_file = max(files)
    latest_date = datetime.strptime(os.path.basename(latest_file).split('.')[0], '%Y%m%d').strftime('%Y-%m-%d')

    return latest_date

def select_year_month_on_web(driver, year, month):
    # div for changing the date
    _ = wait_before_click(driver, 'div', 'class', 'mx-input-wrapper')
    
    # button for changing the year
    _ = wait_before_click(driver, 'button', 'class', 'mx-btn mx-btn-text mx-btn-current-year')
    
    # button for selecting the year
    _ = wait_before_click(driver, 'td', 'data-year', year)
    
    # button for selecting the month
    _ = wait_before_click(driver, 'td', 'data-month', month-1)

    return

def select_and_download_specific_date_on_web(driver, download_dir, weekday_dt):
    # div for changing the date
    _ = wait_before_click(driver, 'div', 'class', 'mx-input-wrapper')
    
    # button for selecting the day
    _ = wait_before_click(driver, 'td', 'title', weekday_dt)
    
    # button for downloading the day
    check_data_bool = wait_before_click(driver, 'button', 'class', 'btn-filter-input mb-8 btn-download text-center')

    if check_data_bool:
        check_if_download_completion(download_dir, weekday_dt)

        return True
    
    return False

def clean_downloaded_data(download_dir, weekday_dt):    
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
    data['Last Trading Date'] = data['Last Trading Date'].apply(lambda val: datetime.strptime(val, "%d %b %Y").strftime('%Y-%m-%d'))

    os.remove(full_path)
    
    save_filename = f"{weekday_dt.replace('-', '')}.csv"
    save_full_path = os.path.join(download_dir, save_filename)
    
    data.to_csv(save_full_path, index=False)

    return 
    
def get_current_month_data():
    today_date = date.today()
    year = today_date.year
    month = today_date.month
    
    driver, download_dir = initialize_driver()
    
    select_year_month_on_web(driver, year, month)
    
    weekday_dates = get_current_month_weekdays()
    
    for weekday_dt in weekday_dates:
        process_data_bool = select_and_download_specific_date_on_web(driver, download_dir, weekday_dt)

        if process_data_bool:
            clean_downloaded_data(download_dir, weekday_dt)
        
    driver.close()

    return
    
def get_all_historical_data(historical_year_month):
    driver, download_dir = initialize_driver()
        
    for year, month in historical_year_month[-1:]:
        select_year_month_on_web(driver, year, month)
        
        weekday_dates = get_past_month_weekdays(year, month)
        
        for weekday_dt in weekday_dates:
            process_data_bool = select_and_download_specific_date_on_web(driver, download_dir, weekday_dt)

            if process_data_bool:
                clean_downloaded_data(download_dir, weekday_dt)

    driver.close()

    return

def get_all_historical_data_from_selected_date():
    driver, download_dir = initialize_driver()

    latest_date = get_latest_date(download_dir)

    weekday_dates = get_all_weekdays_from_selected_date(latest_date)
        
    for weekday_dt in weekday_dates:
        year = datetime.strptime(weekday_dt, '%Y-%m-%d').year
        month = datetime.strptime(weekday_dt, '%Y-%m-%d').month
        
        select_year_month_on_web(driver, year, month)
                
        process_data_bool = select_and_download_specific_date_on_web(driver, download_dir, weekday_dt)

        if process_data_bool:
            clean_downloaded_data(download_dir, weekday_dt)

    driver.close()

    return