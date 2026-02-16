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

def _get_latest_date(download_dir: str) -> str:
    """
    (Internal Helper) Get all file names, named as a date, from a specified directory, then gets the latest date available

    Args:
        donwload_dir (str): The directory to be looked in to
    
    Returns:
        str: The latest date found on the specified directory
    """
    files = glob.glob(os.path.join(download_dir, "*.csv"))
    latest_file = max(files)
    latest_date = datetime.strptime(os.path.basename(latest_file).split('.')[0], '%Y%m%d').strftime('%Y-%m-%d')

    return latest_date

def _get_all_active_market_date(historical_data_dir: str = 'data/stock/historical') -> list:
    """
    (Internal Helper) Get all unique date from data collected from yfinance. This Dates would serve as an active market date

    Args:
        historical_data_dir (str): Directory where the historical data gathered from yfinance is stored
    
    Returns:
        list: A list containing all active market dates
    """
    all_historical_data_path = glob.glob(os.path.join(historical_data_dir, "*.csv"))
    
    all_dates = []
    for historical_data_path in all_historical_data_path:
        acquired_date = pd.read_csv(historical_data_path, usecols=['Date'])['Date'].unique().tolist()
        all_dates = list(set(acquired_date + all_dates))

    all_dates.sort()
        
    return all_dates

def _get_all_weekstart_to_backfill(additional_historical_data_dir: str, active_market_dates: str) -> list:
    """
    (Internal Helper) Get all market dates that have not yet get collected

    Args:
        additional_historical_data_dir (str): The directory where the additional historical data is stored
        active_market_dates (list): A list containing all active market dates
    
    Return:
        list: A list containing all active market dates to be backfilled
    """
    fetched_active_market_dates = set([datetime.strptime(f.stem, '%Y%m%d').strftime('%Y-%m-%d') for f in Path(additional_historical_data_dir).iterdir() if f.is_file() and f.suffix == '.csv'])

    backfill_active_market_dates = list(set(active_market_dates) - fetched_active_market_dates)

    return backfill_active_market_dates
    
def _wait_before_click(driver: webdriver.chrome.webdriver.WebDriver, tag: str, attr_name: str, attr_val: str) -> bool:
    """
    (Internal Helper) Procedures for the web scraping scripts when about to click a button

    Args:
        driver (webdriver.chrome.webdriver.WebDriver): Main object for accessing the web for scraping purposes
        tag (str): The name of the tag for the button about to be clicked
        attr_name (str): The name of the attribute for the button about to be clicked
        attr_val (str): The value of the attribute for the button about to be clicked
    
    Returns:
        bool: A Boolean of whether clicking the button was successful
    """
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

def _initialize_driver(download_dir: str) -> webdriver.chrome.webdriver.WebDriver:
    """
    (Internal Helper) Initalize the driver for accessing the web for scraping purposes
    
    Args:
        download_dir (str): The directory where the downloaded file will be stored instead of storing it directly to the Download directory
    
    Returns:
        webdriver.chrome.webdriver.WebDriver: The driver for accessing the web for scraping purposes
    """
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

def _check_if_download_completion(download_dir: str, weekday_dt: str, timeout: int = 10) -> None:
    """
    (Internal Helper) A timeout procedure during the process of downloading the data

    Args:
        download_dir (str): The directory where the downloaded data is being stored
        weekday_date (str): The date that will be incorporated inside the name of the file to be checked
        timeout (int): The duration of the download timeout
    """
    filename = f"Stock Summary-{weekday_dt.replace('-', '')}.xlsx"
    end_time = time.time() + timeout

    while time.time() < end_time:
        full_path = os.path.join(download_dir, filename)
        
        if os.path.isfile(full_path):
            return
        else:
            time.sleep(1)
    
    return

def _wait_for_page_stability(driver: webdriver.chrome.webdriver.WebDriver, timeout: int = 30, check_interval: float = 0.5) -> bool:
    """
    (Internal Helper) A procedure for waiting until the page is being loaded completely

    Args:
        driver (webdriver.chrome.webdriver.WebDriver): The driver for accessing the web for scraping purposes
        timeout (int): The timeout duration for waiting until the page is fully loaded
        check_interval (float): The interval of checking the loading status of the page
    
    Returns:
        bool: A boolean of whether the page is finished being loaded under the given duration
    """
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

def _select_year_month_on_web(driver: webdriver.chrome.webdriver.WebDriver, year: str, month: str) -> None:
    """
    (Internal Helper) A scraping procedure for selecting the year and month on the web

    Args:
        driver (webdriver.chrome.webdriver.WebDriver): The driver for accessing the web for scraping purposes
        year (str): The year to be selected on the web
        month (str): The month to be selected on the web
    """
    # div for changing the date
    _ = _wait_before_click(driver, 'div', 'class', 'mx-input-wrapper')
    
    # button for changing the year
    _ = _wait_before_click(driver, 'button', 'class', 'mx-btn mx-btn-text mx-btn-current-year')
    
    # button for selecting the year
    _ = _wait_before_click(driver, 'td', 'data-year', year)
    
    # button for selecting the month
    _ = _wait_before_click(driver, 'td', 'data-month', month-1)

    return

def _select_and_download_specific_date_on_web(driver: webdriver.chrome.webdriver.WebDriver, download_dir: str, weekday_dt: str) -> bool:
    """
    (Internal Helper) A scraping procedure for selecting the specific date of data to be downloaded

    Args:
        driver (webdriver.chrome.webdriver.WebDriver): The driver for accessing the web for scraping purposes
        donwload_dir (str): The directory where the downloaded data will be stored
        weekday_dt (str): The date of the data that will be downloaded

    Returns:
        bool: A boolean of whether the file is successfully downloaded
    """
    # div for changing the date
    _ = _wait_before_click(driver, 'div', 'class', 'mx-input-wrapper')
    
    # button for selecting the day
    _ = _wait_before_click(driver, 'td', 'title', weekday_dt)
    
    # wait before everything finished being loaded
    _ = _wait_for_page_stability(driver)

    # button for downloading the day
    check_data_bool = _wait_before_click(driver, 'button', 'class', 'btn-filter-input mb-8 btn-download text-center')

    if check_data_bool:
        _check_if_download_completion(download_dir, weekday_dt)

        return True
    
    return False

def _parse_indo_date(date_str: str) -> str:
    """
    (Internal Helper) Parse date written in Indonesia into a proper datetime format

    Args:
        date_str (str): A date that wants to be parsed
    
    Returns:
        str: A parsed date in a proper datetime format
    """
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

def _clean_downloaded_data(download_dir: str, weekday_dt: str) -> None:
    """
    (Internal Helper) Clean the format and content of the downloaded data

    Args:
        download_dir (str): The directory where the data is being stored
        weekday_dt (str): The date that is incorporated inside the file name, specifying which file to process
    """
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