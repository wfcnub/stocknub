from datetime import datetime, timedelta

from utils.data_fetcher import download_stock_data
from utils.io import append_df_to_csv, get_last_date_from_csv

def fetch_emiten_data(args_tuple):
    """
    Fetch data for a single emiten.

    Args:
        args_tuple: Tuple containing (emiten, start_date, end_date, csv_folder_path, update_mode)

    Returns:
        Tuple of (emiten, success, message)
    """
    emiten, start_date, end_date, csv_folder_path, update_mode = args_tuple

    try:
        csv_file_path = f"{csv_folder_path}/{emiten}.csv"

        if update_mode:
            last_date = get_last_date_from_csv(csv_file_path)
            if last_date:
                last_date_dt = datetime.strptime(last_date, "%Y-%m-%d")
                next_date = last_date_dt + timedelta(days=1)
                start_date = next_date.strftime("%Y-%m-%d")

        df = download_stock_data(emiten, start_date=start_date, end_date=end_date)

        if df is not None and not df.empty:
            append_df_to_csv(df, csv_file_path)
            date_range = f"from {start_date or 'earliest'} to {end_date or 'today'}"
            return (
                emiten,
                True,
                f"Data for {emiten} {date_range} saved to {csv_file_path}.",
            )
        
        return (emiten, False, f"No data found for {emiten}.")
            
    except Exception as e:
        return (emiten, False, f"Error fetching {emiten}: {str(e)}")