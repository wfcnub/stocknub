from datetime import datetime, timedelta

def _get_yesterday_date():
    """
    Get yesterday's date in YYYY-MM-DD format.

    Returns:s
        str: Yesterday's date in 'YYYY-MM-DD' format
    """
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")