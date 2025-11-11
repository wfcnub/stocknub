from datetime import datetime, timedelta

def _get_yesterday_date():
    """
    (Internal Helper) Get yesterday's date in YYYY-MM-DD format.

    Returns:
        str: Yesterday's date in 'YYYY-MM-DD' format
    """
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")