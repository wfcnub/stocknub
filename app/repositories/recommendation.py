import pandas as pd
from typing import Tuple
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from analyticsHub.main import get_daily_recommendations

class RecommendationRepository:
    """
    Acts as a repository to fetch data from the analyticsHub logic.
    """
    def get_daily_recommendations(self, rolling_window: str) -> Tuple[pd.DataFrame, str]:
        return get_daily_recommendations(rolling_window)
