import os
import sys
import pandas as pd
from typing import Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.append(project_root)
from analyticsHub.main import get_daily_recommendations

class RecommendationRepository:
    """
    Acts as a repository to fetch data from the analyticsHub logic.
    """
    def get_daily_recommendations(
        self,
        rolling_window: str,
        market_date: str,
    ) -> Tuple[pd.DataFrame, str]:
        return get_daily_recommendations(rolling_window, market_date)
