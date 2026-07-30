import pandas as pd
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from analyticsHub.main import get_daily_recommendations

class RecommendationRepository:
    """
    Acts as a repository to fetch data from the analyticsHub logic.
    """
    def get_daily_recommendations(self, rolling_window: str) -> Tuple[pd.DataFrame, str]:
        return get_daily_recommendations(rolling_window)
