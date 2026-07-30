import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.append(project_root)

from utils import paths

class TechnicalRepository:
    """
    Repository for fetching technical indicators from CSV.
    """
    def get_technical_indicators(self, ticker: str) -> pd.DataFrame:
        file_path = paths.get_technical_path(ticker)
        if not file_path.exists():
            raise FileNotFoundError(f"Data for ticker {ticker} not found.")
        return pd.read_csv(file_path)
