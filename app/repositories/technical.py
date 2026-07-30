import pandas as pd
from pathlib import Path

class TechnicalRepository:
    """
    Repository for fetching technical indicators from CSV.
    """
    def get_technical_indicators(self, ticker: str) -> pd.DataFrame:
        file_path = Path(__file__).resolve().parent.parent.parent / "data" / "stock" / "technical" / f"{ticker}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Data for ticker {ticker} not found.")
        return pd.read_csv(file_path)
