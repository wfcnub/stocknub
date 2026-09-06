import pandas as pd
from datetime import date, timedelta
from app.repositories.technical import TechnicalRepository

class TechnicalService:
    """
    Service layer for technical indicator logic.
    """
    def __init__(self, repository: TechnicalRepository):
        self.repository = repository
    
    def _get_latest_market_date(self):
        if date.today().weekday() == 0:
            selected_date = (date.today() - timedelta(days=3)).strftime('%Y-%m-%d')
        elif 1 <= date.today().weekday() <= 5:
            selected_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        elif date.today().weekday() == 6:
            selected_date = (date.today() - timedelta(days=2)).strftime('%Y-%m-%d')

        return selected_date
        
    def get_lagged_technical_indicators(self, ticker: str) -> dict:
        df = self.repository.get_technical_indicators(ticker)

        selected_date = self._get_latest_market_date()

        if selected_date not in df['Date'].values:
            raise ValueError("Latest date not found in data")
            
        target_row = df.loc[df['Date'] == selected_date, :].squeeze()
        
        data_dict = target_row.drop('Date').to_dict()
        
        import math
        cleaned_dict = {}
        for k, v in data_dict.items():
            if isinstance(v, float) and math.isnan(v):
                cleaned_dict[k] = None
            else:
                cleaned_dict[k] = v
                
        return {
            selected_date: cleaned_dict
        }

    def check_lagged_availability(self, ticker: str) -> bool:
        df = self.repository.get_technical_indicators(ticker)
        selected_date = self._get_latest_market_date()
        
        return selected_date in df['Date'].values
