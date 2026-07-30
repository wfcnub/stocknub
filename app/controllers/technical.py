from fastapi import HTTPException
from app.services.technical import TechnicalService
from typing import Dict, Any

class TechnicalController:
    """
    Controller layer for technical indicators.
    """
    def __init__(self, service: TechnicalService):
        self.service = service
        
    def get_technical_indicators(self, ticker: str) -> Dict[str, Any]:
        try:
            return self.service.get_lagged_technical_indicators(ticker)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            raise HTTPException(status_code=500, detail=f"Failed to fetch technical indicators. Error: {str(e)}")

    def check_availability(self, ticker: str) -> Dict[str, Any]:
        try:
            is_available = self.service.check_lagged_availability(ticker)
            return {"ticker": ticker, "available": is_available}
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            raise HTTPException(status_code=500, detail=f"Failed to check availability. Error: {str(e)}")
