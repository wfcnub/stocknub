import logging

from fastapi import HTTPException

from app.errors.technical import TechnicalError
from app.schemas.technical import ForecastType, TechnicalAvailability, TechnicalResponse
from app.services.technical import TechnicalService

logger = logging.getLogger(__name__)


class TechnicalController:
    def __init__(self, service: TechnicalService):
        self.service = service

    def _call(self, method, ticker, forecast_type):
        try:
            return method(ticker, forecast_type)
        except TechnicalError as exc:
            raise HTTPException(status_code=exc.status_code, detail={
                "code": exc.code, "message": exc.message,
                "ticker": ticker, "forecast_type": forecast_type,
            }) from exc
        except Exception as exc:
            logger.exception("Unexpected technical API failure ticker=%s forecast_type=%s", ticker, forecast_type)
            raise HTTPException(status_code=500, detail={
                "code": "internal_error", "message": "Technical indicators could not be loaded.",
                "ticker": ticker, "forecast_type": forecast_type,
            }) from exc

    def get_technical_indicators(self, ticker: str, forecast_type: ForecastType) -> TechnicalResponse:
        return self._call(self.service.get_technical_indicators, ticker, forecast_type)

    def check_availability(self, ticker: str, forecast_type: ForecastType) -> TechnicalAvailability:
        return self._call(self.service.check_availability, ticker, forecast_type)
