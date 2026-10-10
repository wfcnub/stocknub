from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.controllers.technical import TechnicalController
from app.repositories.technical import TechnicalRepository
from app.schemas.technical import (
    ForecastType, TechnicalAvailability, TechnicalErrorResponse, TechnicalResponse,
)
from app.services.technical import TechnicalService

router = APIRouter(prefix="/technical", tags=["technical"])
TickerQuery = Annotated[str, Query(pattern=r"^[A-Za-z]{4}$", description="IDX ticker: four ASCII letters; normalized to uppercase.", examples=["AALI", "BBCA"])]
ForecastQuery = Annotated[ForecastType, Query(description="5dd: 5-trading-session forecast horizon; 10dd: 10-trading-session horizon. Both return three historical sessions. 10dd includes both 5dd and 10dd model feature unions.", examples=["5dd", "10dd"])]


def get_technical_repository() -> TechnicalRepository:
    return TechnicalRepository()


def get_technical_service(repo: TechnicalRepository = Depends(get_technical_repository)) -> TechnicalService:
    return TechnicalService(repo)


def get_technical_controller(service: TechnicalService = Depends(get_technical_service)) -> TechnicalController:
    return TechnicalController(service)


@router.get("/check-availability", response_model=TechnicalAvailability,
            responses={500: {"model": TechnicalErrorResponse}})
def check_technical_availability(
    ticker: TickerQuery,
    forecast_type: ForecastQuery,
    controller: TechnicalController = Depends(get_technical_controller),
) -> TechnicalAvailability:
    """Check the same window, metadata completeness, and feature eligibility as retrieval.

    Domain failures return available=false with reason_code and failure_status_code.
    Readiness does not assert freshness; successful checks include source age.
    """
    return controller.check_availability(ticker.upper(), forecast_type)


@router.get("/", response_model=TechnicalResponse,
            responses={code: {"model": TechnicalErrorResponse} for code in (404, 409, 500)})
def get_technical_indicators(
    ticker: TickerQuery,
    forecast_type: ForecastQuery,
    controller: TechnicalController = Depends(get_technical_controller),
) -> TechnicalResponse:
    """Return selected indicators over the latest three available sessions before today in Jakarta.

    Dates and arrays run oldest to newest. Required model-v1 metadata is authoritative:
    5dd uses loss/gain 5dd files; 10dd also uses loss/gain 10dd files. Only exact CSV
    matches are emitted. Missing selected training features are counted, never fabricated.
    Null is unavailable; zero is an observation. Consult feature catalog 1.0 for units
    and publication lags. This is a breaking change from the date-keyed row response.
    The OpenAPI response example uses illustrative counts and values, not live data.
    """
    return controller.get_technical_indicators(ticker.upper(), forecast_type)
