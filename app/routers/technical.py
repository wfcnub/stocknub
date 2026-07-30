from fastapi import APIRouter, Depends
from app.controllers.technical import TechnicalController
from app.services.technical import TechnicalService
from app.repositories.technical import TechnicalRepository
from typing import Dict, Any

router = APIRouter(prefix="/technical", tags=["technical"])

def get_technical_repository() -> TechnicalRepository:
    return TechnicalRepository()

def get_technical_service(repo: TechnicalRepository = Depends(get_technical_repository)) -> TechnicalService:
    return TechnicalService(repo)

def get_technical_controller(service: TechnicalService = Depends(get_technical_service)) -> TechnicalController:
    return TechnicalController(service)

@router.get("/check-availability", response_model=Dict[str, Any])
def check_technical_availability(
    ticker: str,
    controller: TechnicalController = Depends(get_technical_controller)
):
    """
    Check if technical indicators for a given ticker are available for the lagged date (today - 1 day).
    """
    return controller.check_availability(ticker.upper())

@router.get("/", response_model=Dict[str, Any])
def get_technical_indicators(
    ticker: str,
    controller: TechnicalController = Depends(get_technical_controller)
):
    """
    Get the technical indicators for a given ticker, lagged by 1 day.
    The output is the lagged date as the key, mapped to its indicators.
    """
    return controller.get_technical_indicators(ticker.upper())
