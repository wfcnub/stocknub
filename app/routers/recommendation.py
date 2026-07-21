from fastapi import APIRouter, Depends
from app.controllers.recommendation import RecommendationController
from app.services.recommendation import RecommendationService
from app.repositories.recommendation import RecommendationRepository
from app.schemas.recommendation import DailyRecommendationResponse

router = APIRouter(prefix="/analytics", tags=["analytics"])

def get_recommendation_repository() -> RecommendationRepository:
    return RecommendationRepository()

def get_recommendation_service(repo: RecommendationRepository = Depends(get_recommendation_repository)) -> RecommendationService:
    return RecommendationService(repo)

def get_recommendation_controller(service: RecommendationService = Depends(get_recommendation_service)) -> RecommendationController:
    return RecommendationController(service)

@router.get("/daily_recommendations", response_model=DailyRecommendationResponse)
def get_daily_recommendations(
    rolling_window: str = "10dd",
    controller: RecommendationController = Depends(get_recommendation_controller)
):
    """
    Get the daily recommendations for a given rolling window (e.g., '10dd').
    """
    return controller.get_daily_recommendations(rolling_window)
