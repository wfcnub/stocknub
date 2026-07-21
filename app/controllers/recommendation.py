from fastapi import HTTPException
from app.services.recommendation import RecommendationService
from app.schemas.recommendation import DailyRecommendationResponse

class RecommendationController:
    """
    Handles HTTP layer for recommendations.
    """
    def __init__(self, service: RecommendationService):
        self.service = service

    def get_daily_recommendations(self, rolling_window: str) -> DailyRecommendationResponse:
        try:
            return self.service.get_daily_recommendations(rolling_window)
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            raise HTTPException(status_code=500, detail=f"Failed to fetch daily recommendations. Traceback: {trace}")
