from app.repositories.recommendation import RecommendationRepository
from app.schemas.recommendation import DailyRecommendationResponse, RecommendationItem
import math

class RecommendationService:
    """
    Handles business logic and data transformations.
    """
    def __init__(self, repository: RecommendationRepository):
        self.repository = repository

    def get_daily_recommendations(self, rolling_window: str) -> DailyRecommendationResponse:
        df, score_date = self.repository.get_daily_recommendations(rolling_window)
        
        df = df.reset_index()
        records = df.to_dict('records')
        
        score_col = f'Score {rolling_window}'
        recommendations = []
        
        def clean_float(val):
            try:
                val = float(val)
                return 0.0 if math.isnan(val) else val
            except (ValueError, TypeError):
                return 0.0

        for row in records:
            recommendations.append(RecommendationItem(
                ticker=str(row.get('Ticker', '')),
                score=clean_float(row.get(score_col, 0.0)),
                target_buy_price=clean_float(row.get('Target Buy Price', 0.0)),
                target_sell_price=clean_float(row.get('Target Sell Price', 0.0))
            ))
            
        return DailyRecommendationResponse(
            score_date=str(score_date),
            rolling_window=rolling_window,
            recommendations=recommendations
        )
