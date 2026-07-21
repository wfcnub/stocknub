from pydantic import BaseModel
from typing import List

class RecommendationItem(BaseModel):
    ticker: str
    score: float
    target_buy_price: float
    target_sell_price: float

class DailyRecommendationResponse(BaseModel):
    score_date: str
    rolling_window: str
    recommendations: List[RecommendationItem]
