from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for the model-development ticker universe."""

    top_n: int = 150
    min_history_rows: int = 504
    max_stale_calendar_days: int = 7
    min_traded_days_252: int = 220
    max_zero_volume_ratio_252: float = 0.20
    max_unchanged_close_ratio_60: float = 0.20
    max_invalid_bar_ratio: float = 0.01
    min_median_value_traded_60: float = 5_000_000_000
    min_auxiliary_overlap_ratio: float = 0.80
    require_auxiliary_data: bool = True

    min_fundamental_coverage: float = 0.50
    min_fundamental_score: float = 0.20
    min_technical_score: float = 0.20
    min_fetch_coverage: float = 0.80
    fundamental_weight: float = 0.60
    technical_weight: float = 0.40
    missing_fundamental_penalty: float = 0.20

    max_sector_share: float = 0.25
    min_sector_count: int = 2
    incumbent_bonus: float = 0.02

    fundamental_cache_ttl_days: int = 30
    fundamental_fetch_workers: int = 4

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.min_history_rows <= 0:
            raise ValueError("min_history_rows must be positive")
        if self.min_traded_days_252 < 0 or self.min_traded_days_252 > 252:
            raise ValueError("min_traded_days_252 must be between 0 and 252")

        probability_fields = (
            "max_zero_volume_ratio_252",
            "max_unchanged_close_ratio_60",
            "max_invalid_bar_ratio",
            "min_auxiliary_overlap_ratio",
            "min_fundamental_coverage",
            "min_fundamental_score",
            "min_technical_score",
            "min_fetch_coverage",
            "fundamental_weight",
            "technical_weight",
            "max_sector_share",
        )
        for field_name in probability_fields:
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")

        weight_sum = self.fundamental_weight + self.technical_weight
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError("fundamental_weight and technical_weight must sum to 1")
