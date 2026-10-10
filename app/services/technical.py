"""Selected model features over exactly three observed historical sessions."""
import logging
import math
import re
from datetime import date, datetime
from numbers import Integral, Real
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from app.errors.technical import (
    InsufficientSessions, InvalidModelMetadata, InvalidTechnicalData,
    NoEligibleIndicators, NoEligibleSessions, TechnicalError,
)
from app.repositories.technical import TechnicalRepository
from app.schemas.technical import (
    ForecastType, TechnicalAvailability, TechnicalResponse, metadata_sources,
)

logger = logging.getLogger(__name__)
MARKET_TIMEZONE = ZoneInfo("Asia/Jakarta")


def market_clock() -> datetime:
    return datetime.now(MARKET_TIMEZONE)


class TechnicalService:
    def __init__(self, repository: TechnicalRepository,
                 clock: Callable[[], datetime] = market_clock):
        self.repository = repository
        self.clock = clock

    @staticmethod
    def _validate_ticker(ticker: str) -> str:
        if not re.fullmatch(r"[A-Za-z]{4}", ticker):
            raise ValueError("Ticker must be four ASCII letters")
        return ticker.upper()

    @staticmethod
    def _selected_features(metadata) -> set[str]:
        features = metadata.get("selected_features") if isinstance(metadata, dict) else None
        if (not isinstance(features, list) or not features
                or any(not isinstance(name, str) or not name or name != name.strip() for name in features)
                or len(features) != len(set(features))):
            raise InvalidModelMetadata()
        return set(features)

    @staticmethod
    def _validate_technical(df: pd.DataFrame) -> pd.DataFrame:
        headers = list(df.columns)
        if (df.empty or "Date" not in headers or len(headers) != len(set(headers))
                or any(not isinstance(name, str) or not name or name != name.strip() for name in headers)):
            raise InvalidTechnicalData()
        validated = df.copy()
        try:
            # The pipeline publishes date-only ISO strings. Reject timestamps,
            # ambiguous dates, nulls and repeated normalized session dates.
            raw_dates = validated["Date"].tolist()
            if any(not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
                   for value in raw_dates):
                raise ValueError("Invalid ISO session dates")
            validated["Date"] = [date.fromisoformat(value) for value in raw_dates]
            if validated["Date"].duplicated().any():
                raise ValueError("Duplicate session dates")
            for name in headers:
                if name == "Date":
                    continue
                # Validate all source columns, including those outside the union.
                if pd.api.types.is_bool_dtype(validated[name]):
                    raise ValueError("Boolean text is not a numeric observation")
                validated[name] = pd.to_numeric(validated[name], errors="raise")
        except (ValueError, TypeError, OverflowError) as exc:
            raise InvalidTechnicalData() from exc
        return validated.sort_values("Date")

    @staticmethod
    def _json_number(value):
        if pd.isna(value):
            return None
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            return float(value) if math.isfinite(value) else None
        raise InvalidTechnicalData()

    def get_technical_indicators(self, ticker: str, forecast_type: ForecastType) -> TechnicalResponse:
        ticker = self._validate_ticker(ticker)
        sources = metadata_sources(forecast_type)
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Technical service clock must be timezone-aware")
        request_date = now.astimezone(MARKET_TIMEZONE).date()
        try:
            validated = self._validate_technical(self.repository.get_technical_indicators(ticker))
            eligible = validated.loc[validated["Date"] < request_date]
            if eligible.empty:
                raise NoEligibleSessions()
            if len(eligible) < 3:
                raise InsufficientSessions()
            window = eligible.tail(3)
            selected = set()
            with self.repository.metadata_snapshot(ticker, sources):
                for label, source_window in sources:
                    source_features = self._selected_features(
                        self.repository.get_model_metadata(ticker, label, source_window))
                    logger.debug("technical_selection ticker=%s label=%s window=%s selected=%s",
                                 ticker, label, source_window, sorted(source_features))
                    selected.update(source_features)
            csv_columns = [name for name in validated.columns if name != "Date"]
            included = [name for name in csv_columns if name in selected]
            if not included:
                raise NoEligibleIndicators()
            indicators = {name: [self._json_number(value) for value in window[name].tolist()]
                          for name in included}
            dates = [value.isoformat() for value in window["Date"]]
            null_counts = [sum(series[index] is None for series in indicators.values()) for index in range(3)]
            unavailable = sorted(selected - set(csv_columns))
            excluded = [name for name in csv_columns if name not in selected]
            response = TechnicalResponse.model_validate({
                "ticker": ticker, "forecast_type": forecast_type,
                "dates": dates, "data_as_of": dates[-1],
                "quality": {
                    "request_date": request_date.isoformat(),
                    "data_age_calendar_days": (request_date - window["Date"].iloc[-1]).days,
                    "null_counts_by_session": null_counts,
                },
                "feature_filter": {
                    "sources": [{"label_type": "medianLoss" if label == "median_loss" else "medianGain",
                                 "window": source_window} for label, source_window in sources],
                    "selected_union_count": len(selected), "included_count": len(included),
                    "selected_unavailable_count": len(unavailable),
                    "excluded_not_selected_count": len(excluded),
                },
                "indicators": indicators,
            }, context={"selected_features": selected, "csv_columns": csv_columns})
            logger.info("technical_result ticker=%s forecast_type=%s dates=%s union=%s included=%s "
                        "unavailable=%s excluded=%s null_counts=%s age_days=%s outcome=ready",
                        ticker, forecast_type, dates, len(selected), len(included), unavailable,
                        excluded, null_counts, response.quality.data_age_calendar_days)
            return response
        except TechnicalError as exc:
            logger.warning("technical_result ticker=%s forecast_type=%s outcome=%s",
                           ticker, forecast_type, exc.code)
            raise

    def check_availability(self, ticker: str, forecast_type: ForecastType) -> TechnicalAvailability:
        ticker = self._validate_ticker(ticker)
        try:
            response = self.get_technical_indicators(ticker, forecast_type)
        except TechnicalError as exc:
            return TechnicalAvailability(ticker=ticker, forecast_type=forecast_type, available=False,
                                         reason_code=exc.code, failure_status_code=exc.status_code)
        return TechnicalAvailability(ticker=ticker, forecast_type=forecast_type, available=True,
                                     reason_code="ready", data_as_of=response.data_as_of,
                                     quality=response.quality)
