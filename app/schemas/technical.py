"""Version 2.1 contract; dates index every OHLCV and indicator series."""
from datetime import date
from typing import Annotated, Literal

from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, StrictFloat, StrictInt, ValidationInfo,
    field_validator, model_validator,
)

ForecastType = Literal["5dd", "10dd"]
Ticker = Annotated[str, Field(pattern=r"^[A-Za-z]{4}$")]


def validate_iso_date(value: str) -> str:
    date.fromisoformat(value)
    return value


ISODate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$"), AfterValidator(validate_iso_date)]
Count = Annotated[int, Field(strict=True, ge=0)]
Number = StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)] | None
Series = Annotated[list[Number], Field(min_length=3, max_length=3)]
Volume = Annotated[int, Field(strict=True, ge=0)] | None
VolumeSeries = Annotated[list[Volume], Field(min_length=3, max_length=3)]


def metadata_sources(forecast_type: ForecastType) -> tuple[tuple[str, str], ...]:
    """The ten-session horizon intentionally includes five-session models."""
    if forecast_type not in ("5dd", "10dd"):
        raise ValueError("Unsupported forecast type")
    windows = ("5dd",) if forecast_type == "5dd" else ("5dd", "10dd")
    return tuple((label, window) for window in windows
                 for label in ("median_loss", "median_gain"))


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OHLCVSeries(ContractModel):
    """Historical observations aligned with dates, oldest to newest; null is unavailable."""

    open: Series
    high: Series
    low: Series
    close: Series
    volume: VolumeSeries = Field(description="Nonnegative integer or null, aligned with dates; zero is an observation.")

    @model_validator(mode="after")
    def validate_bars(self):
        for index in range(3):
            prices = [self.open[index], self.high[index], self.low[index], self.close[index]]
            if any(value is not None and value < 0 for value in prices):
                raise ValueError("Prices must be nonnegative")
            high, low = self.high[index], self.low[index]
            if high is not None and low is not None and high < low:
                raise ValueError("High must be at least low")
            for value in (self.open[index], self.close[index]):
                if value is not None and ((high is not None and value > high)
                                          or (low is not None and value < low)):
                    raise ValueError("Open/close must lie within observed bounds")
        return self


class OHLCVMetadata(ContractModel):
    """Unknown units/basis must not be inferred from ticker names or future downloads."""

    source: Literal["technical_csv"]
    price_currency: Literal["IDR", "unknown"]
    volume_unit: Literal["shares", "unknown"]
    price_adjustment: Literal["unknown", "adjusted", "unadjusted"]


class FeatureSource(ContractModel):
    label_type: Literal["medianLoss", "medianGain"]
    window: ForecastType


class FeatureFilter(ContractModel):
    source: Literal["model_metadata"] = "model_metadata"
    key: Literal["selected_features"] = "selected_features"
    strategy: Literal["union"] = "union"
    model_version: Literal[1] = 1
    sources: list[FeatureSource]
    selected_union_count: Count
    included_count: Annotated[int, Field(strict=True, ge=1)]
    selected_unavailable_count: Count
    excluded_not_selected_count: Count

    @model_validator(mode="after")
    def validate_counts(self):
        if self.selected_union_count != self.included_count + self.selected_unavailable_count:
            raise ValueError("Selected union counts disagree")
        return self


class TechnicalQuality(ContractModel):
    request_date: ISODate
    data_age_calendar_days: Annotated[int, Field(strict=True, ge=1)]
    freshness_status: Literal["unverified_without_exchange_calendar"] = "unverified_without_exchange_calendar"
    null_counts_by_session: Annotated[list[Count], Field(min_length=3, max_length=3,
        description="Counts unavailable indicators only, aligned with dates.")]
    ohlcv_null_counts_by_session: Annotated[
        list[Annotated[int, Field(strict=True, ge=0, le=5)]], Field(min_length=3, max_length=3,
            description="Counts unavailable OHLCV observations only, aligned with dates; not a freshness guarantee.")
    ]


class TechnicalResponse(ContractModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "schema_version": "2.1", "ticker": "AALI", "forecast_type": "5dd",
        "market_timezone": "Asia/Jakarta",
        "selection_policy": "latest_available_before_request_date",
        "requested_sessions": 3, "returned_sessions": 3,
        "dates": ["2026-09-22", "2026-09-23", "2026-09-24"],
        "order": "oldest_to_newest", "data_as_of": "2026-09-24",
        "quality": {"request_date": "2026-10-10", "data_age_calendar_days": 16,
                    "freshness_status": "unverified_without_exchange_calendar",
                    "null_counts_by_session": [0, 0, 1],
                    "ohlcv_null_counts_by_session": [0, 0, 1]},
        "ohlcv_metadata": {"source": "technical_csv", "price_currency": "unknown",
                           "volume_unit": "unknown", "price_adjustment": "unknown"},
        "ohlcv": {"open": [7000, 7050, 7100], "high": [7100, 7150, 7200],
                  "low": [6950, 7000, 7050], "close": [7050, 7100, 7150],
                  "volume": [1200000, 1300000, None]},
        "feature_filter": {
            "source": "model_metadata", "key": "selected_features", "strategy": "union",
            "model_version": 1,
            "sources": [{"label_type": "medianLoss", "window": "5dd"},
                        {"label_type": "medianGain", "window": "5dd"}],
            "selected_union_count": 3, "included_count": 2,
            "selected_unavailable_count": 1, "excluded_not_selected_count": 5,
        },
        "feature_catalog_version": "1.0",
        "indicators": {"RSI Value": [42.7, 42.7, None], "RSI Up Trend": [0, 0, 0]},
    }]})

    schema_version: Literal["2.1"] = "2.1"
    ticker: Ticker
    forecast_type: ForecastType
    market_timezone: Literal["Asia/Jakarta"] = "Asia/Jakarta"
    selection_policy: Literal["latest_available_before_request_date"] = "latest_available_before_request_date"
    requested_sessions: Literal[3] = 3
    returned_sessions: Literal[3] = 3
    dates: Annotated[list[ISODate], Field(min_length=3, max_length=3,
        description="Shared session axis: dates[i], every OHLCV series[i], and every indicator[i] describe the same historical session. Forecast horizon does not change history length.")]
    order: Literal["oldest_to_newest"] = "oldest_to_newest"
    data_as_of: ISODate
    quality: TechnicalQuality
    ohlcv: OHLCVSeries
    ohlcv_metadata: OHLCVMetadata
    feature_filter: FeatureFilter
    feature_catalog_version: Literal["1.0"] = "1.0"
    indicators: dict[str, Series]

    @field_validator("indicators")
    @classmethod
    def validate_names(cls, value):
        if not value or any(not name or name != name.strip() or name == "Date" for name in value):
            raise ValueError("Indicator names must be nonempty exact CSV headers")
        return value

    @model_validator(mode="after")
    def validate_alignment(self, info: ValidationInfo):
        parsed = [date.fromisoformat(value) for value in self.dates]
        if any(left >= right for left, right in zip(parsed, parsed[1:])):
            raise ValueError("Dates must be unique and strictly increasing")
        if self.data_as_of != self.dates[-1] or self.returned_sessions != len(self.dates):
            raise ValueError("Session count/as-of must agree with dates")
        age = (date.fromisoformat(self.quality.request_date) - parsed[-1]).days
        if age < 1 or age != self.quality.data_age_calendar_days:
            raise ValueError("Age/cutoff must agree with the market request date")
        if self.feature_filter.included_count != len(self.indicators):
            raise ValueError("Included count must agree with indicator count")
        expected_sources = [("medianLoss" if label == "median_loss" else "medianGain", window)
                            for label, window in metadata_sources(self.forecast_type)]
        if [(source.label_type, source.window) for source in self.feature_filter.sources] != expected_sources:
            raise ValueError("Metadata sources must agree with forecast type")
        null_counts = [sum(values[index] is None for values in self.indicators.values())
                       for index in range(3)]
        if self.quality.null_counts_by_session != null_counts:
            raise ValueError("Null counts must agree with filtered observations")
        ohlcv_counts = [sum(values[index] is None for values in self.ohlcv.model_dump().values())
                       for index in range(3)]
        if self.quality.ohlcv_null_counts_by_session != ohlcv_counts:
            raise ValueError("OHLCV null counts must agree with observations")
        # Source membership is validated while constructing the response. The
        # source vectors stay out of ordinary agent context.
        if info.context:
            selected, columns = info.context["selected_features"], info.context["csv_columns"]
            if set(self.indicators) != (selected & set(columns)):
                raise ValueError("Indicators must equal the metadata/CSV intersection")
            if self.feature_filter.selected_union_count != len(selected):
                raise ValueError("Union count must agree with source metadata")
            if len(columns) != self.feature_filter.included_count + self.feature_filter.excluded_not_selected_count:
                raise ValueError("CSV column counts disagree")
        return self


class TechnicalErrorDetail(ContractModel):
    code: str
    message: str
    ticker: Ticker
    forecast_type: ForecastType


class TechnicalErrorResponse(ContractModel):
    detail: TechnicalErrorDetail


class TechnicalAvailability(ContractModel):
    ticker: Ticker
    forecast_type: ForecastType
    available: bool
    reason_code: str
    failure_status_code: int | None = None
    data_as_of: ISODate | None = None
    quality: TechnicalQuality | None = None

    @model_validator(mode="after")
    def validate_readiness(self):
        if self.available:
            if self.reason_code != "ready" or self.failure_status_code is not None or self.data_as_of is None or self.quality is None:
                raise ValueError("Ready responses require quality and as-of information")
        elif self.reason_code == "ready" or self.failure_status_code is None:
            raise ValueError("Unavailable responses require a failure reason")
        return self
