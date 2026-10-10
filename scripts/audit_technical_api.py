"""Read-only artifact/readiness audit and equal-feature wire-format comparison.

Run from the repository root: python -m scripts.audit_technical_api
"""
import argparse
import json
import logging
from collections import Counter
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from app.errors.technical import TechnicalError
from app.repositories.technical import TechnicalRepository
from app.schemas.technical import TechnicalResponse, metadata_sources
from app.services.technical import TechnicalService
from utils import paths


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def compare_formats(response: TechnicalResponse) -> dict:
    """All formats include identical selected features, values, and date axes."""
    shared = {"dates": response.dates, "indicators": response.indicators}
    rows = [{"Date": day, **{name: values[index] for name, values in response.indicators.items()}}
            for index, day in enumerate(response.dates)]
    table = ["| Indicator | " + " | ".join(response.dates) + " |",
             "| --- | --- | --- | --- |"]
    for name, values in response.indicators.items():
        escaped = name.replace("|", "\\|").replace("\n", " ")
        table.append("| " + escaped + " | " + " | ".join(compact_json(value) for value in values) + " |")
    series_bytes = len(compact_json(shared).encode("utf-8"))
    row_bytes = len(compact_json(rows).encode("utf-8"))
    return {
        "shared_date_series_bytes": series_bytes,
        "chronological_row_objects_bytes": row_bytes,
        "markdown_table_bytes": len("\n".join(table).encode("utf-8")),
        "full_response_bytes": len(compact_json(response.model_dump()).encode("utf-8")),
        "series_reduction_vs_rows_percent": round(100 * (1 - series_bytes / row_bytes), 2),
    }


def audit_ticker(service, ticker, forecast_type) -> dict:
    coverage = [{"label_type": label, "window": window,
                 "exists": paths.get_model_artifact_path(1, label, ticker, window, "metadata", "json").is_file()}
                for label, window in metadata_sources(forecast_type)]
    try:
        response = service.get_technical_indicators(ticker, forecast_type)
    except TechnicalError as exc:
        return {"ticker": ticker, "forecast_type": forecast_type, "available": False,
                "reason_code": exc.code, "failure_status_code": exc.status_code, "sources": coverage}
    return {"ticker": ticker, "forecast_type": forecast_type, "available": True,
            "reason_code": "ready", "dates": response.dates, "quality": response.quality.model_dump(),
            "feature_filter": response.feature_filter.model_dump(), "sources": coverage,
            "format_comparison": compare_formats(response)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", action="append", help="Repeat for selected tickers; default: all served CSV tickers")
    parser.add_argument("--forecast-type", choices=("5dd", "10dd"), action="append", help="Default: both horizons")
    parser.add_argument("--request-date", type=date.fromisoformat,
                        help="Fixed Jakarta cutoff for reproducible audits (not historical backtesting)")
    args = parser.parse_args()
    tickers = sorted(set(ticker.upper() for ticker in args.ticker)) if args.ticker else sorted(
        path.stem for path in paths.get_technical_dir().glob("*.csv"))
    forecasts = list(dict.fromkeys(args.forecast_type or ("5dd", "10dd")))
    kwargs = {}
    if args.request_date:
        instant = datetime.combine(args.request_date, time(), tzinfo=ZoneInfo("Asia/Jakarta"))
        kwargs["clock"] = lambda: instant
    logging.basicConfig(level=logging.ERROR)
    service = TechnicalService(TechnicalRepository(), **kwargs)
    records = [audit_ticker(service, ticker, forecast) for ticker in tickers for forecast in forecasts]
    counts = Counter(record["reason_code"] for record in records)
    print(compact_json({"environment": paths.env, "tickers": len(tickers), "requests": len(records),
                        "outcomes": dict(counts), "records": records}))


if __name__ == "__main__":
    main()
