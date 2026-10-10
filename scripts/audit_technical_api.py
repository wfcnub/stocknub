"""Read-only artifact/readiness audit and equivalent schema-2.1 format comparison.

Run: python -m scripts.audit_technical_api --request-date 2026-10-10
Use --tokenizer ENCODING (repeatable) for actual tiktoken counts; otherwise bytes only.
No market data or model artifacts are regenerated.
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
from app.services.technical import OHLCV_HEADERS, TechnicalService, market_clock
from utils import paths


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def extraction_check(response, rows, table):
    """Deterministic consumer smoke check: dates, latest close/volume and all nulls.

    This is not an LLM/agent evaluation. Every candidate is reconstructed exactly.
    The labeled table is JSON, avoiding ambiguous Markdown escaping/types.
    """
    for index, day in enumerate(response.dates):
        assert rows[index]["date"] == day
        for group in ("ohlcv", "indicators"):
            expected = getattr(response, group)
            if group == "ohlcv":
                expected = expected.model_dump()
            assert rows[index][group] == {key: values[index] for key, values in expected.items()}
    assert table["dates"] == response.dates
    reconstructed = {"ohlcv": {}, "indicators": {}}
    for group, name, *values in table["rows"]:
        reconstructed[group][name] = values
    assert reconstructed == {"ohlcv": response.ohlcv.model_dump(), "indicators": response.indicators}
    assert rows[-1]["ohlcv"]["close"] == response.ohlcv.close[-1]
    assert rows[-1]["ohlcv"]["volume"] == response.ohlcv.volume[-1]
    return {"status": "passed", "latest_date": response.dates[-1],
            "latest_close": response.ohlcv.close[-1], "latest_volume": response.ohlcv.volume[-1],
            "ohlcv_null_counts_by_session": response.quality.ohlcv_null_counts_by_session,
            "agent_evaluation": "not_run_no_configured_agent_consumer"}


def compare_formats(response: TechnicalResponse, tokenizers=None) -> dict:
    """Include the same observations, units, quality and provenance in each candidate."""
    shared = {"dates": response.dates, "ohlcv": response.ohlcv.model_dump(), "indicators": response.indicators}
    rows = [{"date": day, **{group: {name: values[index] for name, values in shared[group].items()}
                            for group in ("ohlcv", "indicators")}}
            for index, day in enumerate(response.dates)]
    table = {"dates": response.dates, "columns": ["group", "name", *response.dates],
             "rows": [[group, name, *values] for group in ("ohlcv", "indicators")
                      for name, values in shared[group].items()]}
    common = response.model_dump(exclude={"dates", "ohlcv", "indicators"})
    data_sections = {"shared_date_series": shared, "chronological_row_objects": {"sessions": rows},
                     "labeled_table": {"table": table}}
    full = {name: {**common, **data} for name, data in data_sections.items()}
    baseline = response.model_dump(exclude={"ohlcv", "ohlcv_metadata"})
    baseline["schema_version"] = "2.0"
    baseline["quality"].pop("ohlcv_null_counts_by_session")
    result = {"measurement": "bytes_only" if not tokenizers else "bytes_and_tokens",
              "deterministic_extraction": extraction_check(response, rows, table), "candidates": {}}
    for name, data in data_sections.items():
        result["candidates"][name] = {
            "data_section_bytes": len(compact_json(data).encode("utf-8")),
            "full_response_bytes": len(compact_json(full[name]).encode("utf-8")),
        }
    series_bytes = result["candidates"]["shared_date_series"]["data_section_bytes"]
    row_bytes = result["candidates"]["chronological_row_objects"]["data_section_bytes"]
    full_bytes = result["candidates"]["shared_date_series"]["full_response_bytes"]
    baseline_bytes = len(compact_json(baseline).encode("utf-8"))
    result.update(shared_date_series_bytes=series_bytes, chronological_row_objects_bytes=row_bytes,
                  full_response_bytes=full_bytes, baseline_2_0_full_response_bytes=baseline_bytes,
                  added_ohlcv_bytes=full_bytes - baseline_bytes,
                  series_reduction_vs_rows_percent=round(100 * (1 - series_bytes / row_bytes), 2))
    for encoding_name, encoder in (tokenizers or {}).items():
        for name, data in data_sections.items():
            counts = result["candidates"][name].setdefault("tokens", {})
            counts[encoding_name] = {"data_section": len(encoder.encode(compact_json(data))),
                                     "full_response": len(encoder.encode(compact_json(full[name])))}
        baseline_tokens = len(encoder.encode(compact_json(baseline)))
        series_tokens = result["candidates"]["shared_date_series"]["tokens"][encoding_name]["full_response"]
        rows_tokens = result["candidates"]["chronological_row_objects"]["tokens"][encoding_name]["full_response"]
        result.setdefault("token_comparison", {})[encoding_name] = {
            "baseline_2_0": baseline_tokens, "added_ohlcv": series_tokens - baseline_tokens,
            "shared_matches_or_improves_rows": series_tokens <= rows_tokens,
        }
    return result


def representative_comparisons(response, tokenizers=None):
    results = {}
    for name, count, null_heavy in (("small_union", 2, False), ("large_union", 150, False),
                                     ("null_heavy", 40, True)):
        payload = response.model_dump()
        payload["indicators"] = {f"Representative Feature {index}":
                                 [None, index + 0.123456789, None] if null_heavy else [index, index + 1, index + 2]
                                 for index in range(count)}
        payload["feature_filter"].update(included_count=count, selected_union_count=count,
                                         selected_unavailable_count=0, excluded_not_selected_count=0)
        payload["quality"]["null_counts_by_session"] = [count, 0, count] if null_heavy else [0, 0, 0]
        if null_heavy:
            payload["ohlcv"] = {key: [None, values[1], None] for key, values in payload["ohlcv"].items()}
            payload["quality"]["ohlcv_null_counts_by_session"] = [5, 0, 5]
        results[name] = compare_formats(TechnicalResponse.model_validate(payload), tokenizers)
    return results


def audit_artifact(service, ticker):
    """Audit OHLCV even when missing metadata prevents endpoint readiness."""
    try:
        frame = service._validate_technical(service.repository.get_technical_indicators(ticker))
        missing = sorted(set(OHLCV_HEADERS) - set(frame.columns))
        if missing:
            return {"ticker": ticker, "status": "ohlcv_data_missing", "missing_headers": missing}
        cutoff = service.clock().astimezone(ZoneInfo("Asia/Jakarta")).date()
        window = frame.loc[frame["Date"] < cutoff].tail(3)
        if len(window) < 3:
            return {"ticker": ticker, "status": "insufficient_sessions", "eligible_sessions": len(window)}
        bars = service._ohlcv(window).model_dump()
        return {"ticker": ticker, "status": "valid", "dates": [value.isoformat() for value in window["Date"]],
                "ohlcv_null_counts_by_session": [sum(values[i] is None for values in bars.values()) for i in range(3)],
                "volume_integral_nonnegative": True, "observed_bar_consistency": True,
                "unit_provenance": "unknown_no_persisted_unit_metadata"}
    except TechnicalError as exc:
        return {"ticker": ticker, "status": exc.code}


def audit_ticker(service, ticker, forecast_type, tokenizers=None) -> dict:
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
            "format_comparison": compare_formats(response, tokenizers)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", action="append", help="Repeat for selected tickers; default: all served CSV tickers")
    parser.add_argument("--forecast-type", choices=("5dd", "10dd"), action="append", help="Default: both horizons")
    parser.add_argument("--request-date", type=date.fromisoformat,
                        help="Fixed Jakarta cutoff for reproducible audits (not historical backtesting)")
    parser.add_argument("--tokenizer", action="append", help="Configured consumer's tiktoken encoding; requires tiktoken")
    args = parser.parse_args()
    tokenizers = {}
    if args.tokenizer:
        try:
            import tiktoken
            tokenizers = {name: tiktoken.get_encoding(name) for name in args.tokenizer}
        except (ImportError, ValueError) as exc:
            parser.error(f"Requested tokenizer unavailable: {exc}")
    tickers = sorted(set(ticker.upper() for ticker in args.ticker)) if args.ticker else sorted(
        path.stem for path in paths.get_technical_dir().glob("*.csv"))
    forecasts = list(dict.fromkeys(args.forecast_type or ("5dd", "10dd")))
    instant = (datetime.combine(args.request_date, time(), tzinfo=ZoneInfo("Asia/Jakarta"))
               if args.request_date else market_clock())
    logging.basicConfig(level=logging.ERROR)
    service = TechnicalService(TechnicalRepository(), clock=lambda: instant)
    artifacts = [audit_artifact(service, ticker) for ticker in tickers]
    records = [audit_ticker(service, ticker, forecast, tokenizers) for ticker in tickers for forecast in forecasts]
    representative = {}
    for forecast in forecasts:
        ready = next((record for record in records if record["available"] and record["forecast_type"] == forecast), None)
        if ready:
            response = service.get_technical_indicators(ready["ticker"], forecast)
            representative[forecast] = representative_comparisons(response, tokenizers)
    print(compact_json({"environment": paths.env, "request_date": instant.date().isoformat(),
                        "tickers": len(tickers), "requests": len(records),
                        "outcomes": dict(Counter(record["reason_code"] for record in records)),
                        "artifact_outcomes": dict(Counter(record["status"] for record in artifacts)),
                        "artifacts": artifacts, "records": records, "representative_comparisons": representative,
                        "tokenizers": list(tokenizers),
                        "measurement": "bytes_and_tokens" if tokenizers else "bytes_only_no_configured_tokenizer"}))


if __name__ == "__main__":
    main()
