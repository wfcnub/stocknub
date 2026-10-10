import copy
import json
from datetime import datetime, timezone
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.repositories.technical import TechnicalRepository
from app.routers.technical import get_technical_service, router
from app.schemas.technical import TechnicalResponse, metadata_sources
from app.services.technical import TechnicalService
from utils import paths

NOW = datetime(2026, 10, 10, 0, 1, tzinfo=ZoneInfo("Asia/Jakarta"))


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "STOCK_DIR", tmp_path / "stock")
    dates = ["2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24",
             "2026-10-10", "2026-10-11"]
    frame = pd.DataFrame({
        "Date": dates, "Gain": [1.123456789012345] * 7, "Loss": range(7),
        "Shared": [1, 0, np.inf, -np.inf, np.nan, 1, 1], "TenLoss": range(10, 17),
        "TenGain": range(20, 27), "Zero": [0] * 7, "Null": [np.nan] * 7,
        "Flag": [0, 1, 1, 0, 1, 0, 1], "Open": range(100, 107), "High": range(110, 117),
        "Low": range(90, 97), "Close": range(105, 112),
        "Volume": range(1000, 1007), "Case": [2] * 7,
    })
    technical_path = paths.get_technical_path("AALI")
    technical_path.parent.mkdir(parents=True)
    frame.to_csv(technical_path, index=False)
    selections = {
        ("median_loss", "5dd"): ["Loss", "Shared", "Zero", "Null", "Flag", "Unavailable"],
        ("median_gain", "5dd"): ["Gain", "Shared", "case"],
        ("median_loss", "10dd"): ["TenLoss", "Shared"],
        ("median_gain", "10dd"): ["TenGain", "Gain"],
    }
    for (label, window), selected in selections.items():
        artifact = paths.get_model_artifact_path(1, label, "AALI", window, "metadata", "json")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({"selected_features": selected, "NonZeroRate": 0}))
    repo = TechnicalRepository()
    service = TechnicalService(repo, clock=lambda: NOW)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_technical_service] = lambda: service
    with TestClient(app) as client:
        yield frame, repo, service, client


def metadata_path(label="median_loss", window="5dd", ticker="AALI"):
    return paths.get_model_artifact_path(1, label, ticker, window, "metadata", "json")


def request(client, forecast="5dd", ticker="aali"):
    return client.get("/technical/", params={"ticker": ticker, "forecast_type": forecast})


@pytest.mark.parametrize("route", ["/technical/", "/technical/check-availability"])
@pytest.mark.parametrize("params", [
    {"ticker": "AALI"}, {"ticker": "AALI", "forecast_type": ""},
    {"ticker": "AALI", "forecast_type": "20dd"},
    {"ticker": "AALI", "forecast_type": "../5dd"},
    {"ticker": "../AALI", "forecast_type": "5dd"},
    {"ticker": "AA/LI", "forecast_type": "5dd"},
    {"ticker": "", "forecast_type": "5dd"},
    {"ticker": "ＡＡＬＩ", "forecast_type": "5dd"},
    {"forecast_type": "5dd"},
])
def test_query_validation_before_file_access(dataset, monkeypatch, route, params):
    _, repo, _, client = dataset
    read = Mock(side_effect=AssertionError("Repository must not be read"))
    monkeypatch.setattr(repo, "get_technical_indicators", read)
    monkeypatch.setattr(repo, "get_model_metadata", read)
    assert client.get(route, params=params).status_code == 422
    read.assert_not_called()


@pytest.mark.parametrize("forecast", ["5dd", "10dd"])
def test_union_contract_and_serialization(dataset, monkeypatch, caplog, forecast):
    _, repo, _, client = dataset
    caplog.set_level("INFO")
    reader = Mock(wraps=repo.get_model_metadata)
    monkeypatch.setattr(repo, "get_model_metadata", reader)
    response = request(client, forecast)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ticker"] == "AALI"
    assert payload["forecast_type"] == forecast
    assert payload["schema_version"] == "2.1"
    assert payload["dates"] == ["2026-09-22", "2026-09-23", "2026-09-24"]
    assert payload["ohlcv"] == {"open": [102, 103, 104], "high": [112, 113, 114],
                                "low": [92, 93, 94], "close": [107, 108, 109],
                                "volume": [1002, 1003, 1004]}
    assert payload["quality"]["ohlcv_null_counts_by_session"] == [0, 0, 0]
    assert payload["ohlcv_metadata"] == {"source": "technical_csv", "price_currency": "unknown",
                                          "volume_unit": "unknown", "price_adjustment": "unknown"}
    assert payload["data_as_of"] == payload["dates"][-1]
    assert payload["requested_sessions"] == payload["returned_sessions"] == 3
    assert payload["quality"]["data_age_calendar_days"] == 16
    assert payload["quality"]["null_counts_by_session"] == [2, 2, 2]
    indicators = payload["indicators"]
    names = ["Gain", "Loss", "Shared"]
    if forecast == "10dd":
        names += ["TenLoss", "TenGain"]
    names += ["Zero", "Null", "Flag"]
    assert list(indicators) == names
    assert indicators["Loss"] == [2, 3, 4]
    assert indicators["Gain"] == [1.123456789012345] * 3
    assert indicators["Shared"] == [None] * 3
    assert indicators["Null"] == [None] * 3
    assert indicators["Zero"] == [0] * 3
    assert all(type(value) is int for value in indicators["Flag"])
    assert "Open" not in indicators and "Case" not in indicators and "case" not in indicators
    counts = payload["feature_filter"]
    assert counts["included_count"] == len(names)
    assert counts["selected_union_count"] == len(names) + 2
    assert counts["selected_unavailable_count"] == 2
    assert counts["excluded_not_selected_count"] == 14 - len(names)
    assert reader.call_args_list == [(("AALI", label, window),) for label, window in metadata_sources(forecast)]
    assert "sha256=" in caplog.text and "ohlcv_null_counts" in caplog.text
    json.dumps(payload, allow_nan=False)


def test_five_day_does_not_require_ten_day_metadata_and_subset(dataset):
    _, _, service, client = dataset
    five = service.get_technical_indicators("AALI", "5dd")
    ten = service.get_technical_indicators("AALI", "10dd")
    assert set(five.indicators) < set(ten.indicators)
    assert five.ohlcv == ten.ohlcv
    for label in ("median_loss", "median_gain"):
        metadata_path(label, "10dd").unlink()
    assert request(client, "5dd").status_code == 200
    assert request(client, "10dd").json()["detail"]["code"] == "feature_filter_unavailable"


@pytest.mark.parametrize("label,window", metadata_sources("10dd"))
def test_missing_required_source(dataset, label, window):
    _, _, _, client = dataset
    metadata_path(label, window).unlink()
    response = request(client, "10dd")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "feature_filter_unavailable"


@pytest.mark.parametrize("value", [
    [], None, "features", {}, {"selected_features": None}, {"selected_features": "Loss"},
    {"selected_features": []}, {"selected_features": [""]}, {"selected_features": [" "]},
    {"selected_features": [" Loss"]}, {"selected_features": ["Loss "]},
    {"selected_features": [1]}, {"selected_features": [None]},
    {"selected_features": ["Loss", "Loss"]}, {"selected_features": ["Loss", ["Gain"]]},
])
def test_invalid_metadata_content(dataset, value):
    _, _, _, client = dataset
    metadata_path().write_text(json.dumps(value))
    response = request(client)
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "invalid_model_metadata"
    assert str(paths.STOCK_DIR) not in response.text


@pytest.mark.parametrize("raw", [b"{", b"", b"\xff",
                                 b'{"selected_features":["Loss"],"other":NaN}',
                                 b'{"selected_features":[],"selected_features":["Loss"]}'])
def test_malformed_metadata_json(dataset, raw):
    _, _, _, client = dataset
    metadata_path().write_bytes(raw)
    assert request(client).json()["detail"]["code"] == "invalid_model_metadata"


def test_empty_intersection(dataset):
    _, _, _, client = dataset
    for label, window in metadata_sources("5dd"):
        metadata_path(label, window).write_text(json.dumps({"selected_features": ["Model Only"]}))
    response = request(client)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "no_eligible_indicators"


@pytest.mark.parametrize("forecast", ["5dd", "10dd"])
@pytest.mark.parametrize("order", ["ascending", "reversed", "shuffled"])
def test_row_order_does_not_change_selection(dataset, forecast, order):
    frame, _, service, _ = dataset
    expected = service.get_technical_indicators("AALI", forecast).model_dump()
    if order == "reversed":
        frame = frame.iloc[::-1]
    elif order == "shuffled":
        frame = frame.sample(frac=1, random_state=23)
    frame.to_csv(paths.get_technical_path("AALI"), index=False)
    assert service.get_technical_indicators("AALI", forecast).model_dump() == expected


@pytest.mark.parametrize("count,status,code", [(0, 404, "no_eligible_sessions"),
                                               (1, 409, "insufficient_sessions"),
                                               (2, 409, "insufficient_sessions"),
                                               (3, 200, None), (5, 200, None)])
@pytest.mark.parametrize("forecast", ["5dd", "10dd"])
def test_history_length(dataset, count, status, code, forecast):
    frame, _, _, client = dataset
    frame = pd.concat([frame.iloc[:count], frame.iloc[-2:]])
    frame.to_csv(paths.get_technical_path("AALI"), index=False)
    response = request(client, forecast)
    assert response.status_code == status
    if code:
        assert response.json()["detail"]["code"] == code
    else:
        assert len(response.json()["dates"]) == 3


@pytest.mark.parametrize("forecast", ["5dd", "10dd"])
def test_jakarta_date_at_utc_boundary_and_weekend_gap(dataset, forecast):
    frame, repo, _, _ = dataset
    frame = frame.iloc[:4].copy()
    frame["Date"] = ["2026-10-05", "2026-10-07", "2026-10-09", "2026-10-10"]
    frame.to_csv(paths.get_technical_path("AALI"), index=False)
    # UTC is still October 9, but the Jakarta date is October 10.
    service = TechnicalService(repo, lambda: datetime(2026, 10, 9, 17, 1, tzinfo=timezone.utc))
    result = service.get_technical_indicators("AALI", forecast)
    assert result.ohlcv.open == [100, 101, 102]
    assert result.dates == ["2026-10-05", "2026-10-07", "2026-10-09"]
    assert result.quality.request_date == "2026-10-10"


@pytest.mark.parametrize("raw", [
    "", "Date,Loss\n", "Loss\n1\n", "Date,Loss,Loss\n2026-09-22,1,1\n",
    "Date, Loss\n2026-09-22,1\n", "Date,\n2026-09-22,1\n",
    "Date,Loss\n2026-09-22\n", "Date,Loss\n2026-09-22,1,2\n",
    'Date,Loss\n"2026-09-22,1\n',
    "Date,Loss\n2026-02-30,1\n", "Date,Loss\n2026-9-22,1\n",
    "Date,Loss\n2026-09-22T00:00:00,1\n", "Date,Loss\n,1\n",
    "Date,Loss\n2026-09-22,1\n2026-09-22,2\n",
    "Date,Loss\n2026-09-22,unexpected\n",
    "Date,Loss\n2026-09-22,True\n",
    "Date,Loss,Nonselected\n2026-09-22,1,unexpected\n",
])
def test_invalid_csv(dataset, raw):
    _, _, _, client = dataset
    paths.get_technical_path("AALI").write_text(raw)
    response = request(client)
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "invalid_technical_data"


def test_missing_csv_and_ticker_specific_artifacts(dataset):
    frame, _, _, client = dataset
    assert request(client, ticker="BBCA").status_code == 404
    frame.to_csv(paths.get_technical_path("BBCA"), index=False)
    assert request(client, ticker="BBCA").status_code == 409
    for label, window in metadata_sources("5dd"):
        metadata_path(label, window, "BBCA").write_text(json.dumps({"selected_features": ["TenGain"]}))
    result = request(client, ticker="bbca")
    assert result.status_code == 200
    assert list(result.json()["indicators"]) == ["TenGain"]


def test_metadata_mutation_during_read_rejects_bundle(dataset, monkeypatch):
    _, repo, _, client = dataset
    original = repo.get_model_metadata

    def mutate(ticker, label, window):
        result = original(ticker, label, window)
        if label == "median_gain":
            metadata_path().write_text(json.dumps({"selected_features": ["Gain"]}))
        return result

    monkeypatch.setattr(repo, "get_model_metadata", mutate)
    response = request(client)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "feature_filter_unavailable"


def test_no_metadata_cache_or_reranking(dataset):
    _, _, _, client = dataset
    assert "Loss" in request(client).json()["indicators"]
    metadata_path().write_text(json.dumps({"selected_features": ["Zero", "Null"], "importance": 0}))
    result = request(client).json()["indicators"]
    assert "Loss" not in result
    assert result["Zero"] == [0, 0, 0] and result["Null"] == [None, None, None]


@pytest.mark.parametrize("failure,code,status", [
    ("missing_csv", "technical_data_missing", 404),
    ("missing_metadata", "feature_filter_unavailable", 409),
    ("invalid_metadata", "invalid_model_metadata", 500),
    ("invalid_csv", "invalid_technical_data", 500),
    ("short", "insufficient_sessions", 409),
    ("future", "no_eligible_sessions", 404),
    ("empty_intersection", "no_eligible_indicators", 409),
])
def test_availability_matches_retrieval_failures(dataset, failure, code, status):
    frame, _, _, client = dataset
    if failure == "missing_csv":
        paths.get_technical_path("AALI").unlink()
    elif failure == "missing_metadata":
        metadata_path().unlink()
    elif failure == "invalid_metadata":
        metadata_path().write_text("null")
    elif failure == "invalid_csv":
        paths.get_technical_path("AALI").write_text("")
    elif failure in ("short", "future"):
        frame = frame.iloc[:2] if failure == "short" else frame.iloc[-2:]
        frame.to_csv(paths.get_technical_path("AALI"), index=False)
    else:
        for label, window in metadata_sources("5dd"):
            metadata_path(label, window).write_text(json.dumps({"selected_features": ["Absent"]}))
    assert request(client).status_code == status
    response = client.get("/technical/check-availability", params={"ticker": "aali", "forecast_type": "5dd"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False and payload["reason_code"] == code
    assert payload["failure_status_code"] == status and payload["forecast_type"] == "5dd"


@pytest.mark.parametrize("forecast", ["5dd", "10dd"])
def test_ready_availability_does_not_promise_freshness(dataset, forecast):
    _, _, _, client = dataset
    ready = client.get("/technical/check-availability", params={"ticker": "AALI", "forecast_type": forecast}).json()
    data = request(client, forecast).json()
    assert ready["available"] is True and ready["reason_code"] == "ready"
    assert ready["quality"] == data["quality"]
    assert ready["data_as_of"] == data["data_as_of"]
    assert ready["quality"]["freshness_status"] == "unverified_without_exchange_calendar"


@pytest.mark.parametrize("mutation", ["dates", "as_of", "age", "sources", "count", "union", "nulls",
                                      "length", "string", "infinity", "boolean", "membership", "csv_count"])
def test_schema_rejects_inconsistent_contract(dataset, mutation):
    _, _, service, _ = dataset
    payload = copy.deepcopy(service.get_technical_indicators("AALI", "5dd").model_dump())
    context = None
    if mutation == "dates":
        payload["dates"][1] = payload["dates"][0]
    elif mutation == "as_of":
        payload["data_as_of"] = "2026-09-23"
    elif mutation == "age":
        payload["quality"]["data_age_calendar_days"] = 1
    elif mutation == "sources":
        payload["feature_filter"]["sources"].pop()
    elif mutation == "count":
        payload["feature_filter"]["included_count"] += 1
    elif mutation == "union":
        payload["feature_filter"]["selected_union_count"] += 1
    elif mutation == "nulls":
        payload["quality"]["null_counts_by_session"][0] = 0
    elif mutation == "length":
        payload["indicators"]["Loss"].pop()
    elif mutation in ("string", "infinity", "boolean"):
        payload["indicators"]["Loss"][0] = {"string": "2", "infinity": float("inf"), "boolean": True}[mutation]
    else:
        selected = set(payload["indicators"]) | {"Unavailable", "case"}
        columns = list(payload["indicators"]) + ["TenLoss", "TenGain", "Open", "Case"]
        if mutation == "membership":
            selected.remove("Loss")
        else:
            columns.append("Extra")
        context = {"selected_features": selected, "csv_columns": columns}
    with pytest.raises(ValidationError):
        TechnicalResponse.model_validate(payload, context=context)


def test_openapi_required_inputs_and_series_schema(dataset):
    _, _, _, client = dataset
    schema = client.get("/openapi.json").json()
    for route in ("/technical/", "/technical/check-availability"):
        params = {param["name"]: param for param in schema["paths"][route]["get"]["parameters"]}
        assert params["ticker"]["required"] and params["forecast_type"]["required"]
        assert params["forecast_type"]["schema"]["enum"] == ["5dd", "10dd"]
    series = schema["components"]["schemas"]["TechnicalResponse"]["properties"]["indicators"]["additionalProperties"]
    assert series["minItems"] == series["maxItems"] == 3
    for example in schema["components"]["schemas"]["TechnicalResponse"]["examples"]:
        TechnicalResponse.model_validate(example)


def test_metadata_list_order_does_not_affect_output(dataset):
    _, _, service, _ = dataset
    expected = service.get_technical_indicators("AALI", "10dd").model_dump()
    for label, window in metadata_sources("10dd"):
        path = metadata_path(label, window)
        data = json.loads(path.read_text())
        data["selected_features"].reverse()
        path.write_text(json.dumps(data))
    assert service.get_technical_indicators("AALI", "10dd").model_dump() == expected


def test_audit_reports_ready_and_missing_metadata(dataset):
    from scripts.audit_technical_api import audit_ticker, compact_json

    _, _, service, client = dataset
    report = audit_ticker(service, "AALI", "5dd")
    assert report["available"] is True
    assert all(source["exists"] for source in report["sources"])
    assert report["format_comparison"]["full_response_bytes"] == len(request(client).content)
    assert report["format_comparison"]["shared_date_series_bytes"] < report["format_comparison"]["chronological_row_objects_bytes"]
    json.loads(compact_json(report))
    metadata_path().unlink()
    failure = audit_ticker(service, "AALI", "5dd")
    assert failure["reason_code"] == "feature_filter_unavailable"
    assert failure["available"] is False and failure["sources"][0]["exists"] is False


def test_catalog_definitions_resolve_to_generating_functions():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "documentations/technical_feature_catalog_v1.json").read_text())
    assert catalog["version"] == "1.0"
    for name, definition in catalog["features"].items():
        assert name and all(definition[key] for key in ("meaning", "unit_scale", "numeric_type", "lookback_sessions"))
        source, function = definition["generating_function"].split(":")
        tree = ast.parse((root / source).read_text())
        assert function in {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert definition["publication_lag_sessions"] == (1 if name.startswith(("Foreign ", "Non Regular ")) else 0)
    assert catalog["features"]["MACD Percent"]["unit_scale"] == "decimal ratio"
    assert catalog["features"]["RSI Rate of Change 5"]["unit_scale"] == "RSI points"


def test_safe_unexpected_error(dataset, monkeypatch):
    _, repo, _, client = dataset
    monkeypatch.setattr(repo, "get_technical_indicators", Mock(side_effect=RuntimeError("secret /private/source")))
    response = request(client)
    assert response.status_code == 500 and response.json()["detail"]["code"] == "internal_error"
    assert "secret" not in response.text


def test_native_numpy_scalars_and_naive_clock(dataset):
    _, repo, _, _ = dataset
    assert type(TechnicalService._json_number(np.int64(1))) is int
    assert type(TechnicalService._json_number(np.float64(1.25))) is float
    assert TechnicalService._json_number(np.inf) is None
    with pytest.raises(ValueError, match="timezone-aware"):
        TechnicalService(repo, lambda: datetime(2026, 10, 10)).get_technical_indicators("AALI", "5dd")


def test_invalid_direct_service_inputs_do_not_access_repository(dataset, monkeypatch):
    _, repo, service, _ = dataset
    read = Mock(side_effect=AssertionError("Must not read files"))
    monkeypatch.setattr(repo, "get_technical_indicators", read)
    with pytest.raises(ValueError):
        service.get_technical_indicators("../AALI", "5dd")
    with pytest.raises(ValueError):
        service.get_technical_indicators("AALI", "../5dd")
    read.assert_not_called()


@pytest.mark.parametrize("header", ["Open", "High", "Low", "Close", "Volume"])
@pytest.mark.parametrize("forecast", ["5dd", "10dd"])
def test_missing_ohlcv_header_readiness(dataset, header, forecast):
    frame, _, _, client = dataset
    frame.drop(columns=header).to_csv(paths.get_technical_path("AALI"), index=False)
    result = request(client, forecast)
    assert result.status_code == 409
    assert result.json()["detail"]["code"] == "ohlcv_data_missing"
    ready = client.get("/technical/check-availability", params={"ticker": "AALI", "forecast_type": forecast}).json()
    assert ready["reason_code"] == "ohlcv_data_missing" and ready["failure_status_code"] == 409
    assert str(paths.STOCK_DIR) not in result.text


@pytest.mark.parametrize("header,value", [
    ("Volume", -1), ("Volume", 1.5), ("Volume", "oops"), ("Volume", True),
    ("Open", -1), ("High", -1), ("Low", -1), ("Close", -1),
    ("High", 80), ("Open", 120), ("Open", 80), ("Close", 120), ("Close", 80),
])
def test_invalid_ohlcv_readiness(dataset, header, value):
    frame, _, _, client = dataset
    frame[header] = frame[header].astype(object)
    frame.loc[3, header] = value
    frame.to_csv(paths.get_technical_path("AALI"), index=False)
    result = request(client)
    assert result.status_code == 500
    assert result.json()["detail"]["code"] == "invalid_technical_data"
    ready = client.get("/technical/check-availability", params={"ticker": "AALI", "forecast_type": "5dd"}).json()
    assert ready["reason_code"] == "invalid_technical_data" and ready["failure_status_code"] == 500


@pytest.mark.parametrize("forecast", ["5dd", "10dd"])
def test_partial_ohlcv_nulls_zero_and_integral_volume(dataset, forecast):
    frame, _, _, client = dataset
    frame = frame.astype({name: float for name in ("Open", "High", "Low", "Close", "Volume")})
    frame.loc[2, ["Open", "High", "Low", "Close", "Volume"]] = [np.nan, np.inf, -np.inf, np.nan, np.inf]
    frame.loc[3, ["Open", "High", "Low", "Close", "Volume"]] = [0, 0, 0, 0, 0]
    frame.loc[4, "Volume"] = 1234567.0
    frame.to_csv(paths.get_technical_path("AALI"), index=False)
    result = request(client, forecast)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["dates"] == ["2026-09-22", "2026-09-23", "2026-09-24"]
    assert all(values[:2] == [None, 0] for values in data["ohlcv"].values())
    assert data["ohlcv"]["volume"] == [None, 0, 1234567]
    assert data["quality"]["ohlcv_null_counts_by_session"] == [5, 0, 0]
    assert data["quality"]["null_counts_by_session"] == [2, 2, 2]
    json.dumps(data, allow_nan=False)
    ready = client.get("/technical/check-availability", params={"ticker": "AALI", "forecast_type": forecast}).json()
    assert ready["available"] and ready["quality"] == data["quality"]
    assert "ohlcv" not in ready


@pytest.mark.parametrize("high,low,close,valid", [
    (None, 100, 99, False), (100, None, 101, False),
    (None, None, 500, True), (100, 100, 100, True),
])
def test_partial_bar_comparisons(dataset, high, low, close, valid):
    frame, _, _, client = dataset
    frame.loc[3, ["Open", "High", "Low", "Close"]] = [np.nan, high, low, close]
    frame.to_csv(paths.get_technical_path("AALI"), index=False)
    assert request(client).status_code == (200 if valid else 500)


def test_selected_ohlcv_preserves_indicator_membership_and_counts(dataset):
    _, _, service, _ = dataset
    before = service.get_technical_indicators("AALI", "5dd")
    path = metadata_path()
    metadata = json.loads(path.read_text())
    metadata["selected_features"].append("Open")
    path.write_text(json.dumps(metadata))
    result = service.get_technical_indicators("AALI", "5dd")
    assert result.ohlcv == before.ohlcv
    assert result.indicators["Open"] == result.ohlcv.open
    assert result.feature_filter.included_count == before.feature_filter.included_count + 1
    assert result.feature_filter.selected_union_count == before.feature_filter.selected_union_count + 1
    assert result.feature_filter.excluded_not_selected_count == before.feature_filter.excluded_not_selected_count - 1


@pytest.mark.parametrize("mutation", ["missing", "extra", "length", "bool", "string", "infinity",
                                      "negative_price", "bar", "float_volume", "negative_volume", "bool_volume", "string_volume",
                                      "counts", "count_range", "version"])
def test_ohlcv_schema_rejects_invalid_contract(dataset, mutation):
    _, _, service, _ = dataset
    payload = service.get_technical_indicators("AALI", "5dd").model_dump()
    bars = payload["ohlcv"]
    if mutation == "missing":
        del bars["low"]
    elif mutation == "extra":
        bars["other"] = [1, 2, 3]
    elif mutation == "length":
        bars["open"].pop()
    elif mutation in ("bool", "string", "infinity", "negative_price", "bar"):
        bars["open"][0] = {"bool": True, "string": "102", "infinity": np.inf, "negative_price": -1, "bar": 500}[mutation]
    elif mutation in ("float_volume", "negative_volume", "bool_volume", "string_volume"):
        bars["volume"][0] = {"float_volume": 1002.0, "negative_volume": -1,
                             "bool_volume": True, "string_volume": "1002"}[mutation]
    elif mutation in ("counts", "count_range"):
        payload["quality"]["ohlcv_null_counts_by_session"][0] = 1 if mutation == "counts" else 6
    else:
        payload["schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        TechnicalResponse.model_validate(payload)


def test_ohlcv_single_read_and_numpy_serialization(dataset, monkeypatch):
    frame, repo, service, _ = dataset
    reader = Mock(return_value=frame)
    monkeypatch.setattr(repo, "get_technical_indicators", reader)
    result = service.get_technical_indicators("AALI", "5dd")
    reader.assert_called_once_with("AALI")
    assert all(type(value) is int for value in result.ohlcv.volume)
    json.loads(result.model_dump_json())


def test_openapi_ohlcv_contract(dataset):
    _, _, _, client = dataset
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert {"ohlcv", "ohlcv_metadata"} <= set(schemas["TechnicalResponse"]["required"])
    bars = schemas["OHLCVSeries"]
    assert set(bars["required"]) == set(bars["properties"]) == {"open", "high", "low", "close", "volume"}
    assert bars["additionalProperties"] is False
    for field in bars["properties"].values():
        assert field["minItems"] == field["maxItems"] == 3
    volume = bars["properties"]["volume"]["items"]["anyOf"]
    assert {"minimum": 0, "type": "integer"} in volume


def test_audit_equivalent_formats_and_representative_cases(dataset):
    from scripts.audit_technical_api import audit_artifact, compare_formats, representative_comparisons

    _, _, service, _ = dataset
    for forecast in ("5dd", "10dd"):
        response = service.get_technical_indicators("AALI", forecast)
        result = compare_formats(response)
        assert result["measurement"] == "bytes_only"
        assert result["deterministic_extraction"]["status"] == "passed"
        assert result["added_ohlcv_bytes"] > 0
        assert set(result["candidates"]) == {"shared_date_series", "chronological_row_objects", "labeled_table"}
        assert result["full_response_bytes"] == len(response.model_dump_json().encode())
        cases = representative_comparisons(response)
        assert set(cases) == {"small_union", "large_union", "null_heavy"}
        assert cases["null_heavy"]["deterministic_extraction"]["ohlcv_null_counts_by_session"] == [5, 0, 5]
    metadata_path().unlink()
    assert audit_artifact(service, "AALI")["status"] == "valid"


def test_bar_validation_uses_returned_window_but_numeric_integrity_uses_whole_csv(dataset):
    frame, _, _, client = dataset
    # These bars are outside the selected window and do not affect retrieval.
    frame.loc[[0, 5, 6], "Volume"] = -1
    frame.loc[[0, 5, 6], "Close"] = -1
    frame.to_csv(paths.get_technical_path("AALI"), index=False)
    assert request(client).status_code == 200
    frame["Volume"] = frame["Volume"].astype(object)
    frame.loc[0, "Volume"] = "bad numeric data"
    frame.to_csv(paths.get_technical_path("AALI"), index=False)
    assert request(client).json()["detail"]["code"] == "invalid_technical_data"


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_mixed_boolean_observations_do_not_coerce_to_numbers(dataset, monkeypatch, value):
    frame, repo, _, client = dataset
    frame["Volume"] = frame["Volume"].astype(object)
    frame.loc[3, "Volume"] = value
    monkeypatch.setattr(repo, "get_technical_indicators", lambda ticker: frame)
    assert request(client).json()["detail"]["code"] == "invalid_technical_data"
