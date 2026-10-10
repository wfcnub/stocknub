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
        "Flag": [0, 1, 1, 0, 1, 0, 1], "Open": [100] * 7, "Case": [2] * 7,
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
    assert payload["schema_version"] == "2.0"
    assert payload["dates"] == ["2026-09-22", "2026-09-23", "2026-09-24"]
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
    assert counts["excluded_not_selected_count"] == 10 - len(names)
    assert reader.call_args_list == [(("AALI", label, window),) for label, window in metadata_sources(forecast)]
    assert "sha256=" in caplog.text and "Unavailable" in caplog.text
    json.dumps(payload, allow_nan=False)


def test_five_day_does_not_require_ten_day_metadata_and_subset(dataset):
    _, _, service, client = dataset
    five = service.get_technical_indicators("AALI", "5dd")
    ten = service.get_technical_indicators("AALI", "10dd")
    assert set(five.indicators) < set(ten.indicators)
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


def test_jakarta_date_at_utc_boundary_and_weekend_gap(dataset):
    frame, repo, _, _ = dataset
    frame = frame.iloc[:4].copy()
    frame["Date"] = ["2026-10-05", "2026-10-07", "2026-10-09", "2026-10-10"]
    frame.to_csv(paths.get_technical_path("AALI"), index=False)
    # UTC is still October 9, but the Jakarta date is October 10.
    service = TechnicalService(repo, lambda: datetime(2026, 10, 9, 17, 1, tzinfo=timezone.utc))
    result = service.get_technical_indicators("AALI", "5dd")
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
