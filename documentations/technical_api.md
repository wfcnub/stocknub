# Technical API, schema 2.1

`GET /technical/?ticker=AALI&forecast_type=5dd` returns historical OHLCV and
model-selected technical indicators. `ticker` is four ASCII letters (normalized
to uppercase); `forecast_type` is required and accepts `5dd` or `10dd`.

## Sessions and alignment

Exactly three latest available source sessions strictly before the Jakarta
request date are returned, oldest to newest. For every index `i`, `dates[i]`,
every `ohlcv` series at `i`, and every `indicators` series at `i` describe the
same historical session. Weekends and other source gaps are preserved. Same-day
and future sessions are excluded. No session is skipped to obtain a complete bar.

`forecast_type` controls the forecast horizon and metadata union, not history
length. Identical ticker, artifact and cutoff requests have identical OHLCV
across horizons. The 5dd union uses model-v1 medianLoss and medianGain 5dd
metadata; 10dd adds both 10dd sources. Indicators preserve exact CSV headers and
equal the selected-feature/CSV intersection. OHLCV is always included independently
of selection. If metadata selects `Open`, it also remains in `indicators`.

## Complete illustrative response

Counts and observations below are examples, not live market data.

```json
{
  "data_as_of": "2026-09-24",
  "dates": [
    "2026-09-22",
    "2026-09-23",
    "2026-09-24"
  ],
  "feature_catalog_version": "1.0",
  "feature_filter": {
    "excluded_not_selected_count": 5,
    "included_count": 2,
    "key": "selected_features",
    "model_version": 1,
    "selected_unavailable_count": 1,
    "selected_union_count": 3,
    "source": "model_metadata",
    "sources": [
      {
        "label_type": "medianLoss",
        "window": "5dd"
      },
      {
        "label_type": "medianGain",
        "window": "5dd"
      }
    ],
    "strategy": "union"
  },
  "forecast_type": "5dd",
  "indicators": {
    "RSI Up Trend": [
      0,
      0,
      0
    ],
    "RSI Value": [
      42.7,
      42.7,
      null
    ]
  },
  "market_timezone": "Asia/Jakarta",
  "ohlcv": {
    "close": [
      7050,
      7100,
      7150
    ],
    "high": [
      7100,
      7150,
      7200
    ],
    "low": [
      6950,
      7000,
      7050
    ],
    "open": [
      7000,
      7050,
      7100
    ],
    "volume": [
      1200000,
      1300000,
      null
    ]
  },
  "ohlcv_metadata": {
    "price_adjustment": "unknown",
    "price_currency": "unknown",
    "source": "technical_csv",
    "volume_unit": "unknown"
  },
  "order": "oldest_to_newest",
  "quality": {
    "data_age_calendar_days": 16,
    "freshness_status": "unverified_without_exchange_calendar",
    "null_counts_by_session": [
      0,
      0,
      1
    ],
    "ohlcv_null_counts_by_session": [
      0,
      0,
      1
    ],
    "request_date": "2026-10-10"
  },
  "requested_sessions": 3,
  "returned_sessions": 3,
  "schema_version": "2.1",
  "selection_policy": "latest_available_before_request_date",
  "ticker": "AALI"
}
```

## Field definitions

| Field | Meaning |
| --- | --- |
| `schema_version` | `2.1`; strict consumers must support this version and its required fields. |
| `ticker`, `forecast_type` | Normalized ticker and requested metadata horizon. |
| `market_timezone` | `Asia/Jakarta`, used to calculate the request date. |
| `selection_policy` | `latest_available_before_request_date`. |
| `requested_sessions`, `returned_sessions` | Both always three on success. |
| `dates`, `order`, `data_as_of` | Shared session axis, `oldest_to_newest`, and latest returned session. |
| `ohlcv` | Exactly `open`, `high`, `low`, `close`, `volume`, each with three observations. |
| `ohlcv_metadata` | Source `technical_csv`, currency, volume unit and price adjustment basis. |
| `quality.request_date` | Request date in Jakarta. |
| `quality.data_age_calendar_days` | Calendar-day difference between request date and `data_as_of`. |
| `quality.freshness_status` | `unverified_without_exchange_calendar`; source age is not proof of exchange-session freshness. |
| `quality.null_counts_by_session` | Counts only emitted indicators; three counts aligned with dates. |
| `quality.ohlcv_null_counts_by_session` | Counts only the five OHLCV observations; three counts from zero to five. |
| `feature_filter` | Metadata provenance and selection counts, retaining schema-2.0 semantics. |
| `feature_catalog_version` | Indicator definition catalog version `1.0`; see the catalog status below. |
| `indicators` | Selected source columns as numeric-or-null series on the same dates. |

`feature_filter.selected_union_count` equals `included_count` plus
`selected_unavailable_count`. `included_count` equals the number of indicators;
`excluded_not_selected_count` counts non-Date CSV columns outside the union,
including OHLCV columns when not selected. Adding the OHLCV response object does
not change those counts. The selected/unavailable name lists and source paths
are not included in responses.

## Observations, units and provenance

`null` means unavailable. NaN and either infinity serialize as null. Zero remains
an observation, including zero volume and zero prices. There is no filling,
interpolation, carry-forward, or observation rounding. Finite prices must be
nonnegative. Observed high must be at least observed low; observed open and close
must be within each available bound. Comparisons involving missing values are
not performed. Finite volume must be nonnegative and integral; integral source
floats convert losslessly to JSON integers. Fractional volume is rejected.

The existing stored artifacts have no persisted currency, volume-unit or
adjustment provenance. The current response explicitly reports all three as
`unknown`. The fetcher uses Yahoo `.JK` history and the generator copies its
OHLCV columns, but ticker suffixes alone do not establish artifact units or
adjustment basis. Do not interpret stored prices as execution prices. Future
explicit download settings do not retroactively establish existing provenance.
The typed contract permits `IDR`/`shares` and `adjusted`/`unadjusted` only when
artifact provenance can establish them; serving code currently makes no such claim.

## Failures and readiness

Errors use `{"detail":{"code":"...","message":"...","ticker":"AALI","forecast_type":"5dd"}}`
without exposing source paths. Query validation returns FastAPI's standard 422.

| Status | Code | Meaning |
| --- | --- | --- |
| 404 | `technical_data_missing` | No technical artifact for the ticker. |
| 404 | `no_eligible_sessions` | No source sessions strictly before the cutoff. |
| 409 | `ohlcv_data_missing` | One or more required OHLCV headers are absent. |
| 409 | `insufficient_sessions` | Fewer than three eligible sessions. |
| 409 | `feature_filter_unavailable` | A required metadata source is missing or changed during retrieval. |
| 409 | `no_eligible_indicators` | No selected features exist in the CSV. |
| 500 | `invalid_technical_data` | CSV/numeric integrity failure, negative prices, invalid observed bars, fractional/negative volume. |
| 500 | `invalid_model_metadata` | Invalid required metadata. |
| 500 | `internal_error` | Unexpected failure, safely redacted. |

`GET /technical/check-availability?ticker=AALI&forecast_type=5dd` executes the
same retrieval path, including OHLCV and metadata validation. Domain failures
return HTTP 200 with `available=false`, `reason_code` and `failure_status_code`.
A ready response includes `data_as_of` and the expanded `quality` object without
OHLCV arrays. Present columns with partial or entirely missing observations can
be ready; readiness means a valid payload can be returned, not complete bars or
confirmed exchange-calendar freshness. OHLCV cannot bypass model readiness.

## Migration and rollout

Schema 2.1 retains schema-2.0 field meanings and adds required `ohlcv`,
`ohlcv_metadata`, and `quality.ohlcv_null_counts_by_session`. Update strict
consumer models and their version checks before enabling the new server.
Models that forbid extra fields can reject this additive change. Agent tool
descriptions must state the shared-date rule and historical-session distinction.

No consumer parser or agent tool definition is maintained in this checkout.
External consumers and deployment configuration must be coordinated separately.
If consumers cannot migrate together, publish a temporary parallel 2.0 service
contract during deployment; this implementation serves 2.1 only. Audit deployed
artifacts read-only before rollout. Publish repaired missing-header artifacts
through the normal pipeline rather than using request-time fallbacks. Monitor
`ohlcv_data_missing`, invalid-data errors, separate OHLCV null counts and payload
sizes. Serving logs record concise counts, not full response bodies.

## Validation and context-size audit

Run `python -m pytest tests/test_technical_api.py -q`. For a reproducible read-only
local audit, run `python -m scripts.audit_technical_api --request-date 2026-10-10`.
The audit includes OHLCV validity even where missing metadata prevents readiness.
It compares equivalent complete 2.1 shared arrays, dated row objects and a labeled
JSON table, preserving units, quality, provenance and observations in each.
It reports full-response and data-section bytes, added OHLCV bytes relative to
2.0, and synthetic small/large/null-heavy unions for both available horizons.

If the consuming agent's tiktoken encoding is configured, install tiktoken and
pass `--tokenizer ENCODING` (repeatable). This measures actual input tokens and
reports whether arrays match or improve row costs. Without a configured tokenizer,
results are explicitly byte-only; bytes are not evidence of token savings.
Deterministic extraction checks reconstruct all values and verify latest date,
close, volume and null counts. They are not an LLM/agent smoke evaluation; no
configured agent consumer is present in this repository.

## Existing catalog gap

`documentations/technical_feature_catalog_v1.json`, referenced by README and the
existing catalog regression test, was absent before this change. Consumers cannot
obtain catalog 1.0 definitions from that link until the catalog is restored in a
separate documentation task. The generating functions live in
`prepareTechnicalIndicators/`; do not infer feature meanings from names alone.
The baseline and audit findings are recorded in
[OHLCV implementation validation](technical_ohlcv_validation.md).
