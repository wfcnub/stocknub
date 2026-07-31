import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from selectTickerToProcess.config import SelectionConfig
from selectTickerToProcess.helper import (
    _fetch_fundamentals,
    _calculate_fundamental_score,
)
from selectTickerToProcess.ohlcv import build_ohlcv_universe
from selectTickerToProcess.scoring import (
    FUNDAMENTAL_COLUMNS,
    calculate_fundamental_scores,
    calculate_technical_scores,
    combine_scores,
    select_with_sector_controls,
)

def select_ticker_to_process(ticker: str) -> pd.DataFrame | None:
    """
    A process of selecting a ticker based on fundamental analysis.

    This function evaluates a single ticker to check its metrics and compute its fundamental score.

    Args:
        ticker (str): The ticker symbol to process
    
    Returns:
        pd.DataFrame | None: A pandas dataframe containing the ticker's data, or None if it fails.
    """
    
    fundamental_df = _fetch_fundamentals([ticker])
    
    if fundamental_df.empty:
        return None
        
    scored_ticker_df = _calculate_fundamental_score(fundamental_df)

    if scored_ticker_df.empty:
        return None

    return scored_ticker_df


def _load_fundamental_snapshot(path: Path) -> pd.DataFrame:
    snapshot = pd.read_csv(path)
    if "Ticker" not in snapshot:
        raise ValueError(f"Fundamental snapshot {path} has no Ticker column")
    if "fetched_at" in snapshot:
        snapshot["_fetched_at"] = pd.to_datetime(
            snapshot["fetched_at"], errors="coerce", utc=True
        )
        snapshot = snapshot.sort_values("_fetched_at").drop_duplicates(
            "Ticker", keep="last"
        )
        snapshot = snapshot.drop(columns="_fetched_at")
    else:
        snapshot = snapshot.drop_duplicates("Ticker", keep="last")
    return snapshot


def build_selection_universe(
    ohlcv_dir: Path,
    auxiliary_dir: Path | None,
    industry_path: Path,
    config: SelectionConfig,
    fundamental_cache_path: Path | None = None,
    fundamental_snapshot_path: Path | None = None,
    previous_selection_path: Path | None = None,
    refresh_fundamentals: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the selected model universe and its full rejection audit."""
    audit = build_ohlcv_universe(ohlcv_dir, auxiliary_dir, config)
    if audit.empty:
        return pd.DataFrame(), audit, {"fundamental_fetch_coverage": 0.0}

    industry = pd.read_csv(industry_path, usecols=["Ticker", "Industry"])
    industry = industry.drop_duplicates("Ticker", keep="last")
    audit = audit.merge(industry, on="Ticker", how="left")
    audit["Industry"] = audit["Industry"].fillna("Unknown")

    eligible_indices = audit.index[audit["ohlcv_eligible"]]
    if len(eligible_indices):
        technical = calculate_technical_scores(audit.loc[eligible_indices])
        technical_columns = [
            column
            for column in technical.columns
            if column not in audit.columns or column == "technical_score"
        ]
        for column in technical_columns:
            audit.loc[eligible_indices, column] = technical[column]

    eligible_tickers = audit.loc[audit["ohlcv_eligible"], "Ticker"].tolist()
    if fundamental_snapshot_path is not None:
        fundamentals = _load_fundamental_snapshot(fundamental_snapshot_path)
    else:
        fundamentals = _fetch_fundamentals(
            eligible_tickers,
            cache_path=fundamental_cache_path,
            ttl_days=config.fundamental_cache_ttl_days,
            max_workers=config.fundamental_fetch_workers,
            force_refresh=refresh_fundamentals,
        )

    duplicate_columns = [
        column for column in ["Industry", "fundamental_score", "technical_score"]
        if column in fundamentals
    ]
    fundamentals = fundamentals.drop(columns=duplicate_columns, errors="ignore")
    keep_columns = [
        column
        for column in fundamentals.columns
        if column == "Ticker"
        or column in FUNDAMENTAL_COLUMNS
        or column.startswith("fetch_")
        or column in {"fetched_at", "fundamental_cache_stale", "averageVolume", "volume", "regularMarketPrice"}
    ]
    fundamentals = fundamentals[keep_columns].drop_duplicates("Ticker", keep="last")
    audit = audit.merge(fundamentals, on="Ticker", how="left")

    metric_columns = [column for column in FUNDAMENTAL_COLUMNS if column in audit]
    audit["fundamental_provider_available"] = audit[metric_columns].notna().any(axis=1)
    if "fundamental_cache_stale" in audit:
        cache_stale = audit["fundamental_cache_stale"].fillna(False).astype(bool)
    elif "fetched_at" in audit:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
            days=config.fundamental_cache_ttl_days
        )
        cache_stale = pd.to_datetime(
            audit["fetched_at"], errors="coerce", utc=True
        ).lt(cutoff).fillna(False)
    else:
        cache_stale = pd.Series(False, index=audit.index)
    audit["fundamental_data_fresh"] = (
        audit["fundamental_provider_available"] & ~cache_stale
    )
    fetched_eligible = audit.loc[
        audit["ohlcv_eligible"], "fundamental_data_fresh"
    ].sum()
    fetch_coverage = (
        float(fetched_eligible / len(eligible_tickers)) if eligible_tickers else 0.0
    )

    score_indices = audit.index[
        audit["ohlcv_eligible"] & audit["fundamental_provider_available"]
    ]
    if len(score_indices):
        fundamental_scored = calculate_fundamental_scores(
            audit.loc[score_indices],
            config,
        )
        score_columns = [
            column
            for column in fundamental_scored.columns
            if column not in audit.columns or column.startswith("fundamental_") or column.endswith("_score")
        ]
        for column in score_columns:
            audit.loc[score_indices, column] = fundamental_scored[column]

    incumbents: set[str] = set()
    if previous_selection_path is not None and previous_selection_path.is_file():
        try:
            incumbents = set(pd.read_csv(previous_selection_path, usecols=["Ticker"])["Ticker"])
        except (OSError, ValueError, pd.errors.ParserError):
            incumbents = set()

    for column in ["fundamental_coverage", "fundamental_score", "technical_score"]:
        if column not in audit:
            audit[column] = np.nan

    combinable = audit["fundamental_score"].notna() & audit["technical_score"].notna()
    if combinable.any():
        combined = combine_scores(audit.loc[combinable], config, incumbents)
        for column in ["incumbent", "combined_score", "selection_priority"]:
            audit.loc[combinable, column] = combined[column]
    if "incumbent" not in audit:
        audit["incumbent"] = False
    else:
        audit["incumbent"] = audit["incumbent"].fillna(False).astype(bool)

    audit["selection_eligible"] = (
        audit["ohlcv_eligible"]
        & audit["fundamental_data_fresh"]
        & audit["fundamental_coverage"].fillna(0).ge(config.min_fundamental_coverage)
        & audit["fundamental_score"].fillna(0).ge(config.min_fundamental_score)
        & audit["technical_score"].fillna(0).ge(config.min_technical_score)
    )

    selection_reasons = audit["ohlcv_eligibility_reasons"].fillna("").str.split(";")
    selection_reasons = selection_reasons.map(
        lambda values: [value for value in values if value]
    )
    additional_reasons = [
        (
            "fundamental_data_unavailable",
            audit["ohlcv_eligible"] & ~audit["fundamental_provider_available"],
        ),
        (
            "stale_fundamental_data",
            audit["ohlcv_eligible"]
            & audit["fundamental_provider_available"]
            & ~audit["fundamental_data_fresh"],
        ),
        (
            "insufficient_fundamental_coverage",
            audit["ohlcv_eligible"]
            & audit["fundamental_provider_available"]
            & audit["fundamental_coverage"].fillna(0).lt(config.min_fundamental_coverage),
        ),
        (
            "fundamental_score_below_minimum",
            audit["ohlcv_eligible"]
            & audit["fundamental_score"].notna()
            & audit["fundamental_score"].lt(config.min_fundamental_score),
        ),
        (
            "technical_score_below_minimum",
            audit["ohlcv_eligible"]
            & audit["technical_score"].notna()
            & audit["technical_score"].lt(config.min_technical_score),
        ),
    ]
    for reason, mask in additional_reasons:
        for position in np.flatnonzero(mask.to_numpy(dtype=bool)):
            selection_reasons.iloc[position].append(reason)
    audit["selection_rejection_reasons"] = selection_reasons.map(";".join)

    candidates = audit[audit["selection_eligible"]].copy()
    selected_tickers = select_with_sector_controls(candidates, config)
    audit["selected"] = audit["Ticker"].isin(selected_tickers)
    not_selected_candidates = audit["selection_eligible"] & ~audit["selected"]
    audit.loc[
        not_selected_candidates,
        "selection_rejection_reasons",
    ] = "not_selected_by_rank_or_sector_capacity"
    audit["overall_rank"] = np.nan
    if not candidates.empty:
        candidate_ranks = candidates["selection_priority"].rank(
            method="first", ascending=False
        )
        audit.loc[candidates.index, "overall_rank"] = candidate_ranks

    audit["selector_version"] = "fundamental_technical_v1"
    audit["selection_config_json"] = json.dumps(asdict(config), sort_keys=True)

    selected = audit[audit["selected"]].sort_values(
        ["selection_priority", "Ticker"], ascending=[False, True]
    )
    audit = audit.sort_values(
        ["selected", "selection_eligible", "selection_priority", "Ticker"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    selected = selected.reset_index(drop=True)

    summary = {
        "universe_count": int(len(audit)),
        "ohlcv_eligible_count": int(audit["ohlcv_eligible"].sum()),
        "fundamental_fetch_coverage": fetch_coverage,
        "selection_eligible_count": int(audit["selection_eligible"].sum()),
        "selected_count": int(len(selected)),
    }
    return selected, audit, summary
