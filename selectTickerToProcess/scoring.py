import numpy as np
import pandas as pd

from selectTickerToProcess.config import SelectionConfig


FUNDAMENTAL_COLUMNS = [
    "trailingPE",
    "forwardPE",
    "pegRatio",
    "priceToBook",
    "returnOnEquity",
    "returnOnAssets",
    "profitMargins",
    "operatingMargins",
    "debtToEquity",
    "currentRatio",
    "quickRatio",
    "freeCashflow",
    "operatingCashflow",
    "revenueGrowth",
    "earningsGrowth",
    "marketCap",
]

TECHNICAL_FACTORS = {
    "technical_trend_score": {
        "distance_from_ma50": True,
        "distance_from_ma200": True,
        "ma200_slope_63": True,
    },
    "technical_momentum_score": {
        "momentum_6m_ex_1m": True,
        "momentum_12m_ex_1m": True,
    },
    "technical_persistence_score": {"trend_persistence_252": True},
    "technical_risk_score": {
        "downside_volatility_252": False,
        "max_drawdown_252": True,
    },
    "technical_volume_score": {"volume_confirmation_20_to_252": True},
}


def _numeric(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _winsorized_rank(
    values: pd.Series,
    higher_is_better: bool,
    groups: pd.Series | None = None,
) -> pd.Series:
    """Return stable [0, 1] percentile scores with neutral missing values."""
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def rank_group(group_values: pd.Series) -> pd.Series:
        valid = group_values.dropna()
        if len(valid) < 3:
            return pd.Series(np.nan, index=group_values.index, dtype=float)
        lower = valid.quantile(0.05)
        upper = valid.quantile(0.95)
        clipped = group_values.clip(lower=lower, upper=upper)
        ranked = clipped.rank(method="average", pct=True)
        return ranked if higher_is_better else 1 - ranked + (1 / len(valid))

    global_rank = rank_group(values)
    if groups is None:
        score = global_rank
    else:
        score = values.groupby(groups, group_keys=False).apply(rank_group)
        score = score.reindex(values.index).fillna(global_rank)
    return score.clip(0, 1).fillna(0.5)


def _valid_positive(values: pd.Series) -> pd.Series:
    return values.where(values > 0)


def _add_ranked_factor(
    result: pd.DataFrame,
    factor_name: str,
    metrics: dict[str, tuple[pd.Series, bool]],
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    score_columns = []
    valid_columns = []
    for metric_name, (values, higher_is_better) in metrics.items():
        score_column = f"{metric_name}_score"
        result[score_column] = _winsorized_rank(values, higher_is_better, groups)
        score_columns.append(score_column)
        valid_columns.append(values.notna().rename(metric_name))

    result[factor_name] = result[score_columns].mean(axis=1)
    coverage = pd.concat(valid_columns, axis=1).mean(axis=1)
    return result, coverage


def calculate_fundamental_scores(
    data: pd.DataFrame,
    config: SelectionConfig,
) -> pd.DataFrame:
    """Calculate industry-relative, coverage-aware fundamental factor scores."""
    result = data.copy()
    groups = result.get("Industry", pd.Series("Unknown", index=result.index)).fillna("Unknown")

    market_cap = _numeric(result, "marketCap").where(lambda value: value > 0)
    free_cashflow_yield = _numeric(result, "freeCashflow") / market_cap
    operating_cashflow_yield = _numeric(result, "operatingCashflow") / market_cap

    factors = {
        "fundamental_profitability_score": {
            "returnOnEquity": (_numeric(result, "returnOnEquity"), True),
            "returnOnAssets": (_numeric(result, "returnOnAssets"), True),
            "profitMargins": (_numeric(result, "profitMargins"), True),
            "operatingMargins": (_numeric(result, "operatingMargins"), True),
        },
        "fundamental_growth_score": {
            "revenueGrowth": (_numeric(result, "revenueGrowth"), True),
            "earningsGrowth": (_numeric(result, "earningsGrowth"), True),
        },
        "fundamental_valuation_score": {
            "trailingPE": (_valid_positive(_numeric(result, "trailingPE")), False),
            "forwardPE": (_valid_positive(_numeric(result, "forwardPE")), False),
            "pegRatio": (_valid_positive(_numeric(result, "pegRatio")), False),
            "priceToBook": (_valid_positive(_numeric(result, "priceToBook")), False),
        },
        "fundamental_health_score": {
            "currentRatio": (_numeric(result, "currentRatio").where(lambda value: value >= 0), True),
            "quickRatio": (_numeric(result, "quickRatio").where(lambda value: value >= 0), True),
            "debtToEquity": (_numeric(result, "debtToEquity").where(lambda value: value >= 0), False),
        },
        "fundamental_cashflow_score": {
            "freeCashflowYield": (free_cashflow_yield, True),
            "operatingCashflowYield": (operating_cashflow_yield, True),
        },
    }

    coverage_parts = []
    for factor_name, metrics in factors.items():
        result, coverage = _add_ranked_factor(
            result,
            factor_name,
            metrics,
            groups,
        )
        coverage_parts.append(coverage.rename(factor_name))

    factor_columns = list(factors)
    general_score = result[factor_columns].mean(axis=1)
    financial_mask = groups.str.casefold().eq("financials")
    financial_columns = [
        "fundamental_profitability_score",
        "fundamental_growth_score",
        "fundamental_valuation_score",
    ]
    raw_score = general_score.where(
        ~financial_mask,
        result[financial_columns].mean(axis=1),
    )

    metric_coverage = pd.concat(coverage_parts, axis=1)
    general_coverage = metric_coverage.mean(axis=1)
    financial_coverage = metric_coverage[financial_columns].mean(axis=1)
    result["fundamental_coverage"] = general_coverage.where(
        ~financial_mask,
        financial_coverage,
    )
    result["fundamental_score_before_coverage_penalty"] = raw_score
    result["fundamental_score"] = (
        raw_score
        - config.missing_fundamental_penalty
        * (1 - result["fundamental_coverage"])
    ).clip(0, 1)
    return result


def calculate_technical_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate cross-sectional OHLCV technical factor scores."""
    result = data.copy()
    for factor_name, metrics in TECHNICAL_FACTORS.items():
        score_columns = []
        for metric_name, higher_is_better in metrics.items():
            column_name = f"{metric_name}_technical_rank"
            result[column_name] = _winsorized_rank(
                _numeric(result, metric_name),
                higher_is_better,
            )
            score_columns.append(column_name)
        result[factor_name] = result[score_columns].mean(axis=1)

    weights = {
        "technical_trend_score": 0.35,
        "technical_momentum_score": 0.30,
        "technical_persistence_score": 0.15,
        "technical_risk_score": 0.10,
        "technical_volume_score": 0.10,
    }
    result["technical_score"] = sum(
        result[column] * weight for column, weight in weights.items()
    )
    return result


def combine_scores(
    data: pd.DataFrame,
    config: SelectionConfig,
    incumbent_tickers: set[str] | None = None,
) -> pd.DataFrame:
    result = data.copy()
    incumbent_tickers = incumbent_tickers or set()
    result["incumbent"] = result["Ticker"].isin(incumbent_tickers)
    result["combined_score"] = (
        config.fundamental_weight * result["fundamental_score"]
        + config.technical_weight * result["technical_score"]
    )
    result["selection_priority"] = (
        result["combined_score"]
        + result["incumbent"].astype(float) * config.incumbent_bonus
    )
    return result


def select_with_sector_controls(
    candidates: pd.DataFrame,
    config: SelectionConfig,
) -> list[str]:
    """Select top tickers with soft sector floors and hard sector caps."""
    if candidates.empty:
        return []

    ordered = candidates.sort_values(
        ["selection_priority", "Ticker"],
        ascending=[False, True],
    )
    sector_cap = max(
        config.min_sector_count,
        int(np.ceil(config.top_n * config.max_sector_share)),
    )
    selected: list[str] = []
    selected_set: set[str] = set()
    sector_counts: dict[str, int] = {}

    for _, sector_rows in ordered.groupby("Industry", sort=True):
        for row in sector_rows.head(config.min_sector_count).itertuples():
            if len(selected) >= config.top_n:
                break
            selected.append(row.Ticker)
            selected_set.add(row.Ticker)
            sector = row.Industry
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    for row in ordered.itertuples():
        if len(selected) >= config.top_n:
            break
        if row.Ticker in selected_set:
            continue
        sector = row.Industry
        if sector_counts.get(sector, 0) >= sector_cap:
            continue
        selected.append(row.Ticker)
        selected_set.add(row.Ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    return selected
