from functools import lru_cache

import numpy as np
import pandas as pd

from utils import paths


CONTEXT_SOURCE_FEATURES = (
    "Log Return 1D",
    "Realized Volatility 20D",
    "RSI Value",
    "Volume ZScore 20D",
)


@lru_cache(maxsize=1)
def _ticker_industry_mapping() -> dict[str, str]:
    mapping = pd.read_csv(
        paths.get_selected_ticker_and_industry_list_path(),
        usecols=["Ticker", "Industry"],
    ).drop_duplicates("Ticker")
    return mapping.set_index("Ticker")["Industry"].fillna("Unknown").to_dict()


def attach_ticker_metadata(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    result = data.copy()
    result["Ticker"] = ticker
    result["Industry"] = _ticker_industry_mapping().get(ticker, "Unknown")
    return result


@lru_cache(maxsize=1)
def load_cross_sectional_context() -> pd.DataFrame:
    """Build point-in-time market and industry medians from same-date features."""
    frames = []
    mapping = _ticker_industry_mapping()
    requested = ["Date", *CONTEXT_SOURCE_FEATURES]
    for file_path in sorted(paths.get_label_dir().glob("*.csv")):
        header = pd.read_csv(file_path, nrows=0).columns
        available = [column for column in requested if column in header]
        if "Date" not in available:
            continue
        frame = pd.read_csv(file_path, usecols=available)
        for column in CONTEXT_SOURCE_FEATURES:
            if column not in frame:
                frame[column] = np.nan
        frame["Ticker"] = file_path.stem
        frame["Industry"] = mapping.get(file_path.stem, "Unknown")
        frames.append(frame[["Date", "Ticker", "Industry", *CONTEXT_SOURCE_FEATURES]])

    if not frames:
        return pd.DataFrame(columns=["Date", "Industry"])

    combined = pd.concat(frames, ignore_index=True)
    market = combined.groupby("Date", sort=False)[list(CONTEXT_SOURCE_FEATURES)].median()
    market.columns = [f"Market Median {column}" for column in market.columns]
    industry = combined.groupby(["Date", "Industry"], sort=False)[
        list(CONTEXT_SOURCE_FEATURES)
    ].median()
    industry.columns = [f"Industry Median {column}" for column in industry.columns]
    return market.reset_index().merge(industry.reset_index(), on="Date", how="left")


def add_cross_sectional_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add same-date market/industry context without using future observations."""
    if "Ticker" not in data:
        raise ValueError("Ticker metadata is required for cross-sectional features")
    result = data.copy()
    if "Industry" not in result:
        result["Industry"] = result["Ticker"].map(_ticker_industry_mapping())
    result["Industry"] = result["Industry"].fillna("Unknown").astype(str)
    result["Ticker"] = result["Ticker"].fillna("Unknown").astype(str)

    context = load_cross_sectional_context()
    result = result.merge(context, on=["Date", "Industry"], how="left")
    added: list[str] = []
    for source in CONTEXT_SOURCE_FEATURES:
        market_column = f"Market Median {source}"
        industry_column = f"Industry Median {source}"
        market_relative = f"{source} Relative To Market"
        industry_relative = f"{source} Relative To Industry"
        if source not in result:
            result[source] = np.nan
        result[market_relative] = result[source] - result[market_column]
        result[industry_relative] = result[source] - result[industry_column]
        added.extend(
            [market_column, industry_column, market_relative, industry_relative]
        )
    return result, added


def select_usable_features(
    development_data: pd.DataFrame,
    candidate_features: list[str],
    categorical_features: list[str],
    max_missing_fraction: float,
) -> tuple[list[str], dict[str, str]]:
    """Remove train-only unusable and exactly duplicated numerical features."""
    selected: list[str] = []
    removed: dict[str, str] = {}
    numeric_selected: list[str] = []

    for column in candidate_features:
        if column not in development_data:
            removed[column] = "missing"
            continue
        if column in categorical_features:
            selected.append(column)
            continue
        series = development_data[column]
        if float(series.isna().mean()) > max_missing_fraction:
            removed[column] = "too_many_missing_values"
            continue
        if series.nunique(dropna=True) <= 1:
            removed[column] = "constant"
            continue
        duplicate_of = next(
            (
                existing
                for existing in numeric_selected
                if series.equals(development_data[existing])
            ),
            None,
        )
        if duplicate_of is not None:
            removed[column] = f"duplicate_of:{duplicate_of}"
            continue
        selected.append(column)
        numeric_selected.append(column)

    if not selected:
        raise ValueError("Feature hygiene removed every candidate feature")
    return selected, removed
