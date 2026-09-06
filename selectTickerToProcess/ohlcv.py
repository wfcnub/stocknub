from pathlib import Path

import numpy as np
import pandas as pd

from selectTickerToProcess.config import SelectionConfig


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return numerator / denominator


def _read_auxiliary_overlap(
    auxiliary_path: Path | None,
    recent_dates: pd.Series,
) -> tuple[bool, float]:
    if auxiliary_path is None or not auxiliary_path.is_file():
        return False, 0.0

    try:
        auxiliary_dates = pd.read_csv(auxiliary_path, usecols=["Date"])
        if auxiliary_dates.empty:
            return False, 0.0
        auxiliary_date_set = set(
            pd.to_datetime(auxiliary_dates["Date"], errors="coerce").dropna()
        )
        valid_recent_dates = set(recent_dates.dropna())
        if not valid_recent_dates:
            return False, 0.0
        return True, len(valid_recent_dates & auxiliary_date_set) / len(valid_recent_dates)
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False, 0.0


def calculate_ohlcv_features(
    ticker: str,
    ohlcv_path: Path,
    auxiliary_path: Path | None = None,
) -> dict:
    """Calculate inexpensive model-readiness and technical features from OHLCV."""
    base = {
        "Ticker": ticker,
        "ohlcv_readable": False,
        "ohlcv_error": "",
    }

    try:
        data = pd.read_csv(
            ohlcv_path,
            usecols=["Date", "Open", "High", "Low", "Close", "Volume"],
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        base["ohlcv_error"] = str(exc)
        return base

    if data.empty:
        base["ohlcv_error"] = "empty_ohlcv"
        return base

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")

    duplicate_dates = int(data["Date"].duplicated().sum())
    data = data.drop_duplicates(subset=["Date"], keep="last")
    close = data["Close"]
    volume = data["Volume"]
    returns = close.pct_change(fill_method=None)
    value_traded = close * volume

    recent_252 = data.tail(252)
    recent_returns_60 = returns.tail(60).dropna()
    valid_ohlcv = (
        data[["Open", "High", "Low", "Close", "Volume"]]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    invalid_price_rows = int(
        ((data[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)).sum()
    )
    invalid_volume_rows = int((data["Volume"] < 0).sum())
    bar_max = data[["Open", "Close", "High"]].abs().max(axis=1).clip(lower=1)
    bar_tolerance = bar_max * 1e-8
    invalid_bar_rows = int(
        (
            (data["High"] + bar_tolerance < data["Low"])
            | (
                data["High"] + bar_tolerance
                < data[["Open", "Close"]].max(axis=1)
            )
            | (
                data["Low"] - bar_tolerance
                > data[["Open", "Close"]].min(axis=1)
            )
        ).sum()
    )

    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    last_close = close.iloc[-1] if len(close) else np.nan
    last_ma50 = ma50.iloc[-1] if len(ma50) else np.nan
    last_ma200 = ma200.iloc[-1] if len(ma200) else np.nan
    previous_ma200 = ma200.iloc[-64] if len(ma200) >= 64 else np.nan

    momentum_6m_ex_1m = (
        _safe_ratio(close.iloc[-21], close.iloc[-127]) - 1
        if len(close) >= 127
        else np.nan
    )
    momentum_12m_ex_1m = (
        _safe_ratio(close.iloc[-21], close.iloc[-253]) - 1
        if len(close) >= 253
        else np.nan
    )
    downside_returns = returns.tail(252)
    downside_returns = downside_returns[downside_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    trailing_close = close.tail(252)
    max_drawdown = (trailing_close / trailing_close.cummax() - 1).min()

    above_ma200 = (close > ma200).tail(252)
    valid_ma200 = ma200.tail(252).notna()
    trend_persistence = (
        above_ma200[valid_ma200].mean() if valid_ma200.any() else np.nan
    )

    adv20 = value_traded.tail(20).median()
    adv60 = value_traded.tail(60).median()
    adv252 = value_traded.tail(252).median()
    volume_confirmation = _safe_ratio(adv20, adv252)
    auxiliary_exists, auxiliary_overlap = _read_auxiliary_overlap(
        auxiliary_path,
        recent_252["Date"],
    )

    base.update(
        {
            "ohlcv_readable": True,
            "ohlcv_rows": int(len(data)),
            "ohlcv_start_date": data["Date"].min(),
            "ohlcv_end_date": data["Date"].max(),
            "duplicate_date_rows": duplicate_dates,
            "invalid_ohlcv_rows": int((~valid_ohlcv).sum()),
            "invalid_price_rows": invalid_price_rows,
            "invalid_volume_rows": invalid_volume_rows,
            "invalid_bar_rows": invalid_bar_rows,
            "invalid_bar_ratio": invalid_bar_rows / len(data),
            "traded_days_252": int((recent_252["Volume"] > 0).sum()),
            "zero_volume_ratio_252": float((recent_252["Volume"] <= 0).mean()),
            "unchanged_close_ratio_60": float(
                (recent_returns_60 == 0).mean() if len(recent_returns_60) else 1.0
            ),
            "median_value_traded_20": adv20,
            "median_value_traded_60": adv60,
            "median_value_traded_252": adv252,
            "last_close": last_close,
            "distance_from_ma50": _safe_ratio(last_close, last_ma50) - 1,
            "distance_from_ma200": _safe_ratio(last_close, last_ma200) - 1,
            "ma200_slope_63": _safe_ratio(last_ma200, previous_ma200) - 1,
            "momentum_6m_ex_1m": momentum_6m_ex_1m,
            "momentum_12m_ex_1m": momentum_12m_ex_1m,
            "trend_persistence_252": trend_persistence,
            "downside_volatility_252": downside_volatility,
            "max_drawdown_252": max_drawdown,
            "volume_confirmation_20_to_252": volume_confirmation,
            "auxiliary_data_exists": auxiliary_exists,
            "auxiliary_overlap_ratio_252": auxiliary_overlap,
        }
    )
    return base


def add_ohlcv_eligibility(
    features: pd.DataFrame,
    config: SelectionConfig,
) -> pd.DataFrame:
    """Apply deterministic data-quality and liquidity gates."""
    result = features.copy()
    valid_end_dates = pd.to_datetime(result["ohlcv_end_date"], errors="coerce")
    market_latest_date = (
        valid_end_dates.value_counts().idxmax() if valid_end_dates.notna().any() else pd.NaT
    )
    result["stale_calendar_days"] = (market_latest_date - valid_end_dates).dt.days

    reason_masks = [
        ("unreadable_ohlcv", ~result["ohlcv_readable"].fillna(False)),
        ("insufficient_history", result["ohlcv_rows"].fillna(0) < config.min_history_rows),
        (
            "stale_ohlcv",
            result["stale_calendar_days"].fillna(np.inf)
            > config.max_stale_calendar_days,
        ),
        ("duplicate_dates", result["duplicate_date_rows"].fillna(0) > 0),
        ("invalid_ohlcv", result["invalid_ohlcv_rows"].fillna(1) > 0),
        ("invalid_prices", result["invalid_price_rows"].fillna(1) > 0),
        ("invalid_volume", result["invalid_volume_rows"].fillna(1) > 0),
        (
            "invalid_price_bars",
            result["invalid_bar_ratio"].fillna(1) > config.max_invalid_bar_ratio,
        ),
        (
            "insufficient_traded_days",
            result["traded_days_252"].fillna(0) < config.min_traded_days_252,
        ),
        (
            "excess_zero_volume",
            result["zero_volume_ratio_252"].fillna(1)
            > config.max_zero_volume_ratio_252,
        ),
        (
            "excess_unchanged_close",
            result["unchanged_close_ratio_60"].fillna(1)
            > config.max_unchanged_close_ratio_60,
        ),
        (
            "insufficient_traded_value",
            result["median_value_traded_60"].fillna(0)
            < config.min_median_value_traded_60,
        ),
    ]

    if config.require_auxiliary_data:
        reason_masks.extend(
            [
                (
                    "missing_auxiliary_data",
                    ~result["auxiliary_data_exists"].fillna(False),
                ),
                (
                    "insufficient_auxiliary_overlap",
                    result["auxiliary_overlap_ratio_252"].fillna(0)
                    < config.min_auxiliary_overlap_ratio,
                ),
            ]
        )

    reasons = [[] for _ in range(len(result))]
    for reason, mask in reason_masks:
        for position in np.flatnonzero(mask.to_numpy(dtype=bool)):
            reasons[position].append(reason)

    result["ohlcv_eligibility_reasons"] = [";".join(items) for items in reasons]
    result["ohlcv_eligible"] = result["ohlcv_eligibility_reasons"].eq("")
    return result


def build_ohlcv_universe(
    ohlcv_dir: Path,
    auxiliary_dir: Path | None,
    config: SelectionConfig,
) -> pd.DataFrame:
    records = []
    for ohlcv_path in sorted(ohlcv_dir.glob("*.csv")):
        auxiliary_path = (
            auxiliary_dir / ohlcv_path.name if auxiliary_dir is not None else None
        )
        records.append(
            calculate_ohlcv_features(
                ticker=ohlcv_path.stem,
                ohlcv_path=ohlcv_path,
                auxiliary_path=auxiliary_path,
            )
        )

    if not records:
        return pd.DataFrame()
    return add_ohlcv_eligibility(pd.DataFrame(records), config)
