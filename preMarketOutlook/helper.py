import json
import pandas as pd
import yfinance as yf
from pathlib import Path
from curl_cffi import requests
from datetime import datetime, timedelta

VIX_TICKER = "^VIX"
USDIDR_TICKER = "USDIDR=X"
SP500_TICKER = "^GSPC"
NIKKEI_TICKER = "^N225"

LOOKBACK_DAYS = 252

def _create_session() -> requests.Session:
    """
    (Internal Helper) Create a curl_cffi session that impersonates a real browser.

    Returns:
        requests.Session: A session configured for yfinance usage
    """
    return requests.Session(impersonate="chrome123")


def _fetch_indicator_history(
    ticker_symbol: str,
    lookback_days: int = LOOKBACK_DAYS,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    (Internal Helper) Fetch historical daily close data for a given yfinance ticker.

    Args:
        ticker_symbol (str): The yfinance ticker symbol (e.g. '^VIX', 'USDIDR=X')
        lookback_days (int): Number of calendar days to look back
        session (requests.Session | None): Optional pre-created session

    Returns:
        pd.DataFrame: DataFrame with columns ['Date', 'Close'] sorted ascending by date
    """
    if session is None:
        session = _create_session()

    ticker = yf.Ticker(ticker_symbol, session=session)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    history = ticker.history(start=start_date, end=end_date)

    if history.empty:
        raise ValueError(
            f"No data returned for {ticker_symbol} over the last {lookback_days} days."
        )

    history.reset_index(inplace=True)
    history["Date"] = history["Date"].dt.date

    return history[["Date", "Close"]].copy()


def _fetch_intraday_price(
    ticker_symbol: str,
    session: requests.Session | None = None,
) -> tuple:
    """
    (Internal Helper) Fetch the latest intraday price for a currently-trading index.

    Uses 1-minute interval data for the current day. If intraday data is not
    available (e.g. market hasn't opened yet), falls back to the most recent
    daily close.

    Args:
        ticker_symbol (str): The yfinance ticker symbol (e.g. '^N225')
        session (requests.Session | None): Optional pre-created session

    Returns:
        tuple: (timestamp_str, latest_price, is_live)
            - timestamp_str: human-readable timestamp of the data point
            - latest_price: the most recent price
            - is_live: True if the price is from today's intraday session
    """
    if session is None:
        session = _create_session()

    ticker = yf.Ticker(ticker_symbol, session=session)

    # Try 1-minute intraday data for today
    intraday = ticker.history(period="1d", interval="1m")

    if not intraday.empty:
        latest = intraday.iloc[-1]
        timestamp_str = str(latest.name)
        return timestamp_str, float(latest["Close"]), True

    # Fallback to most recent daily close
    daily = ticker.history(period="5d")
    if daily.empty:
        raise ValueError(f"No data available for {ticker_symbol}.")

    latest = daily.iloc[-1]
    timestamp_str = str(latest.name.date())
    return timestamp_str, float(latest["Close"]), False


def _get_latest_value(history: pd.DataFrame) -> tuple:
    """
    (Internal Helper) Extract the most recent close value and its date from a history df.

    Args:
        history (pd.DataFrame): DataFrame with 'Date' and 'Close' columns

    Returns:
        tuple: (latest_date, latest_close)
    """
    latest_row = history.iloc[-1]
    return latest_row["Date"], float(latest_row["Close"])


def _compute_percentile_rank(history: pd.DataFrame, value: float) -> float:
    """
    (Internal Helper) Compute the percentile rank of a value within a historical series.

    Args:
        history (pd.DataFrame): DataFrame with a 'Close' column
        value (float): The value to rank

    Returns:
        float: Percentile rank between 0 and 100
    """
    closes = history["Close"].values
    rank = (closes < value).sum() / len(closes) * 100
    return round(rank, 2)


def _classify_vix(value: float) -> dict:
    """
    (Internal Helper) Classify the VIX level into a sentiment tier.

    Tiers:
        < 15    → Low Volatility    — market is calm, complacent
        15-20   → Normal            — typical market conditions
        20-25   → Elevated          — rising caution
        25-30   → High              — significant fear
        > 30    → Extreme           — panic / crisis territory

    Args:
        value (float): The current VIX level

    Returns:
        dict: Classification result with 'tier', 'sentiment', and 'description'
    """
    if value < 15:
        return {
            "tier": "Low Volatility",
            "sentiment": "Bullish",
            "description": "Market is calm and complacent — low fear among investors.",
        }
    elif value < 20:
        return {
            "tier": "Normal",
            "sentiment": "Neutral",
            "description": "Typical volatility — no unusual stress detected.",
        }
    elif value < 25:
        return {
            "tier": "Elevated",
            "sentiment": "Cautious",
            "description": "Volatility is rising — investors are becoming cautious.",
        }
    elif value < 30:
        return {
            "tier": "High",
            "sentiment": "Bearish",
            "description": "Significant fear in the market — risk-off behaviour likely.",
        }
    else:
        return {
            "tier": "Extreme",
            "sentiment": "Very Bearish",
            "description": "Panic-level volatility — markets in crisis mode.",
        }


def _classify_usdidr_change(
    current_value: float, history: pd.DataFrame
) -> dict:
    """
    (Internal Helper) Classify the USD/IDR exchange rate based on recent movement
    relative to historical context.

    A rising USD/IDR signals Rupiah weakening (capital outflow pressure),
    which is typically negative for the Indonesian equity market.

    Args:
        current_value (float): Latest USD/IDR exchange rate
        history (pd.DataFrame): Historical data with 'Close' column

    Returns:
        dict: Classification result with 'change_pct', 'tier', 'sentiment',
              and 'description'
    """
    prev_close = float(history["Close"].iloc[-2])
    change_pct = round((current_value - prev_close) / prev_close * 100, 4)

    sma_20 = history["Close"].tail(20).mean()
    vs_sma_20_pct = round((current_value - sma_20) / sma_20 * 100, 4)

    if change_pct <= -0.3:
        tier = "Strong IDR"
        sentiment = "Bullish"
        description = (
            f"Rupiah strengthened significantly ({change_pct:+.2f}% vs prev close) "
            f"— positive signal for IDX equity inflows."
        )
    elif change_pct <= -0.05:
        tier = "Mild IDR Strengthening"
        sentiment = "Slightly Bullish"
        description = (
            f"Rupiah modestly stronger ({change_pct:+.2f}% vs prev close) "
            f"— slightly supportive for IDX."
        )
    elif change_pct <= 0.05:
        tier = "Stable"
        sentiment = "Neutral"
        description = (
            f"USD/IDR is flat ({change_pct:+.2f}% vs prev close) "
            f"— no directional pressure from FX."
        )
    elif change_pct <= 0.3:
        tier = "Mild IDR Weakening"
        sentiment = "Slightly Bearish"
        description = (
            f"Rupiah mildly weaker ({change_pct:+.2f}% vs prev close) "
            f"— minor headwind for IDX."
        )
    else:
        tier = "Sharp IDR Weakening"
        sentiment = "Bearish"
        description = (
            f"Rupiah weakened sharply ({change_pct:+.2f}% vs prev close) "
            f"— risk-off signal, potential foreign outflows from IDX."
        )

    return {
        "change_pct": change_pct,
        "vs_sma_20_pct": vs_sma_20_pct,
        "tier": tier,
        "sentiment": sentiment,
        "description": description,
    }


def _classify_index_change(index_name: str, current_value: float, history: pd.DataFrame) -> dict:
    """
    (Internal Helper) Classify a stock index based on its daily percentage change.

    Tiers:
        <= -2.0%  → Sharp Decline   — Very Bearish
        <= -0.5%  → Decline          — Bearish
        <= -0.1%  → Mild Decline     — Slightly Bearish
        <= +0.1%  → Flat             — Neutral
        <= +0.5%  → Gain             — Slightly Bullish
        >  +0.5%  → Strong Rally     — Bullish

    Args:
        index_name (str): Human-readable name (e.g. 'S&P 500', 'Nikkei 225')
        current_value (float): The latest index value
        history (pd.DataFrame): Historical data with 'Close' column (at least 2 rows)

    Returns:
        dict: Classification with 'change_pct', 'tier', 'sentiment', 'description'
    """
    prev_close = float(history["Close"].iloc[-2])
    change_pct = round((current_value - prev_close) / prev_close * 100, 4)

    if change_pct <= -2.0:
        tier = "Sharp Decline"
        sentiment = "Very Bearish"
        description = (
            f"{index_name} dropped sharply ({change_pct:+.2f}%) "
            f"— strong risk-off signal for emerging markets."
        )
    elif change_pct <= -0.5:
        tier = "Decline"
        sentiment = "Bearish"
        description = (
            f"{index_name} declined ({change_pct:+.2f}%) "
            f"— negative sentiment may spill over to IDX."
        )
    elif change_pct <= -0.1:
        tier = "Mild Decline"
        sentiment = "Slightly Bearish"
        description = (
            f"{index_name} edged lower ({change_pct:+.2f}%) "
            f"— mild caution, limited directional impact."
        )
    elif change_pct <= 0.1:
        tier = "Flat"
        sentiment = "Neutral"
        description = (
            f"{index_name} was flat ({change_pct:+.2f}%) "
            f"— no clear directional cue."
        )
    elif change_pct <= 0.5:
        tier = "Gain"
        sentiment = "Slightly Bullish"
        description = (
            f"{index_name} gained modestly ({change_pct:+.2f}%) "
            f"— mildly supportive for regional sentiment."
        )
    else:
        tier = "Strong Rally"
        sentiment = "Bullish"
        description = (
            f"{index_name} rallied ({change_pct:+.2f}%) "
            f"— positive risk appetite, likely tailwind for IDX."
        )

    return {
        "change_pct": change_pct,
        "tier": tier,
        "sentiment": sentiment,
        "description": description,
    }


def _derive_overall_outlook(sentiments: dict) -> dict:
    """
    (Internal Helper) Combine all indicator sentiments into a single market outlook.

    The scoring system assigns numeric weights to each sentiment level and averages them
    to produce a composite score.

    Args:
        sentiments (dict): Mapping of indicator name to its sentiment string.
                           e.g. {"VIX": "Bearish", "USD/IDR": "Neutral", ...}

    Returns:
        dict: Composite outlook with 'composite_score', 'outlook', and 'rationale'
    """
    sentiment_scores = {
        "Very Bearish": -2,
        "Bearish": -1,
        "Slightly Bearish": -0.5,
        "Cautious": -0.5,
        "Neutral": 0,
        "Slightly Bullish": 0.5,
        "Bullish": 1,
    }

    individual_scores = {}
    for name, sentiment in sentiments.items():
        individual_scores[name] = sentiment_scores.get(sentiment, 0)

    composite = sum(individual_scores.values()) / len(individual_scores)

    if composite >= 0.5:
        outlook = "Bullish"
    elif composite >= 0:
        outlook = "Neutral-to-Bullish"
    elif composite >= -0.5:
        outlook = "Neutral-to-Bearish"
    else:
        outlook = "Bearish"

    breakdown_parts = [
        f"{name} is {sentiments[name]} (score {score:+.1f})"
        for name, score in individual_scores.items()
    ]
    rationale = (
        ", ".join(breakdown_parts)
        + f". Composite score: {composite:+.2f} → Overall outlook: {outlook}."

    )

    return {
        "composite_score": round(composite, 2),
        "outlook": outlook,
        "rationale": rationale,
    }


def _save_outlook_to_json(outlook: dict) -> None:
    """
    Save the pre-market outlook dictionary as a JSON file under data/pre_market_outlook/.

    The file is saved with the current date as the filename (e.g. 2026-04-11.json),
    overwriting any previous run for the same day.

    Args:
        outlook (dict): The outlook dictionary produced by generate_pre_market_outlook()
    """
    serializable = json.loads(
        json.dumps(outlook, default=str)
    )

    output_path = Path("data/pre_market_outlook").with_suffix(".json")

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Pre-market outlook saved to {output_path}")