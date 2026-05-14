from datetime import date, timedelta
import pandas as pd
import streamlit as st
from polygon import RESTClient

# API key stored in .streamlit/secrets.toml as POLYGON_API_KEY
_API_KEY = st.secrets.get("POLYGON_API_KEY", "")

# Polygon ticker format for indices uses "I:" prefix
TICKER_MAP = {
    "SPX": "I:SPX",
    "NDX": "I:NDX",
    "RUT": "I:RUT",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
}

# Options trade on ETFs, not index symbols — map indices to ETF equivalents
OPTIONS_TICKER_MAP = {
    "SPX": "SPY",
    "NDX": "QQQ",
    "RUT": "IWM",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
}


def _client():
    return RESTClient(api_key=_API_KEY)


def fetch_ohlc(index_label: str, period: str = "1y") -> pd.DataFrame:
    ticker = TICKER_MAP[index_label]
    to_date = date.today()
    from_date = to_date - timedelta(days=366)

    aggs = _client().get_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=from_date,
        to=to_date,
        adjusted=True,
        limit=50000,
    )

    rows = [
        {
            "Date": pd.Timestamp(a.timestamp, unit="ms", tz="UTC").normalize(),
            "Open":  a.open,
            "High":  a.high,
            "Low":   a.low,
            "Close": a.close,
        }
        for a in aggs
    ]
    df = pd.DataFrame(rows).set_index("Date")
    df.index = df.index.tz_localize(None)
    return df.dropna()


def fetch_weekly_ohlc(daily_df: pd.DataFrame) -> pd.DataFrame:
    return daily_df.resample("W").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Avg=("Close", "mean"),
    ).dropna()


def fetch_21day_ohlc(daily_df: pd.DataFrame) -> pd.DataFrame:
    return daily_df.resample("21D").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Avg=("Close", "mean"),
    ).dropna()


def fetch_45day_ohlc(daily_df: pd.DataFrame) -> pd.DataFrame:
    return daily_df.resample("45D").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Avg=("Close", "mean"),
    ).dropna()


def fetch_options_chain(index_label: str):
    """Returns calls and puts DataFrames, or (None, None) if unavailable."""
    ticker = OPTIONS_TICKER_MAP[index_label]
    client = _client()

    # Get nearest expiry
    contracts = list(client.list_options_contracts(
        underlying_ticker=ticker,
        expired=False,
        limit=250,
        sort="expiration_date",
        order="asc",
    ))
    if not contracts:
        return None, None

    nearest_expiry = contracts[0].expiration_date

    # Snapshot gives bid/ask/IV — requires paid plan; returns empty on free tier
    try:
        snapshots = list(client.list_snapshot_options_chain(
            underlying_asset=ticker,
            expiration_date=nearest_expiry,
            limit=250,
        ))
    except Exception:
        return None, None

    if not snapshots:
        return None, None

    rows = []
    for s in snapshots:
        d = s.details
        g = s.greeks
        q = s.day
        rows.append({
            "strike":            d.strike_price if d else None,
            "option_type":       d.contract_type if d else None,
            "bid":               s.last_quote.bid if s.last_quote else None,
            "ask":               s.last_quote.ask if s.last_quote else None,
            "impliedVolatility": s.implied_volatility if hasattr(s, "implied_volatility") else None,
            "openInterest":      s.open_interest if hasattr(s, "open_interest") else None,
            "volume":            q.volume if q else None,
        })

    df = pd.DataFrame(rows)
    calls = df[df["option_type"] == "call"].drop(columns=["option_type"]).reset_index(drop=True)
    puts  = df[df["option_type"] == "put"].drop(columns=["option_type"]).reset_index(drop=True)
    return calls, puts
