import time
import yfinance as yf
import pandas as pd

TICKER_MAP = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "RUT": "^RUT",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
}

_RETRIES = 4
_RETRY_DELAY = 5  # seconds


def _retry(fn, *args, **kwargs):
    for attempt in range(_RETRIES):
        try:
            result = fn(*args, **kwargs)
            # yf.download swallows rate limit errors and returns empty DataFrame
            if isinstance(result, pd.DataFrame) and result.empty:
                raise RuntimeError("yfinance returned empty DataFrame (likely rate limited)")
            return result
        except Exception as e:
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_RETRY_DELAY * (attempt + 1))


def fetch_ohlc(index_label: str, period: str = "1y") -> pd.DataFrame:
    ticker = TICKER_MAP[index_label]
    df = _retry(yf.download, ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


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
    ticker = TICKER_MAP[index_label]
    tk = yf.Ticker(ticker)

    def _fetch():
        expirations = tk.options
        if not expirations:
            return None, None
        chain = tk.option_chain(expirations[0])
        return chain.calls, chain.puts

    return _retry(_fetch)
