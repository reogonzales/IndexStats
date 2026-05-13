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


def fetch_ohlc(index_label: str, period: str = "1y") -> pd.DataFrame:
    ticker = TICKER_MAP[index_label]
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    # Flatten MultiIndex columns that yfinance produces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def fetch_weekly_ohlc(daily_df: pd.DataFrame) -> pd.DataFrame:
    weekly = daily_df.resample("W").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Avg=("Close", "mean"),
    ).dropna()
    return weekly


def fetch_21day_ohlc(daily_df: pd.DataFrame) -> pd.DataFrame:
    result = daily_df.resample("21D").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Avg=("Close", "mean"),
    ).dropna()
    return result


def fetch_45day_ohlc(daily_df: pd.DataFrame) -> pd.DataFrame:
    result = daily_df.resample("45D").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Avg=("Close", "mean"),
    ).dropna()
    return result


def fetch_options_chain(index_label: str):
    ticker = TICKER_MAP[index_label]
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        return None, None
    nearest = expirations[0]
    chain = tk.option_chain(nearest)
    return chain.calls, chain.puts
