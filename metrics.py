import numpy as np
import pandas as pd

PERCENTILES = [10, 20, 40, 50, 60, 70, 80, 90]


def compute_daily_metrics(daily_df: pd.DataFrame) -> pd.Series:
    dhdc = (daily_df["High"] - daily_df["Close"]) / daily_df["Close"]
    dcdl = (daily_df["Close"] - daily_df["Low"]) / daily_df["Close"]
    return dhdc, dcdl


def compute_weekly_metrics(weekly_df: pd.DataFrame):
    whwa = (weekly_df["High"] - weekly_df["Avg"]) / weekly_df["Avg"]
    wawl = (weekly_df["Avg"] - weekly_df["Low"]) / weekly_df["Avg"]
    return whwa, wawl


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def percentile_table(daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    dhdc, dcdl = compute_daily_metrics(daily_df)
    whwa, wawl = compute_weekly_metrics(weekly_df)

    daily_perf = daily_df["Close"].pct_change()
    weekly_perf = weekly_df["Close"].pct_change()

    rows = {}
    for name, series in [
        ("DHDC", dhdc), ("DCDL", -dcdl), ("WHWA", whwa), ("WAWL", -wawl),
        ("Daily Performance", daily_perf), ("Weekly Performance", weekly_perf),
    ]:
        rows[name] = {f"{p}%": np.percentile(series.dropna(), p) for p in PERCENTILES}

    df = pd.DataFrame(rows).T
    df.index.name = "Metric"
    return df
