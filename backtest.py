import numpy as np
import pandas as pd
from pathlib import Path

BACKTEST_START = pd.Timestamp("2026-01-01")
_DATA_DIR = Path(__file__).parent


def _csv_path(index_label: str) -> Path:
    return _DATA_DIR / f"backtest_daily_{index_label}.csv"


def _compute_row(daily_df: pd.DataFrame, loc: int) -> dict:
    lookback = daily_df.iloc[:loc]
    if len(lookback) < 20:
        return None

    prev_close = float(lookback["Close"].iloc[-1])
    low_ret  = (lookback["Low"]   / lookback["Close"].shift(1) - 1).dropna()
    high_ret = (lookback["High"]  / lookback["Close"].shift(1) - 1).dropna()
    perf     = lookback["Close"].pct_change().dropna()

    proj_price = prev_close * (1 + float(np.percentile(perf, 50)))

    actual = daily_df.iloc[loc]
    actual_close = float(actual["Close"])
    pct = (proj_price - actual_close) / proj_price * 100

    def fmt_range(p_low, p_high):
        lo = prev_close * (1 + float(np.percentile(low_ret,  p_low)))
        hi = prev_close * (1 + float(np.percentile(high_ret, p_high)))
        return f"${lo:,.2f} − ${hi:,.2f}"

    d = daily_df.index[loc]
    return {
        "Date":               d.date().isoformat(),
        "Close Price":        actual_close,
        "Proj Price":         proj_price,
        "%":                  pct,
        "Day Low - Day Hi":   f"${float(actual['Low']):,.2f} − ${float(actual['High']):,.2f}",
        "Proj Range [30/70]": fmt_range(30, 70),
        "Proj Range [20/80]": fmt_range(20, 80),
    }


def load_or_update_backtest(index_label: str, daily_df: pd.DataFrame) -> pd.DataFrame:
    csv = _csv_path(index_label)
    existing = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    existing_dates = set(existing["Date"]) if not existing.empty else set()

    yesterday = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize()
    target_dates = daily_df.loc[BACKTEST_START:yesterday].index

    new_rows = []
    for d in target_dates:
        if d.date().isoformat() in existing_dates:
            continue
        loc = daily_df.index.get_loc(d)
        row = _compute_row(daily_df, loc)
        if row:
            new_rows.append(row)

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_csv(csv, index=False)
        return combined
    return existing
