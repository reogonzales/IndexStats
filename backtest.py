import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).parent
_REQUIRED_COLS = {"Proj Range [40/60]", "hit_4060", "hit_3070", "hit_2080"}


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

    def proj_bounds(p_low, p_high):
        lo = prev_close * (1 + float(np.percentile(low_ret,  p_low)))
        hi = prev_close * (1 + float(np.percentile(high_ret, p_high)))
        return lo, hi

    lo40, hi60 = proj_bounds(40, 60)
    lo30, hi70 = proj_bounds(30, 70)
    lo20, hi80 = proj_bounds(20, 80)

    d = daily_df.index[loc]
    return {
        "Date":               d.date().isoformat(),
        "Close Price":        actual_close,
        "Proj Price":         proj_price,
        "%":                  pct,
        "Day Low - Day Hi":   f"${float(actual['Low']):,.2f} − ${float(actual['High']):,.2f}",
        "Proj Range [40/60]": f"${lo40:,.2f} − ${hi60:,.2f}",
        "Proj Range [30/70]": f"${lo30:,.2f} − ${hi70:,.2f}",
        "Proj Range [20/80]": f"${lo20:,.2f} − ${hi80:,.2f}",
        "hit_4060":           int(lo40 <= actual_close <= hi60),
        "hit_3070":           int(lo30 <= actual_close <= hi70),
        "hit_2080":           int(lo20 <= actual_close <= hi80),
    }


def load_or_update_backtest(index_label: str, daily_df: pd.DataFrame) -> pd.DataFrame:
    csv = _csv_path(index_label)
    if csv.exists():
        existing = pd.read_csv(csv)
        if not _REQUIRED_COLS.issubset(existing.columns):
            csv.unlink()
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    existing_dates = set(existing["Date"]) if not existing.empty else set()

    backtest_start = (pd.Timestamp.today() - pd.Timedelta(days=365)).normalize()
    yesterday = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize()
    target_dates = daily_df.loc[backtest_start:yesterday].index

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
