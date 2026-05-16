import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).parent
_REQUIRED_COLS = {"Proj Range [40/60]", "hit_4060", "hit_3070", "hit_2080"}
_MIN_LOOKBACK = {"daily": 20, "5d": 20, "21d": 30, "45d": 50}


def _csv_path(index_label: str, period: str = "daily") -> Path:
    return _DATA_DIR / f"backtest_{period}_{index_label}.csv"


def _compute_row(ohlc_df: pd.DataFrame, loc: int, min_lookback: int = 20) -> dict:
    """1-day backtest row: lookback is data before loc; actual is at loc."""
    lookback = ohlc_df.iloc[:loc]
    if len(lookback) < min_lookback:
        return None

    prev_close = float(lookback["Close"].iloc[-1])
    low_ret  = (lookback["Low"]   / lookback["Close"].shift(1) - 1).dropna()
    high_ret = (lookback["High"]  / lookback["Close"].shift(1) - 1).dropna()
    perf     = lookback["Close"].pct_change().dropna()

    proj_price = prev_close * (1 + float(np.percentile(perf, 50)))

    actual = ohlc_df.iloc[loc]
    actual_close = float(actual["Close"])
    pct = (proj_price - actual_close) / proj_price * 100

    def proj_bounds(p_low, p_high):
        lo = prev_close * (1 + float(np.percentile(low_ret,  p_low)))
        hi = prev_close * (1 + float(np.percentile(high_ret, p_high)))
        return lo, hi

    lo40, hi60 = proj_bounds(40, 60)
    lo30, hi70 = proj_bounds(30, 70)
    lo20, hi80 = proj_bounds(20, 80)

    d = ohlc_df.index[loc]
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


def _compute_row_nd(daily_df: pd.DataFrame, loc_pred: int, n_days: int, min_lookback: int) -> dict:
    """N-day rolling backtest row.

    loc_pred: prediction day index (data up to this day is the lookback).
    Outcome date = daily_df.index[loc_pred + n_days].
    """
    lookback = daily_df.iloc[:loc_pred + 1]
    if len(lookback) < min_lookback:
        return None

    pred_close = float(daily_df.iloc[loc_pred]["Close"])

    perf     = (lookback["Close"] / lookback["Close"].shift(n_days) - 1).dropna()
    low_ret  = (lookback["Low"].rolling(n_days).min()  / lookback["Close"].shift(n_days) - 1).dropna()
    high_ret = (lookback["High"].rolling(n_days).max() / lookback["Close"].shift(n_days) - 1).dropna()

    if len(perf) < 2:
        return None

    proj_price   = pred_close * (1 + float(np.percentile(perf, 50)))
    actual_close = float(daily_df.iloc[loc_pred + n_days]["Close"])
    actual_low   = float(daily_df.iloc[loc_pred + 1:loc_pred + n_days + 1]["Low"].min())
    actual_high  = float(daily_df.iloc[loc_pred + 1:loc_pred + n_days + 1]["High"].max())
    pct = (proj_price - actual_close) / proj_price * 100

    def proj_bounds(p_low, p_high):
        lo = pred_close * (1 + float(np.percentile(low_ret,  p_low)))
        hi = pred_close * (1 + float(np.percentile(high_ret, p_high)))
        return lo, hi

    lo40, hi60 = proj_bounds(40, 60)
    lo30, hi70 = proj_bounds(30, 70)
    lo20, hi80 = proj_bounds(20, 80)

    d = daily_df.index[loc_pred + n_days]
    return {
        "Date":               d.date().isoformat(),
        "Close Price":        actual_close,
        "Proj Price":         proj_price,
        "%":                  pct,
        "Day Low - Day Hi":   f"${actual_low:,.2f} − ${actual_high:,.2f}",
        "Proj Range [40/60]": f"${lo40:,.2f} − ${hi60:,.2f}",
        "Proj Range [30/70]": f"${lo30:,.2f} − ${hi70:,.2f}",
        "Proj Range [20/80]": f"${lo20:,.2f} − ${hi80:,.2f}",
        "hit_4060":           int(lo40 <= actual_close <= hi60),
        "hit_3070":           int(lo30 <= actual_close <= hi70),
        "hit_2080":           int(lo20 <= actual_close <= hi80),
    }


def load_or_update_backtest(
    index_label: str,
    daily_df: pd.DataFrame,
    period: str = "daily",
    n_days: int = 1,
) -> pd.DataFrame:
    csv = _csv_path(index_label, period)
    if csv.exists():
        existing = pd.read_csv(csv)
        if not _REQUIRED_COLS.issubset(existing.columns):
            csv.unlink()
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    existing_dates = set(existing["Date"]) if not existing.empty else set()
    min_lookback = _MIN_LOOKBACK.get(period, 20)

    backtest_start = (pd.Timestamp.today() - pd.Timedelta(days=365)).normalize()
    through_date   = pd.Timestamp.today().normalize()

    new_rows = []

    if n_days == 1:
        # Daily: iterate over outcome dates directly
        for d in daily_df.loc[backtest_start:through_date].index:
            if d.date().isoformat() in existing_dates:
                continue
            loc = daily_df.index.get_loc(d)
            row = _compute_row(daily_df, loc, min_lookback)
            if row:
                new_rows.append(row)
    else:
        # N-day rolling: iterate over prediction indices
        n_total = len(daily_df)
        for loc_pred in range(n_total - n_days):
            outcome_date = daily_df.index[loc_pred + n_days]
            if outcome_date < backtest_start or outcome_date > through_date:
                continue
            if outcome_date.date().isoformat() in existing_dates:
                continue
            row = _compute_row_nd(daily_df, loc_pred, n_days, min_lookback)
            if row:
                new_rows.append(row)

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_csv(csv, index=False)
        return combined
    return existing
