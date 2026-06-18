import math
import numpy as np
import pandas as pd
from metrics import compute_streak

PAIRS = [(10, 90), (20, 80), (30, 70), (40, 60), (50, 50)]


def _make_projection_df(current_price, low_return, high_return, perf):
    proj_price = current_price * (1 + float(np.percentile(perf, 50)))
    rows = {}
    for p_low, p_high in PAIRS:
        proj_low  = current_price * (1 + float(np.percentile(low_return,  p_low)))
        proj_high = current_price * (1 + float(np.percentile(high_return, p_high)))
        rows[f"{p_low}% / {p_high}%"] = {
            "Proj Price": proj_price,
            "Proj Range": f"${proj_low:,.2f} − ${proj_high:,.2f}",
        }
    df = pd.DataFrame(rows).T
    df.index.name = "Percentile"
    return df


def project_prices(current_price: float, daily_df, weekly_df, df_21day, df_45day):
    def returns(df):
        low_ret  = (df["Low"]   / df["Close"].shift(1) - 1).dropna()
        high_ret = (df["High"]  / df["Close"].shift(1) - 1).dropna()
        perf     = df["Close"].pct_change().dropna()
        return low_ret, high_ret, perf

    tomorrow_df  = _make_projection_df(current_price, *returns(daily_df))
    five_day_df  = _make_projection_df(current_price, *returns(weekly_df))
    day21_df     = _make_projection_df(current_price, *returns(df_21day))
    day45_df     = _make_projection_df(current_price, *returns(df_45day))

    return tomorrow_df, five_day_df, day21_df, day45_df


def compute_tomorrow_probabilities(
    current_price: float,
    daily_df: pd.DataFrame,
    vix_close: float,
) -> dict:
    returns = daily_df["Close"].pct_change().dropna()
    MIN_WINDOW = 20
    MIN_HIST = MIN_WINDOW + 2

    # Realized vol (20-day)
    hist_vol = float(returns.iloc[-MIN_WINDOW:].std()) if len(returns) >= MIN_WINDOW else float(returns.std())
    hist_1sd = current_price * hist_vol
    hist_2sd = hist_1sd * 2

    # VIX-implied vol
    if vix_close > 0 and not math.isnan(vix_close):
        vix_vol = (vix_close / 100.0) / math.sqrt(252)
        vix_1sd = current_price * vix_vol
        vix_2sd = vix_1sd * 2
    else:
        vix_vol = vix_1sd = vix_2sd = float("nan")

    # Rolling hit rates (lookahead-free: roll_std[i] uses returns[i-20..i-1])
    hist_hit_1sd = hist_hit_2sd = float("nan")
    hist_sample_n = 0
    if len(returns) >= MIN_HIST:
        roll_std = returns.shift(1).rolling(MIN_WINDOW).std().dropna()
        aligned = returns.loc[roll_std.index]
        valid_mask = roll_std > 0
        valid_std = roll_std[valid_mask]
        aligned = aligned.loc[valid_std.index]
        n = len(aligned)
        if n > 0:
            hist_sample_n = n
            hist_hit_1sd = float((aligned.abs() <= valid_std).sum() / n)
            hist_hit_2sd = float((aligned.abs() <= 2 * valid_std).sum() / n)

    # Streak + conditional probabilities
    streak_len, streak_pct = compute_streak(daily_df)
    streak_dir = 1 if streak_len > 0 else (-1 if streak_len < 0 else 0)

    cond_p_up = cond_p_1sd = cond_p_2sd = None
    cond_n = 0
    MIN_COND = 10

    if streak_dir != 0 and len(returns) >= MIN_HIST:
        prior_sign = np.sign(returns.shift(1)).fillna(0)
        cond_returns = returns[prior_sign == streak_dir]
        cond_n = len(cond_returns)
        if cond_n >= MIN_COND:
            cond_p_up = float((cond_returns > 0).sum() / cond_n)
            cond_roll = returns.shift(1).rolling(MIN_WINDOW).std().loc[cond_returns.index].dropna()
            cond_ret_filtered = cond_returns.loc[cond_roll.index]
            if len(cond_ret_filtered) > 0:
                cond_p_1sd = float((cond_ret_filtered.abs() <= cond_roll).sum() / len(cond_ret_filtered))
                cond_p_2sd = float((cond_ret_filtered.abs() <= 2 * cond_roll).sum() / len(cond_ret_filtered))

    return {
        "hist_vol_daily":    hist_vol,
        "hist_1sd_dollar":   hist_1sd,
        "hist_2sd_dollar":   hist_2sd,
        "hist_1sd_pct":      hist_vol * 100,
        "hist_2sd_pct":      hist_vol * 200,
        "vix_vol_daily":     vix_vol,
        "vix_1sd_dollar":    vix_1sd,
        "vix_2sd_dollar":    vix_2sd,
        "vix_1sd_pct":       float("nan") if math.isnan(vix_vol) else vix_vol * 100,
        "vix_2sd_pct":       float("nan") if math.isnan(vix_vol) else vix_vol * 200,
        "hist_hit_1sd":      hist_hit_1sd,
        "hist_hit_2sd":      hist_hit_2sd,
        "hist_sample_n":     hist_sample_n,
        "streak_len":        streak_len,
        "streak_pct":        streak_pct,
        "cond_p_up":         cond_p_up,
        "cond_p_within_1sd": cond_p_1sd,
        "cond_p_within_2sd": cond_p_2sd,
        "cond_n":            cond_n,
    }
