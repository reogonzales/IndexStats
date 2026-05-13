import numpy as np
import pandas as pd

PAIRS = [(10, 90), (20, 80), (40, 60), (50, 50)]


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
