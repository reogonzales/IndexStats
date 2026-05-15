import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data_yf
import data_poly
from metrics import percentile_table, compute_rsi, compute_macd
from pricing import black_scholes, iv_rank, iv_percentile
from projections import project_prices


@st.cache_data(ttl=3600)
def _fetch_ohlc_yf(index_label):
    return data_yf.fetch_ohlc(index_label)


@st.cache_data(ttl=3600)
def _fetch_options_yf(index_label):
    return data_yf.fetch_options_chain(index_label)


@st.cache_data(ttl=3600)
def _fetch_ohlc_poly(index_label):
    return data_poly.fetch_ohlc(index_label)


@st.cache_data(ttl=3600)
def _fetch_options_poly(index_label):
    return data_poly.fetch_options_chain(index_label)


st.set_page_config(page_title="IndexStats", layout="wide")
st.markdown(
    """
    <link rel="manifest" href="/app/static/manifest.json">
    <style>
        /* Tighter side padding on small screens */
        @media (max-width: 768px) {
            .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
        }
        /* Horizontal scroll for wide tables on mobile */
        .stDataFrame { overflow-x: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("IndexStats — Index Options Statistics")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    index_label = st.selectbox("Select Index", ["SPX", "NDX", "RUT", "SPY", "QQQ", "IWM"])
with col2:
    source = st.radio("Data Source", ["Polygon.io", "yfinance"], horizontal=False)
with col3:
    st.write("")
    st.write("")
    run = st.button("Calculate", type="primary", use_container_width=True)

_use_polygon = (source == "Polygon.io")
_ETF_PROXY = {"SPX": "SPY", "NDX": "QQQ", "RUT": "IWM"}

if run:
    if _use_polygon and index_label in _ETF_PROXY:
        st.caption(f"Note: {index_label} uses {_ETF_PROXY[index_label]} (ETF proxy) — index data requires a paid Polygon plan.")
    with st.spinner(f"Fetching data for {index_label}…"):
        daily_df = _fetch_ohlc_poly(index_label) if _use_polygon else _fetch_ohlc_yf(index_label)
        weekly_df = data_poly.fetch_weekly_ohlc(daily_df)
        df_21day  = data_poly.fetch_21day_ohlc(daily_df)
        df_45day  = data_poly.fetch_45day_ohlc(daily_df)

    try:
        calls_df, puts_df = (
            _fetch_options_poly(index_label) if _use_polygon
            else _fetch_options_yf(index_label)
        )
    except Exception:
        calls_df, puts_df = None, None

    current_price = float(daily_df["Close"].iloc[-1])

    # ── Current price header ─────────────────────────────────────────────────
    st.metric("Current Price", f"${current_price:,.2f}")

    # ── Price + RSI + MACD chart ──────────────────────────────────────────────
    rsi = compute_rsi(daily_df["Close"])
    macd_line, signal_line, histogram = compute_macd(daily_df["Close"])

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.03,
        subplot_titles=("Price", "RSI (14)", "MACD (12, 26, 9)"),
    )

    # Price
    fig.add_trace(
        go.Scatter(x=daily_df.index, y=daily_df["Close"], name="Price",
                   line=dict(color="#2196F3", width=1.5)),
        row=1, col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=rsi.index, y=rsi, name="RSI",
                   line=dict(color="#FF9800", width=1.5)),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)

    # MACD line + signal
    fig.add_trace(
        go.Scatter(x=macd_line.index, y=macd_line, name="MACD",
                   line=dict(color="#2196F3", width=1.5)),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=signal_line.index, y=signal_line, name="Signal",
                   line=dict(color="#FF9800", width=1.5)),
        row=3, col=1,
    )
    # Histogram bars (green above zero, red below)
    colors = ["#26A69A" if v >= 0 else "#EF5350" for v in histogram]
    fig.add_trace(
        go.Bar(x=histogram.index, y=histogram, name="Histogram",
               marker_color=colors, opacity=0.6),
        row=3, col=1,
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_layout(height=500, showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # ── Metric percentile table ───────────────────────────────────────────────
    st.subheader("Daily & Weekly Range Metrics — Percentile Table")
    pct_table = percentile_table(daily_df, weekly_df)
    st.dataframe(
        pct_table.style.format("{:.4%}"),
        use_container_width=True,
    )
    with st.expander("Metric definitions"):
        st.markdown(
            "### Daily Metrics\n"
            "- **DHDC** (Daily High − Daily Close) / Daily Close — how far the intraday high reached *above* the closing price. Always positive. Higher percentiles = larger upside wicks.\n"
            "- **DCDL** (Daily Close − Daily Low) / Daily Close — how far the intraday low fell *below* the closing price. Displayed as negative (downside). Higher absolute percentiles = larger downside wicks.\n"
            "- **Daily Performance** — day-over-day % change in closing price (close-to-close return). Negative values mean the day closed lower than the prior day.\n"
            "\n"
            "### Weekly Metrics\n"
            "- **WHWA** (Weekly High − Weekly Avg) / Weekly Avg — how far the week's intraday high reached above the weekly average close. Always positive.\n"
            "- **WAWL** (Weekly Avg − Weekly Low) / Weekly Avg — how far the week's intraday low fell below the weekly average close. Displayed as negative.\n"
            "- **Weekly Avg** = mean of that week's daily closing prices (Monday–Friday), NOT (High + Low) / 2.\n"
            "- **Weekly Performance** — week-over-week % change in the weekly closing price.\n"
            "\n"
            "### Percentile Table\n"
            "- Percentiles shown: 10%, 20%, 40%, 50%, 60%, 70%, 80%, 90% — computed from 1 year of historical data.\n"
            "- **10%** = only 10% of historical days had a value this low or lower (rare downside).\n"
            "- **90%** = only 10% of historical days exceeded this value (rare upside).\n"
            "- **50%** = median — half of historical days were above, half below.\n"
            "\n"
            "### Price Projections\n"
            "- **Proj Price** = current price × (1 + median historical return for that period). Represents the most typical expected closing price if the period performs at its historical median.\n"
            "- **Proj % Change** = (Proj Price − current price) / current price. How much price is projected to move from today's close to the target date.\n"
            "- **Proj Range** = intraday low–high band. The row label (e.g. 10%/90%) means ~80% of historical periods had their intraday low stay above Proj Low AND their intraday high stay below Proj High.\n"
            "- Ranges are computed from prior-close-relative returns: Low/prev Close − 1 and High/prev Close − 1.\n"
        )
    st.divider()

    # ── Price projections ─────────────────────────────────────────────────────
    st.subheader("Price Projections")
    tomorrow_df, five_day_df, day21_df, day45_df = project_prices(
        current_price, daily_df, weekly_df, df_21day, df_45day
    )

    def show_proj_table(df, label, base_price):
        proj_price = float(df["Proj Price"].iloc[0])
        pct = (proj_price / base_price - 1) * 100
        sign = "+" if pct >= 0 else ""
        header = f"{label} — Proj Price: ${proj_price:,.2f} ({sign}{pct:.2f}%)"
        with st.expander(header, expanded=False):
            st.caption(f"Current Price: ${base_price:,.2f}")
            st.dataframe(df.drop(columns=["Proj Price"]), use_container_width=True)

    from datetime import date, timedelta
    today = date.today()
    # Next business day for Tomorrow
    tomorrow_date = today + timedelta(days=1)
    if tomorrow_date.weekday() == 5:
        tomorrow_date += timedelta(days=2)
    elif tomorrow_date.weekday() == 6:
        tomorrow_date += timedelta(days=1)

    show_proj_table(tomorrow_df, f"Tomorrow ({tomorrow_date.strftime('%m/%d/%Y')})",          current_price)
    show_proj_table(five_day_df, f"5 Days Out ({(today + timedelta(days=5)).strftime('%m/%d/%Y')})",   current_price)
    show_proj_table(day21_df,    f"21 Days Out ({(today + timedelta(days=21)).strftime('%m/%d/%Y')})", current_price)
    show_proj_table(day45_df,    f"45 Days Out ({(today + timedelta(days=45)).strftime('%m/%d/%Y')})", current_price)
    st.divider()

    # ── Back Test Results ─────────────────────────────────────────────────────
    import backtest as _bt
    st.subheader("Back Test Results")
    with st.spinner("Loading backtest data…"):
        bt_df = _bt.load_or_update_backtest(index_label, daily_df)

    if bt_df.empty:
        st.info("No backtest data yet — run again tomorrow.")
    else:
        display_bt = (
            bt_df.sort_values("Date", ascending=False)
                 .rename(columns={"%": "Proj Err %"})
                 .reset_index(drop=True)
        )
        with st.expander(f"Daily ({len(display_bt)} rows, since 2026-01-01)", expanded=False):
            st.dataframe(
                display_bt.style.format({
                    "Close Price": "${:,.2f}",
                    "Proj Price":  "${:,.2f}",
                    "Proj Err %":  "{:+.2f}%",
                }),
                use_container_width=True,
            )
    st.divider()

    # ── IV & Options summary ──────────────────────────────────────────────────
    st.subheader("Implied Volatility & Options Summary")

    if calls_df is not None and not calls_df.empty:
        calls_df = calls_df.copy()
        calls_df["moneyness"] = (calls_df["strike"] - current_price).abs()
        atm_row = calls_df.loc[calls_df["moneyness"].idxmin()]

        iv_from_chain = float(atm_row.get("impliedVolatility", np.nan))
        T = 30 / 365
        r = 0.05
        atm_bs_price = black_scholes(current_price, float(atm_row["strike"]), T, r,
                                     iv_from_chain if not np.isnan(iv_from_chain) else 0.2)

        iv_series = calls_df["impliedVolatility"].dropna()
        iv_series = iv_series[iv_series > 0]

        iv_col1, iv_col2 = st.columns(2)
        with iv_col1:
            st.metric("ATM Strike", f"${float(atm_row['strike']):,.2f}")
        with iv_col2:
            st.metric("ATM IV (Chain)", f"{iv_from_chain * 100:.1f}%" if not np.isnan(iv_from_chain) else "N/A")
        iv_col3, iv_col4 = st.columns(2)
        with iv_col3:
            ivr = iv_rank(iv_series) if len(iv_series) > 1 else float("nan")
            st.metric("IV Rank", f"{ivr:.1f}%" if not np.isnan(ivr) else "N/A")
        with iv_col4:
            ivp = iv_percentile(iv_series) if len(iv_series) > 1 else float("nan")
            st.metric("IV Percentile", f"{ivp:.1f}%" if not np.isnan(ivp) else "N/A")

        st.markdown(f"**ATM Black-Scholes Call Price** (≈30 days, r=5%): **${atm_bs_price:,.2f}**")

        with st.expander("Full options chain (calls)"):
            display_cols = [c for c in ["strike", "bid", "ask", "impliedVolatility",
                                        "openInterest", "volume"] if c in calls_df.columns]
            st.dataframe(calls_df[display_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.info("Options chain not available for this index via yfinance.")
