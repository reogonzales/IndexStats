# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run app.py
```

Install dependencies:
```bash
pip install -r requirements.txt
```

No test suite exists. Validate changes by running the app and exercising the affected path manually.

## Architecture Overview

IndexStats is a single-page Streamlit app that fetches OHLC and options data for equity indices/ETFs, computes statistical metrics and price projections, and backtests those projections against historical data.

### Module responsibilities

| File | Role |
|---|---|
| `app.py` | Streamlit UI — layout, all `st.*` calls, wires every other module together |
| `data_yf.py` | yfinance data source — `fetch_ohlc`, `fetch_weekly_ohlc`, `fetch_21day_ohlc`, `fetch_45day_ohlc`, `fetch_options_chain` |
| `data_poly.py` | Polygon.io data source — identical public API to `data_yf`; free tier maps index labels to ETF proxies |
| `metrics.py` | Percentile table (`percentile_table`), RSI, MACD |
| `projections.py` | Forward price projection tables (`project_prices`) |
| `pricing.py` | Black-Scholes pricing, IV calculation, IV rank/percentile |
| `backtest.py` | Incremental backtest engine; persists results to CSV |

### Data flow

1. `app.py` calls `data_poly.fetch_ohlc` or `data_yf.fetch_ohlc` → 1-year daily OHLC DataFrame (DatetimeIndex, columns: Open/High/Low/Close).
2. Weekly/21-day/45-day frames are derived via `resample().agg(...)` — they add an `Avg` column (mean of daily closes for the period, **not** (H+L)/2).
3. `metrics.percentile_table` computes DHDC, DCDL, WHWA, WAWL, and performance percentiles from those frames.
4. `projections.project_prices` builds projection tables using **prior-close-relative returns**: `Low/prevClose − 1` and `High/prevClose − 1`.
5. `backtest.load_or_update_backtest` reads a CSV (`backtest_{period}_{index_label}.csv`), appends only new/missing dates, and writes back. The backtest applies the same projection logic as the live view.

### Sign conventions (critical)

- **DCDL** = `(Close − Low) / Close` — internally positive, **displayed as negative** in the percentile table.
- **WAWL** = `(Avg − Low) / Avg` — internally positive, **displayed as negative** in the percentile table.
- Do not flip these to show as positive or change how they are stored.

### Supported tickers

`SPX`, `NDX`, `RUT`, `SPY`, `QQQ`, `IWM`

Polygon free tier maps index labels (SPX/NDX/RUT) to their ETF proxies (SPY/QQQ/IWM). yfinance uses real index tickers (`^GSPC`, `^NDX`, `^RUT`).

### Configuration

- Polygon API key: `.streamlit/secrets.toml` → key `POLYGON_API_KEY`
- Static file serving (for PWA manifest): `.streamlit/config.toml` → `enableStaticServing = true`
- Streamlit data cache TTL: 1 hour (`@st.cache_data(ttl=3600)` in `app.py`)

### Backtest CSV format

Files: `backtest_{period}_{index_label}.csv` (e.g. `backtest_daily_SPX.csv`)  
Periods: `daily`, `5d`, `21d`, `45d`  
Required columns (checked on load; file is deleted and rebuilt if missing): `Proj Range [40/60]`, `hit_4060`, `hit_3070`, `hit_2080`
