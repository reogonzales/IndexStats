import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call") -> float:
    if T <= 0 or market_price <= 0:
        return float("nan")
    intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    if market_price <= intrinsic:
        return float("nan")
    try:
        iv = brentq(
            lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price,
            1e-6, 10.0, xtol=1e-6, maxiter=200,
        )
        return iv
    except (ValueError, RuntimeError):
        return float("nan")


def iv_rank(iv_series) -> float:
    current = iv_series.iloc[-1]
    low, high = iv_series.min(), iv_series.max()
    if high == low:
        return float("nan")
    return (current - low) / (high - low) * 100


def iv_percentile(iv_series) -> float:
    current = iv_series.iloc[-1]
    return (iv_series < current).mean() * 100
