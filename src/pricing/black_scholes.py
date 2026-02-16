"""
Black-Scholes pricing for binary options (cash-or-nothing).

For Polymarket BTC 5-min markets:
- Up token = binary call: P_up = N(d₂)
- Down token = binary put: P_down = N(-d₂) = 1 - P_up
- d₂ = ln(S/K) / (σ√T)

At 5 minutes, T ≈ 9.513 × 10⁻⁶ years, so:
- Discount factor ≈ 1.0 (negligible)
- Drift term ≈ 0.0 (negligible)
- σ√T ≈ 0.15% at σ=50% annualized
"""

import math
from scipy.stats import norm


def bs_binary_price(S: float, K: float, sigma: float, T_years: float) -> tuple[float, float]:
    """
    Black-Scholes fair value for binary Up/Down tokens.
    
    Args:
        S: Current BTC spot price
        K: Strike price (window open price from Chainlink)
        sigma: Annualized volatility (e.g., 0.45 = 45%)
        T_years: Time to expiry in years
    
    Returns:
        (p_up, p_down) - Fair probabilities for Up and Down tokens
    """
    if T_years <= 0 or sigma <= 0:
        if S > K:
            return (1.0, 0.0)
        elif S < K:
            return (0.0, 1.0)
        else:
            return (0.5, 0.5)
    
    sigma_sqrt_t = sigma * math.sqrt(T_years)
    if sigma_sqrt_t < 1e-10:
        return (1.0, 0.0) if S > K else (0.0, 1.0) if S < K else (0.5, 0.5)
    
    d2 = math.log(S / K) / sigma_sqrt_t
    p_up = norm.cdf(d2)
    p_down = 1.0 - p_up
    
    return (p_up, p_down)


def bs_delta(S: float, K: float, sigma: float, T_years: float) -> tuple[float, float]:
    """
    Binary option delta: sensitivity to underlying price.
    Delta_binary = n(d₂) / (S × σ√T)
    
    Returns:
        (delta_up, delta_down) per $1 move in BTC
    """
    if T_years <= 0 or sigma <= 0:
        return (0.0, 0.0)
    
    sigma_sqrt_t = sigma * math.sqrt(T_years)
    if sigma_sqrt_t < 1e-10:
        return (0.0, 0.0)
    
    d2 = math.log(S / K) / sigma_sqrt_t
    pdf_d2 = norm.pdf(d2)
    delta_up = pdf_d2 / (S * sigma_sqrt_t)
    delta_down = -delta_up
    
    return (delta_up, delta_down)


def implied_probability(S: float, K: float, sigma: float, T_years: float) -> dict:
    """
    Full pricing output with all relevant metrics.
    
    Returns dict with:
        p_up, p_down: fair probabilities
        d2: the d₂ value
        sigma_sqrt_t: volatility window
        delta_up, delta_down: per-$1 sensitivities
        expected_5min_move: expected BTC move in dollars
    """
    p_up, p_down = bs_binary_price(S, K, sigma, T_years)
    delta_up, delta_down = bs_delta(S, K, sigma, T_years)
    
    sigma_sqrt_t = sigma * math.sqrt(T_years) if T_years > 0 else 0
    d2 = math.log(S / K) / sigma_sqrt_t if sigma_sqrt_t > 1e-10 else 0
    expected_move = S * sigma_sqrt_t  # 1-sigma move in dollars
    
    return {
        'p_up': p_up,
        'p_down': p_down,
        'd2': d2,
        'sigma_sqrt_t': sigma_sqrt_t,
        'delta_up': delta_up,
        'delta_down': delta_down,
        'expected_5min_move': expected_move,
    }


def taker_fee(price: float) -> float:
    """
    Dynamic taker fee per share.
    fee = 0.25 × (p × (1-p))²
    
    Peaks at p=0.50 (1.56%), vanishes at extremes.
    """
    if price <= 0 or price >= 1:
        return 0.0
    return 0.25 * (price * (1.0 - price)) ** 2


def min_edge_at_price(price: float) -> float:
    """
    Minimum edge needed for profitability after taker fee.
    
    Note: only 1× fee, not 2×. In binary options, you buy tokens and hold 
    to settlement — settlement pays $1.00 per winning share automatically.
    There is no sell-side taker fee at resolution.
    """
    return taker_fee(price)


# ============================================================
# Volatility conversion helpers
# ============================================================

SECONDS_PER_YEAR = 365.25 * 24 * 3600  # 31,557,600

def annualized_to_5min(sigma_annual: float) -> float:
    """Convert annualized vol to 5-minute vol."""
    T = 300 / SECONDS_PER_YEAR
    return sigma_annual * math.sqrt(T)

def vol_table(sigma_annual: float, btc_price: float = 97000) -> list[dict]:
    """Generate volatility conversion table."""
    intervals = [
        ("1 min", 60),
        ("5 min", 300),
        ("15 min", 900),
        ("1 hour", 3600),
        ("1 day", 86400),
    ]
    result = []
    for name, seconds in intervals:
        T = seconds / SECONDS_PER_YEAR
        period_vol = sigma_annual * math.sqrt(T)
        expected_move = btc_price * period_vol
        result.append({
            'interval': name,
            'period_vol': period_vol,
            'expected_move_pct': period_vol * 100,
            'expected_move_usd': expected_move,
        })
    return result
