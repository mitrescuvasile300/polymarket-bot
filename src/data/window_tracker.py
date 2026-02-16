"""
BTC 5-min window strike price tracker.

The critical issue: to calculate BS fair value, we need K (the BTC price 
at window open). The Polymarket CLOB doesn't expose K directly - it's 
set by Chainlink at each window open.

Approaches to determine K:
1. Cache BTC price at each 300-second boundary (best for live trading)
2. Invert BS from market mid price + current S (works when market is liquid)
3. Use Polymarket RTDS to get Chainlink price at window open

For now, we use approach 1 (price caching) + approach 2 (BS inversion) as validation.
"""

import math
import time
from typing import Optional
from scipy.stats import norm


class WindowTracker:
    """
    Tracks BTC opening prices for each 5-minute window.
    
    The most critical component for accurate BS pricing.
    Without the correct K, our fair value estimate is garbage.
    """
    
    def __init__(self):
        # window_ts → opening BTC price
        self._strikes: dict[int, float] = {}
        self._last_price: float = 0.0
        self._last_price_ts: float = 0.0
    
    def update_price(self, price: float, timestamp: Optional[float] = None):
        """
        Update the current BTC price and cache strike for any new windows.
        Call this frequently (every few seconds) to catch window opens.
        """
        ts = timestamp or time.time()
        self._last_price = price
        self._last_price_ts = ts
        
        # Check if we're at a window boundary
        window_ts = (int(ts) // 300) * 300
        if window_ts not in self._strikes:
            # We're in a new window - cache the opening price
            self._strikes[window_ts] = price
        
        # Cleanup old windows (keep last 20)
        if len(self._strikes) > 20:
            sorted_keys = sorted(self._strikes.keys())
            for k in sorted_keys[:-20]:
                del self._strikes[k]
    
    def get_strike(self, window_ts: int) -> Optional[float]:
        """Get the cached strike price for a window."""
        return self._strikes.get(window_ts)
    
    def estimate_strike_from_market(
        self,
        market_up_mid: float,
        current_btc: float,
        sigma: float,
        time_remaining_sec: float,
    ) -> Optional[float]:
        """
        Estimate K by inverting Black-Scholes from the market mid price.
        
        If market_up_mid = N(d2) and d2 = ln(S/K) / (σ√T), then:
        K = S × exp(-d2 × σ√T)  where  d2 = N⁻¹(market_up_mid)
        """
        if market_up_mid <= 0.01 or market_up_mid >= 0.99:
            return None  # Can't reliably invert at extremes
        if time_remaining_sec <= 0 or sigma <= 0 or current_btc <= 0:
            return None
        
        T_years = time_remaining_sec / (365.25 * 24 * 3600)
        sigma_sqrt_t = sigma * math.sqrt(T_years)
        
        if sigma_sqrt_t < 1e-10:
            return None
        
        # Invert: d2 = N⁻¹(market_mid)
        try:
            d2 = norm.ppf(market_up_mid)
            K = current_btc * math.exp(-d2 * sigma_sqrt_t)
            return K
        except Exception:
            return None
    
    def get_best_strike(
        self,
        window_ts: int,
        current_btc: float,
        market_up_mid: Optional[float] = None,
        sigma: float = 0.45,
        time_remaining_sec: float = 150,
    ) -> tuple[float, str]:
        """
        Get best estimate of strike price K.
        
        Returns (K, source) where source is 'cached', 'inverted', or 'approximated'.
        """
        # Priority 1: Cached from window open
        cached = self.get_strike(window_ts)
        if cached:
            return (cached, 'cached')
        
        # Priority 2: Invert from market mid price
        if market_up_mid and 0.05 < market_up_mid < 0.95:
            inverted = self.estimate_strike_from_market(
                market_up_mid, current_btc, sigma, time_remaining_sec
            )
            if inverted and abs(inverted - current_btc) / current_btc < 0.01:
                return (inverted, 'inverted')
        
        # Priority 3: Use current price (worst case)
        return (current_btc, 'approximated')
    
    @property
    def tracked_windows(self) -> dict[int, float]:
        """Get all tracked window strikes."""
        return dict(self._strikes)
