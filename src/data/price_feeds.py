"""
BTC price feeds and volatility estimation.

Primary: Binance WebSocket (fastest, sub-10ms)
Fallback: Kraken REST API
Future: Polymarket RTDS WebSocket (Binance + Chainlink)
"""

import asyncio
import math
import time
import aiohttp
import json
from typing import Optional
from config.settings import (
    BINANCE_WS, BINANCE_REST, KRAKEN_REST,
    VOL_LOOKBACK, VOL_MIN, VOL_MAX, VOL_DEFAULT
)


SECONDS_PER_YEAR = 365.25 * 24 * 3600


class BTCPriceFeed:
    """
    BTC price tracker with rolling realized volatility.
    
    Supports multiple data sources:
    - Binance WebSocket (primary, if accessible)
    - Kraken REST (fallback)
    - Manual price injection (for backtesting)
    """
    
    def __init__(self):
        self.price: float = 0.0
        self.last_update: float = 0.0
        self.price_history: list[tuple[float, float]] = []  # (timestamp, price)
        self.vol_annualized: float = VOL_DEFAULT
        self._source: str = "none"
    
    @property
    def source(self) -> str:
        return self._source
    
    @property
    def age_seconds(self) -> float:
        """Seconds since last price update."""
        return time.time() - self.last_update if self.last_update > 0 else float('inf')
    
    def inject_price(self, price: float, timestamp: Optional[float] = None):
        """Manually inject a price (for backtesting)."""
        ts = timestamp or time.time()
        self.price = price
        self.last_update = ts
        self._source = "manual"
        self._record(ts, price)
    
    async def fetch_binance(self, session: aiohttp.ClientSession) -> Optional[float]:
        """Fetch from Binance REST API."""
        try:
            async with session.get(
                f"{BINANCE_REST}/ticker/price",
                params={"symbol": "BTCUSDT"},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = float(data.get('price', 0))
                    if price > 0:
                        self._update(price, "binance")
                        return price
        except Exception:
            pass
        return None
    
    async def fetch_kraken(self, session: aiohttp.ClientSession) -> Optional[float]:
        """Fetch from Kraken REST API."""
        try:
            async with session.get(
                f"{KRAKEN_REST}/Ticker",
                params={"pair": "XBTUSD"},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get('result', {})
                    if result:
                        ticker = list(result.values())[0]
                        price = float(ticker.get('c', [0])[0])
                        if price > 0:
                            self._update(price, "kraken")
                            return price
        except Exception:
            pass
        return None
    
    async def fetch_price(self, session: aiohttp.ClientSession) -> float:
        """Fetch price from best available source."""
        # Try Binance first (faster, more accurate)
        price = await self.fetch_binance(session)
        if price:
            return price
        
        # Fallback to Kraken
        price = await self.fetch_kraken(session)
        if price:
            return price
        
        return self.price
    
    def _update(self, price: float, source: str):
        """Update internal state with new price."""
        ts = time.time()
        self.price = price
        self.last_update = ts
        self._source = source
        self._record(ts, price)
    
    def _record(self, ts: float, price: float):
        """Record price and recompute volatility."""
        self.price_history.append((ts, price))
        
        # Trim history
        if len(self.price_history) > VOL_LOOKBACK * 2:
            self.price_history = self.price_history[-VOL_LOOKBACK * 2:]
        
        self._compute_volatility()
    
    def _compute_volatility(self):
        """
        Compute rolling realized volatility from price history.
        
        Uses log returns, annualized by time-weighting.
        This is more accurate than Deribit implied vol for 5-min pricing.
        """
        if len(self.price_history) < 10:
            return
        
        # Compute log returns
        returns = []
        for i in range(1, len(self.price_history)):
            t0, p0 = self.price_history[i - 1]
            t1, p1 = self.price_history[i]
            dt = t1 - t0
            if dt > 0 and p0 > 0 and p1 > 0:
                log_return = math.log(p1 / p0)
                returns.append((dt, log_return))
        
        if len(returns) < 5:
            return
        
        # Annualize: variance per year = variance per period × periods per year
        avg_dt = sum(dt for dt, _ in returns) / len(returns)
        if avg_dt <= 0:
            return
        
        # Variance with Bessel's correction (subtract mean, divide by N-1)
        # Avoids overestimating vol during trending periods
        log_returns = [r for _, r in returns]
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        periods_per_year = SECONDS_PER_YEAR / avg_dt
        annual_variance = variance * periods_per_year
        self.vol_annualized = math.sqrt(max(annual_variance, 0))
        
        # Clamp to reasonable range
        self.vol_annualized = max(VOL_MIN, min(VOL_MAX, self.vol_annualized))
    
    def get_5min_stats(self) -> dict:
        """Get volatility stats for 5-minute window."""
        T_5min = 300 / SECONDS_PER_YEAR
        sigma_5min = self.vol_annualized * math.sqrt(T_5min)
        expected_move = self.price * sigma_5min if self.price > 0 else 0
        
        return {
            'vol_annualized': self.vol_annualized,
            'vol_5min': sigma_5min,
            'expected_move_pct': sigma_5min * 100,
            'expected_move_usd': expected_move,
            'price': self.price,
            'source': self._source,
            'age_seconds': self.age_seconds,
            'data_points': len(self.price_history),
        }
