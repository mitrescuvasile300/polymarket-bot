"""
Polymarket API client for market discovery and order book data.

Endpoints:
- Gamma API: Market discovery by slug pattern
- CLOB API: Order book, prices, spreads
- CLOB WebSocket: Real-time order book updates
"""

import asyncio
import json
import time
import aiohttp
from typing import Optional
from config.settings import GAMMA_API, CLOB_API, WINDOW_SECONDS


class PolymarketClient:
    """
    Polymarket read-only client for scanning markets.
    No authentication needed for read operations.
    """
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self._owns_session = session is None
    
    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session
    
    async def close(self):
        if self._owns_session and self._session:
            await self._session.close()
    
    # ============================================================
    # Market Discovery
    # ============================================================
    
    @staticmethod
    def get_window_timestamp(ts: Optional[float] = None) -> int:
        """Get the 5-min window start timestamp (aligned to 300s boundary)."""
        t = int(ts or time.time())
        return (t // WINDOW_SECONDS) * WINDOW_SECONDS
    
    @staticmethod
    def get_slug(window_ts: int) -> str:
        """Generate the predictable market slug."""
        return f"btc-updown-5m-{window_ts}"
    
    async def discover_market(self, window_ts: int) -> Optional[dict]:
        """
        Look up a BTC 5-min market by window timestamp.
        
        Returns:
            Market dict with slug, title, token_ids, outcomes, window times
            or None if not found.
        """
        slug = self.get_slug(window_ts)
        try:
            async with self.session.get(
                f"{GAMMA_API}/events",
                params={"slug": slug},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return None
                
                events = await resp.json()
                if not events:
                    return None
                
                event = events[0] if isinstance(events, list) else events
                markets = event.get('markets', [])
                if not markets:
                    return None
                
                m = markets[0]
                token_ids = json.loads(m['clobTokenIds']) if isinstance(m.get('clobTokenIds'), str) else m.get('clobTokenIds', [])
                outcomes = json.loads(m['outcomes']) if isinstance(m.get('outcomes'), str) else m.get('outcomes', [])
                
                return {
                    'slug': slug,
                    'title': event.get('title', ''),
                    'condition_id': m.get('conditionId', ''),
                    'window_start': window_ts,
                    'window_end': window_ts + WINDOW_SECONDS,
                    'token_ids': token_ids,  # [Up_token, Down_token]
                    'outcomes': outcomes,     # ['Up', 'Down']
                }
        except Exception as e:
            return None
    
    async def discover_current_markets(self) -> list[dict]:
        """Discover current and next BTC 5-min markets."""
        now = time.time()
        current_window = self.get_window_timestamp(now)
        
        tasks = [
            self.discover_market(current_window),
            self.discover_market(current_window + WINDOW_SECONDS),
        ]
        results = await asyncio.gather(*tasks)
        return [m for m in results if m is not None]
    
    # ============================================================
    # Order Book Data
    # ============================================================
    
    async def get_order_book(self, token_id: str) -> dict:
        """
        Get full order book for a token.
        
        Returns:
            {bids: [{price, size}], asks: [{price, size}]}
        """
        try:
            async with self.session.get(
                f"{CLOB_API}/book",
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    book = await resp.json()
                    return {
                        'bids': book.get('bids', []),
                        'asks': book.get('asks', []),
                    }
        except Exception:
            pass
        return {'bids': [], 'asks': []}
    
    async def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get best price for a token (BUY or SELL side)."""
        try:
            async with self.session.get(
                f"{CLOB_API}/price",
                params={"token_id": token_id, "side": side},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = float(data.get('price', 0))
                    return price if price > 0 else None
        except Exception:
            pass
        return None
    
    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token."""
        try:
            async with self.session.get(
                f"{CLOB_API}/midpoint",
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    mid = float(data.get('mid', 0))
                    return mid if mid > 0 else None
        except Exception:
            pass
        return None
    
    async def get_spread(self, token_id: str) -> Optional[dict]:
        """Get spread for a token."""
        try:
            async with self.session.get(
                f"{CLOB_API}/spread",
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        return None
    
    async def get_market_summary(self, market: dict) -> dict:
        """
        Get complete price summary for a market (both Up and Down).
        
        Returns dict with:
            Up: {bid, ask, mid, spread, bid_size, ask_size}
            Down: {bid, ask, mid, spread, bid_size, ask_size}
        """
        result = {}
        for i, tid in enumerate(market.get('token_ids', [])):
            outcome = market['outcomes'][i] if i < len(market.get('outcomes', [])) else f'token_{i}'
            book = await self.get_order_book(tid)
            
            bids = book['bids']
            asks = book['asks']
            
            best_bid = float(bids[0]['price']) if bids else 0.0
            best_ask = float(asks[0]['price']) if asks else 1.0
            bid_size = float(bids[0].get('size', 0)) if bids else 0.0
            ask_size = float(asks[0].get('size', 0)) if asks else 0.0
            mid = (best_bid + best_ask) / 2 if bids and asks else None
            spread = best_ask - best_bid
            
            # Depth: total size at top 3 levels
            bid_depth = sum(float(b.get('size', 0)) for b in bids[:3])
            ask_depth = sum(float(a.get('size', 0)) for a in asks[:3])
            
            result[outcome] = {
                'bid': best_bid,
                'ask': best_ask,
                'mid': mid,
                'spread': spread,
                'bid_size': bid_size,
                'ask_size': ask_size,
                'bid_depth_3': bid_depth,
                'ask_depth_3': ask_depth,
                'book_levels': len(bids) + len(asks),
            }
        
        return result
