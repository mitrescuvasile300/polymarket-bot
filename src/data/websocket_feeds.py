"""
Real-time WebSocket feeds for price data and order book updates.

Three feeds:
1. Binance trade stream — Real-time BTC trades (sub-10ms)
2. Polymarket RTDS — Binance + Chainlink prices (settlement source!)
3. Polymarket CLOB WS — Real-time order book updates

These replace REST polling (5s delay → sub-10ms).
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import aiohttp

from config.settings import (
    BINANCE_WS,
    RTDS_WS,
    CLOB_WS,
    VOL_LOOKBACK,
    VOL_MIN,
    VOL_MAX,
    VOL_DEFAULT,
)

SECONDS_PER_YEAR = 365.25 * 24 * 3600

logger = logging.getLogger(__name__)


# ================================================================
# Binance Trade WebSocket
# ================================================================

@dataclass
class BinanceTrade:
    """A single Binance trade."""
    price: float
    quantity: float
    timestamp: int  # milliseconds
    is_buyer_maker: bool


class BinanceTradeStream:
    """
    Real-time BTC trade stream from Binance.
    
    Provides sub-10ms price updates. This is the PRIMARY price discovery
    source — Chainlink oracle lags Binance by 1-3 seconds.
    
    Usage:
        stream = BinanceTradeStream(on_trade=my_callback)
        await stream.connect()
        # ... stream runs until stop
        await stream.disconnect()
    """
    
    def __init__(
        self,
        on_trade: Optional[Callable[[BinanceTrade], None]] = None,
        on_price: Optional[Callable[[float, int], None]] = None,
        url: str = BINANCE_WS,
    ):
        self.on_trade = on_trade
        self.on_price = on_price  # Simpler callback: (price, timestamp_ms)
        self.url = url
        
        self.last_price: float = 0.0
        self.last_timestamp: int = 0
        self.trade_count: int = 0
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def connect(self):
        """Connect and start receiving trades."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Binance WS connecting to {self.url}")
    
    async def disconnect(self):
        """Disconnect and clean up."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Binance WS disconnected")
    
    async def _run_loop(self):
        """Main connection loop with auto-reconnect."""
        while self._running:
            try:
                self._session = aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(
                    self.url,
                    heartbeat=20,
                    timeout=aiohttp.ClientTimeout(total=30),
                )
                logger.info("Binance WS connected")
                
                async for msg in self._ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._handle_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"Binance WS error: {self._ws.exception()}")
                        break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Binance WS connection error: {e}")
            finally:
                if self._session and not self._session.closed:
                    await self._session.close()
            
            if self._running:
                logger.info("Binance WS reconnecting in 2s...")
                await asyncio.sleep(2)
    
    def _handle_message(self, raw: str):
        """Parse and dispatch a trade message."""
        try:
            data = json.loads(raw)
            trade = BinanceTrade(
                price=float(data['p']),
                quantity=float(data['q']),
                timestamp=int(data['T']),
                is_buyer_maker=data.get('m', False),
            )
            
            self.last_price = trade.price
            self.last_timestamp = trade.timestamp
            self.trade_count += 1
            
            if self.on_trade:
                self.on_trade(trade)
            if self.on_price:
                self.on_price(trade.price, trade.timestamp)
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Binance parse error: {e}")


# ================================================================
# Polymarket RTDS WebSocket
# ================================================================

@dataclass
class RTDSPrice:
    """Price update from RTDS (Binance or Chainlink)."""
    source: str         # 'binance' or 'chainlink'
    symbol: str         # 'btcusdt' or 'btc/usd'
    price: float
    timestamp: float    # Unix seconds


class RTDSStream:
    """
    Polymarket Real-Time Data Stream.
    
    Streams BOTH Binance and Chainlink prices simultaneously.
    Chainlink is the SETTLEMENT source — critical for accurate K (strike) pricing.
    
    Subscribe to:
    - crypto_prices (Binance feed through Polymarket)
    - crypto_prices_chainlink (Oracle feed — THIS is what determines settlement)
    
    Usage:
        rtds = RTDSStream(on_binance=fn1, on_chainlink=fn2)
        await rtds.connect()
    """
    
    def __init__(
        self,
        on_binance: Optional[Callable[[float, float], None]] = None,
        on_chainlink: Optional[Callable[[float, float], None]] = None,
        on_price: Optional[Callable[[RTDSPrice], None]] = None,
        url: str = RTDS_WS,
    ):
        self.on_binance = on_binance      # (price, timestamp)
        self.on_chainlink = on_chainlink  # (price, timestamp) — settlement source!
        self.on_price = on_price          # Generic callback
        self.url = url
        
        self.binance_price: float = 0.0
        self.chainlink_price: float = 0.0
        self.binance_ts: float = 0.0
        self.chainlink_ts: float = 0.0
        self.update_count: int = 0
        
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    @property
    def price_divergence(self) -> float:
        """Current Binance-Chainlink price difference."""
        if self.binance_price > 0 and self.chainlink_price > 0:
            return abs(self.binance_price - self.chainlink_price)
        return 0.0
    
    @property
    def chainlink_lag_ms(self) -> float:
        """Estimated Chainlink lag behind Binance in ms."""
        if self.binance_ts > 0 and self.chainlink_ts > 0:
            return (self.binance_ts - self.chainlink_ts) * 1000
        return 0.0
    
    async def connect(self):
        """Connect and subscribe to BTC price feeds."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("RTDS connecting...")
    
    async def disconnect(self):
        """Disconnect cleanly."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("RTDS disconnected")
    
    async def _run_loop(self):
        """Connection loop with auto-reconnect."""
        while self._running:
            try:
                self._session = aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(
                    self.url,
                    heartbeat=30,
                    timeout=aiohttp.ClientTimeout(total=30),
                )
                logger.info("RTDS connected")
                
                # Subscribe to Binance + Chainlink BTC feeds
                await self._subscribe()
                
                async for msg in self._ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._handle_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"RTDS error: {self._ws.exception()}")
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"RTDS connection error: {e}")
            finally:
                if self._session and not self._session.closed:
                    await self._session.close()
            
            if self._running:
                logger.info("RTDS reconnecting in 2s...")
                await asyncio.sleep(2)
    
    async def _subscribe(self):
        """Send subscription messages for BTC price feeds."""
        # Binance prices through Polymarket
        binance_sub = {
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices",
                "type": "update",
                "filters": "btcusdt",
            }]
        }
        
        # Chainlink oracle prices (SETTLEMENT SOURCE)
        chainlink_sub = {
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": json.dumps({"symbol": "btc/usd"}),
            }]
        }
        
        await self._ws.send_str(json.dumps(binance_sub))
        await self._ws.send_str(json.dumps(chainlink_sub))
        logger.info("RTDS subscribed to BTC Binance + Chainlink feeds")
    
    def _handle_message(self, raw: str):
        """Parse and dispatch RTDS price update."""
        try:
            data = json.loads(raw)
            topic = data.get('topic', '')
            
            if topic == 'crypto_prices':
                # Binance price
                payload = data.get('data', data)
                price = float(payload.get('price', payload.get('p', 0)))
                ts = float(payload.get('timestamp', time.time()))
                
                if price > 0:
                    self.binance_price = price
                    self.binance_ts = ts
                    self.update_count += 1
                    
                    update = RTDSPrice('binance', 'btcusdt', price, ts)
                    if self.on_binance:
                        self.on_binance(price, ts)
                    if self.on_price:
                        self.on_price(update)
            
            elif topic == 'crypto_prices_chainlink':
                # Chainlink oracle price
                payload = data.get('data', data)
                price = float(payload.get('price', payload.get('p', 0)))
                ts = float(payload.get('timestamp', time.time()))
                
                if price > 0:
                    self.chainlink_price = price
                    self.chainlink_ts = ts
                    self.update_count += 1
                    
                    update = RTDSPrice('chainlink', 'btc/usd', price, ts)
                    if self.on_chainlink:
                        self.on_chainlink(price, ts)
                    if self.on_price:
                        self.on_price(update)
                        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"RTDS parse error: {e}")


# ================================================================
# Polymarket CLOB Order Book WebSocket
# ================================================================

@dataclass
class BookUpdate:
    """Order book update for a token."""
    token_id: str
    bids: list  # [{price, size}]
    asks: list  # [{price, size}]
    timestamp: float
    best_bid: float = 0.0
    best_ask: float = 1.0
    spread: float = 1.0


class CLOBBookStream:
    """
    Real-time order book updates from Polymarket CLOB.
    
    Subscribe to specific token IDs to see bids/asks change instantly.
    Much faster than REST polling for detecting filled orders and spread changes.
    
    Usage:
        book = CLOBBookStream(
            token_ids=["0xabc...", "0xdef..."],
            on_update=my_callback,
        )
        await book.connect()
    """
    
    def __init__(
        self,
        token_ids: Optional[list[str]] = None,
        on_update: Optional[Callable[[BookUpdate], None]] = None,
        url: str = CLOB_WS,
    ):
        self.token_ids = token_ids or []
        self.on_update = on_update
        self.url = url
        
        self.books: dict[str, BookUpdate] = {}  # token_id -> latest
        self.update_count: int = 0
        
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def add_tokens(self, token_ids: list[str]):
        """Add token IDs to subscribe to (call before or after connect)."""
        new_ids = [t for t in token_ids if t not in self.token_ids]
        self.token_ids.extend(new_ids)
        # If already connected, subscribe to new tokens
        if self._ws and not self._ws.closed and new_ids:
            asyncio.create_task(self._subscribe_tokens(new_ids))
    
    def remove_tokens(self, token_ids: list[str]):
        """Unsubscribe from token IDs."""
        for tid in token_ids:
            if tid in self.token_ids:
                self.token_ids.remove(tid)
            self.books.pop(tid, None)
    
    async def connect(self):
        """Connect and start receiving book updates."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"CLOB WS connecting ({len(self.token_ids)} tokens)...")
    
    async def disconnect(self):
        """Disconnect cleanly."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("CLOB WS disconnected")
    
    async def _run_loop(self):
        """Connection loop with auto-reconnect."""
        while self._running:
            try:
                self._session = aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(
                    self.url,
                    heartbeat=30,
                    timeout=aiohttp.ClientTimeout(total=30),
                )
                logger.info("CLOB WS connected")
                
                if self.token_ids:
                    await self._subscribe_tokens(self.token_ids)
                
                async for msg in self._ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._handle_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"CLOB WS error: {self._ws.exception()}")
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"CLOB WS connection error: {e}")
            finally:
                if self._session and not self._session.closed:
                    await self._session.close()
                # Clear stale book data on disconnect to prevent trading on
                # outdated orderbook state during reconnection gap
                self.books.clear()
            
            if self._running:
                logger.info("CLOB WS reconnecting in 2s (books cleared)...")
                await asyncio.sleep(2)
    
    async def _subscribe_tokens(self, token_ids: list[str]):
        """Subscribe to book updates for specific tokens."""
        for tid in token_ids:
            sub = {
                "auth": {},
                "markets": [tid],
                "assets_ids": [tid],
                "type": "market",
            }
            await self._ws.send_str(json.dumps(sub))
        logger.info(f"CLOB WS subscribed to {len(token_ids)} tokens")
    
    def _handle_message(self, raw: str):
        """Parse book update."""
        try:
            data = json.loads(raw)
            
            # Handle different message formats
            if isinstance(data, list):
                for item in data:
                    self._process_book_event(item)
            elif isinstance(data, dict):
                self._process_book_event(data)
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"CLOB WS parse error: {e}")
    
    def _process_book_event(self, event: dict):
        """Process a single book event."""
        asset_id = event.get('asset_id', event.get('market', ''))
        if not asset_id:
            return
        
        bids = event.get('bids', [])
        asks = event.get('asks', [])
        
        # Parse price levels
        bid_levels = []
        ask_levels = []
        for b in bids:
            if isinstance(b, dict):
                bid_levels.append({'price': float(b.get('price', 0)), 'size': float(b.get('size', 0))})
        for a in asks:
            if isinstance(a, dict):
                ask_levels.append({'price': float(a.get('price', 0)), 'size': float(a.get('size', 0))})
        
        best_bid = bid_levels[0]['price'] if bid_levels else 0.0
        best_ask = ask_levels[0]['price'] if ask_levels else 1.0
        
        update = BookUpdate(
            token_id=asset_id,
            bids=bid_levels,
            asks=ask_levels,
            timestamp=time.time(),
            best_bid=best_bid,
            best_ask=best_ask,
            spread=best_ask - best_bid,
        )
        
        self.books[asset_id] = update
        self.update_count += 1
        
        if self.on_update:
            self.on_update(update)


# ================================================================
# Combined Feed Manager
# ================================================================

class PriceFeedManager:
    """
    Manages all WebSocket feeds in one place.
    
    Provides unified access to:
    - BTC price (Binance direct, RTDS Binance, RTDS Chainlink)
    - Order book state for active tokens
    - Price divergence and lag metrics
    
    Usage:
        feeds = PriceFeedManager()
        await feeds.start()
        
        btc_price = feeds.btc_price
        chainlink_price = feeds.chainlink_price
        book = feeds.get_book(token_id)
        
        await feeds.stop()
    """
    
    def __init__(self):
        self.binance = BinanceTradeStream(
            on_price=self._on_binance_price,
        )
        self.rtds = RTDSStream(
            on_binance=self._on_rtds_binance,
            on_chainlink=self._on_rtds_chainlink,
        )
        self.clob = CLOBBookStream(
            on_update=self._on_book_update,
        )
        
        # Unified state
        self.btc_price: float = 0.0
        self.chainlink_price: float = 0.0
        self.btc_timestamp: float = 0.0
        self.chainlink_timestamp: float = 0.0
        
        # Rolling price history for realized vol calculation
        self._price_history: list[tuple[float, float]] = []  # (timestamp, price)
        self.realized_vol: float = VOL_DEFAULT  # Annualized realized volatility
        
        # Callbacks for the scanner/trader
        self._price_callbacks: list[Callable] = []
        self._book_callbacks: list[Callable] = []
    
    def on_price_change(self, callback: Callable[[float, str], None]):
        """Register callback for BTC price changes. Args: (price, source)."""
        self._price_callbacks.append(callback)
    
    def on_book_change(self, callback: Callable[[BookUpdate], None]):
        """Register callback for order book changes."""
        self._book_callbacks.append(callback)
    
    async def start(self, use_binance_direct: bool = True, use_rtds: bool = True):
        """Start all feeds."""
        tasks = []
        if use_binance_direct:
            tasks.append(self.binance.connect())
        if use_rtds:
            tasks.append(self.rtds.connect())
        # CLOB WS started separately when we know token IDs
        await asyncio.gather(*tasks)
        logger.info("All price feeds started")
    
    async def start_book_feed(self, token_ids: list[str]):
        """Start CLOB book feed for specific tokens."""
        self.clob.add_tokens(token_ids)
        if not self.clob._running:
            await self.clob.connect()
    
    async def stop(self):
        """Stop all feeds."""
        await asyncio.gather(
            self.binance.disconnect(),
            self.rtds.disconnect(),
            self.clob.disconnect(),
        )
        logger.info("All feeds stopped")
    
    def get_book(self, token_id: str, max_age_seconds: float = 5.0) -> Optional[BookUpdate]:
        """
        Get latest order book for a token, with staleness check.
        
        Returns None if book data is older than max_age_seconds to prevent
        trading on stale orderbook during WebSocket reconnection gaps.
        """
        book = self.clob.books.get(token_id)
        if book and (time.time() - book.timestamp) > max_age_seconds:
            logger.debug(f"Book for {token_id[:16]}... is stale ({time.time() - book.timestamp:.1f}s old)")
            return None
        return book
    
    @property
    def divergence(self) -> float:
        """Current Binance-Chainlink price divergence."""
        if self.btc_price > 0 and self.chainlink_price > 0:
            return abs(self.btc_price - self.chainlink_price)
        return 0.0
    
    @property
    def divergence_pct(self) -> float:
        """Price divergence as percentage."""
        if self.btc_price > 0 and self.chainlink_price > 0:
            return abs(self.btc_price - self.chainlink_price) / self.btc_price * 100
        return 0.0
    
    def status(self) -> dict:
        """Complete feed status."""
        return {
            'btc_price': self.btc_price,
            'chainlink_price': self.chainlink_price,
            'divergence_usd': self.divergence,
            'divergence_pct': f"{self.divergence_pct:.4f}%",
            'realized_vol': self.realized_vol,
            'vol_data_points': len(self._price_history),
            'binance_trades': self.binance.trade_count,
            'rtds_updates': self.rtds.update_count,
            'clob_updates': self.clob.update_count,
            'active_books': len(self.clob.books),
        }
    
    def _compute_realized_vol(self):
        """
        Compute rolling realized volatility from WebSocket price history.
        
        Uses Bessel-corrected log returns, annualized by time-weighting.
        Same methodology as BTCPriceFeed but operates on WS data.
        """
        if len(self._price_history) < 10:
            return
        
        # Compute log returns
        returns = []
        for i in range(1, len(self._price_history)):
            t0, p0 = self._price_history[i - 1]
            t1, p1 = self._price_history[i]
            dt = t1 - t0
            if dt > 0 and p0 > 0 and p1 > 0:
                log_return = math.log(p1 / p0)
                returns.append((dt, log_return))
        
        if len(returns) < 5:
            return
        
        avg_dt = sum(dt for dt, _ in returns) / len(returns)
        if avg_dt <= 0:
            return
        
        # Bessel-corrected variance
        log_returns = [r for _, r in returns]
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        periods_per_year = SECONDS_PER_YEAR / avg_dt
        annual_variance = variance * periods_per_year
        vol = math.sqrt(max(annual_variance, 0))
        
        # Clamp to reasonable range
        self.realized_vol = max(VOL_MIN, min(VOL_MAX, vol))
    
    # Internal callbacks
    def _on_binance_price(self, price: float, ts_ms: int):
        self.btc_price = price
        self.btc_timestamp = ts_ms / 1000.0
        
        # Record for rolling vol calculation
        ts = ts_ms / 1000.0
        self._price_history.append((ts, price))
        if len(self._price_history) > VOL_LOOKBACK * 2:
            self._price_history = self._price_history[-VOL_LOOKBACK * 2:]
        self._compute_realized_vol()
        
        for cb in self._price_callbacks:
            try:
                cb(price, 'binance')
            except Exception:
                pass
    
    def _on_rtds_binance(self, price: float, ts: float):
        # Only update if we don't have direct Binance
        if self.binance.trade_count == 0:
            self.btc_price = price
            self.btc_timestamp = ts
    
    def _on_rtds_chainlink(self, price: float, ts: float):
        self.chainlink_price = price
        self.chainlink_timestamp = ts
        for cb in self._price_callbacks:
            try:
                cb(price, 'chainlink')
            except Exception:
                pass
    
    def _on_book_update(self, update: BookUpdate):
        for cb in self._book_callbacks:
            try:
                cb(update)
            except Exception:
                pass
