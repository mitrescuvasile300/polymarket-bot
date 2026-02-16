"""
Window Lifecycle Manager for BTC 5-minute binary options.

Automates the full trading loop:
1. Discover upcoming window 10-20s before it opens
2. Subscribe to its orderbook via CLOB WebSocket
3. Capture Chainlink strike price at window open
4. Monitor for BS mispricing opportunities during the 5 minutes
5. Execute trades when edge exceeds threshold
6. Cancel all orders 30s before close
7. Track settlement at expiry
8. Move to next window

This is the "glue" between scanner, execution, and data feeds.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable

import aiohttp

from config.settings import (
    WINDOW_SECONDS,
    STOP_BEFORE_EXPIRY,
    MIN_EDGE,
    BANKROLL,
)
from src.pricing.black_scholes import bs_binary_price, taker_fee, SECONDS_PER_YEAR
from src.pricing.kelly import kelly_bet_size, edge_summary
from src.data.polymarket_api import PolymarketClient
from src.data.websocket_feeds import PriceFeedManager
from src.execution.trader import Trader, OrderSide, OrderResult
from src.execution.positions import PositionTracker
from src.risk.manager import RiskManager, TradeRecord

logger = logging.getLogger(__name__)


class WindowState(str, Enum):
    """States in the window lifecycle."""
    WAITING = "waiting"        # Before window opens
    DISCOVERY = "discovery"    # Discovering market on Gamma API
    STRIKE_CAPTURE = "strike"  # Capturing K at window open
    MONITORING = "monitoring"  # Scanning for opportunities
    TRADING = "trading"        # Active trade(s) placed
    CLOSING = "closing"        # Cancelling orders before expiry
    SETTLED = "settled"        # Window expired, position settled
    SKIPPED = "skipped"        # No opportunity found


@dataclass
class WindowSession:
    """Tracks one 5-minute window's lifecycle."""
    window_ts: int
    slug: str = ""
    state: WindowState = WindowState.WAITING
    
    # Market data
    market: dict = field(default_factory=dict)
    token_ids: list = field(default_factory=list)
    
    # Pricing
    strike_binance: float = 0.0     # BTC at window open (Binance)
    strike_chainlink: float = 0.0   # BTC at window open (Chainlink) — settlement reference
    strike_used: float = 0.0        # Actual K used for pricing
    strike_source: str = ""
    sigma: float = 0.0
    
    # Trades
    trades: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    
    # Timing
    discovered_at: float = 0.0
    opened_at: float = 0.0
    first_trade_at: float = 0.0
    closed_at: float = 0.0
    
    # Results
    pnl: float = 0.0
    
    @property
    def window_end(self) -> int:
        return self.window_ts + WINDOW_SECONDS
    
    @property
    def time_remaining(self) -> float:
        return max(0, self.window_end - time.time())
    
    @property
    def is_expired(self) -> bool:
        return time.time() >= self.window_end
    
    @property
    def should_stop_trading(self) -> bool:
        return self.time_remaining < STOP_BEFORE_EXPIRY


class WindowLifecycleManager:
    """
    Runs the continuous window trading loop.
    
    Each 5-minute cycle:
    1. Pre-open (T-20s to T-0s): Discover market, subscribe to book
    2. Open (T+0s): Capture Chainlink K via RTDS
    3. Active (T+0s to T+270s): Monitor BS mispricing, trade if +EV
    4. Close (T+270s to T+300s): Cancel orders, wait for settlement
    5. Settle: Record P&L, move to next window
    
    Args:
        trader: Trader instance (handles order execution)
        feeds: PriceFeedManager (WebSocket price feeds)
        min_edge: Minimum edge to trigger a trade
        bankroll: Starting bankroll
    """
    
    def __init__(
        self,
        trader: Trader,
        feeds: PriceFeedManager,
        min_edge: float = MIN_EDGE,
        bankroll: float = BANKROLL,
        on_signal: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
    ):
        self.trader = trader
        self.feeds = feeds
        self.min_edge = min_edge
        self.bankroll = bankroll
        self.on_signal = on_signal    # Callback when signal detected
        self.on_trade = on_trade      # Callback when trade executed
        
        # Session tracking
        self.current_session: Optional[WindowSession] = None
        self.session_history: list[WindowSession] = []
        self._running = False
        
        # Stats
        self.windows_scanned: int = 0
        self.windows_traded: int = 0
        self.total_pnl: float = 0.0
    
    async def run(
        self,
        duration_minutes: int = 60,
        scan_interval: float = 1.0,
    ):
        """
        Run the window lifecycle loop.
        
        Args:
            duration_minutes: How long to run (0 = indefinitely)
            scan_interval: Seconds between price checks within a window
        """
        self._running = True
        end_time = time.time() + duration_minutes * 60 if duration_minutes > 0 else float('inf')
        
        logger.info(
            f"🚀 Window Manager started | "
            f"Bankroll: ${self.bankroll:.2f} | Min edge: {self.min_edge:.1%} | "
            f"Mode: {'DRY RUN' if self.trader.dry_run else 'LIVE'}"
        )
        
        async with aiohttp.ClientSession() as session:
            poly_client = PolymarketClient(session)
            
            while self._running and time.time() < end_time:
                try:
                    await self._process_window_cycle(poly_client, scan_interval)
                except Exception as e:
                    logger.error(f"Window cycle error: {e}", exc_info=True)
                    await asyncio.sleep(5)
        
        self._print_summary()
        logger.info("Window Manager stopped")
    
    def stop(self):
        """Stop the manager gracefully."""
        self._running = False
    
    async def _process_window_cycle(
        self,
        poly_client: PolymarketClient,
        scan_interval: float,
    ):
        """Process one complete 5-minute window cycle."""
        now = time.time()
        current_window_ts = (int(now) // WINDOW_SECONDS) * WINDOW_SECONDS
        next_window_ts = current_window_ts + WINDOW_SECONDS
        
        # Determine which window to work on
        time_in_current = now - current_window_ts
        time_until_next = next_window_ts - now
        
        if time_in_current < WINDOW_SECONDS - STOP_BEFORE_EXPIRY:
            # Current window still tradeable
            target_ts = current_window_ts
        else:
            # Current window closing, prepare for next
            target_ts = next_window_ts
            # Wait until next window opens
            logger.info(f"Waiting {time_until_next:.0f}s for next window ({target_ts})...")
            await asyncio.sleep(min(time_until_next, 30))
            if not self._running:
                return
        
        # Create window session
        session = WindowSession(window_ts=target_ts)
        self.current_session = session
        self.windows_scanned += 1
        
        # Phase 1: DISCOVERY
        session.state = WindowState.DISCOVERY
        market = await poly_client.discover_market(target_ts)
        
        if not market:
            session.state = WindowState.SKIPPED
            logger.debug(f"No market found for window {target_ts}")
            await asyncio.sleep(5)
            return
        
        session.market = market
        session.slug = market['slug']
        session.token_ids = market.get('token_ids', [])
        session.discovered_at = time.time()
        
        # Subscribe to order book WebSocket for these tokens
        if session.token_ids:
            await self.feeds.start_book_feed(session.token_ids)
        
        logger.info(f"📊 Window {session.slug} | tokens: {len(session.token_ids)}")
        
        # Phase 2: STRIKE CAPTURE
        # Wait until window starts (if pre-open)
        if time.time() < target_ts:
            wait_time = target_ts - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        session.state = WindowState.STRIKE_CAPTURE
        session.opened_at = time.time()
        
        # Capture K from best available source
        if self.feeds.chainlink_price > 0:
            session.strike_chainlink = self.feeds.chainlink_price
            session.strike_used = self.feeds.chainlink_price
            session.strike_source = "chainlink_rtds"
        elif self.feeds.btc_price > 0:
            session.strike_binance = self.feeds.btc_price
            session.strike_used = self.feeds.btc_price
            session.strike_source = "binance_ws"
        else:
            # Fallback: REST
            async with aiohttp.ClientSession() as s:
                from src.data.price_feeds import BTCPriceFeed
                feed = BTCPriceFeed()
                price = await feed.fetch_price(s)
                if price > 0:
                    session.strike_used = price
                    session.strike_source = "rest_fallback"
        
        if session.strike_used <= 0:
            session.state = WindowState.SKIPPED
            logger.warning(f"No BTC price for strike capture — skipping window")
            return
        
        logger.info(
            f"⚡ K={session.strike_used:,.2f} ({session.strike_source}) | "
            f"BTC={self.feeds.btc_price:,.2f} | "
            f"Divergence: ${self.feeds.divergence:.0f}"
        )
        
        # Phase 3: MONITORING — scan for opportunities
        session.state = WindowState.MONITORING
        traded = False
        
        while self._running and not session.should_stop_trading:
            now = time.time()
            btc_price = self.feeds.btc_price
            
            if btc_price <= 0:
                await asyncio.sleep(scan_interval)
                continue
            
            # Calculate BS fair value
            T_remaining = (session.window_end - now) / SECONDS_PER_YEAR
            sigma = 0.45  # TODO: use rolling realized vol from feeds
            session.sigma = sigma
            
            p_up_fair, p_down_fair = bs_binary_price(
                btc_price, session.strike_used, sigma, T_remaining
            )
            
            # Get market prices (WebSocket if available, else REST)
            up_ask, down_ask = 1.0, 1.0
            up_bid, down_bid = 0.0, 0.0
            
            if len(session.token_ids) >= 2:
                up_book = self.feeds.get_book(session.token_ids[0])
                down_book = self.feeds.get_book(session.token_ids[1])
                
                if up_book:
                    up_ask = up_book.best_ask
                    up_bid = up_book.best_bid
                if down_book:
                    down_ask = down_book.best_ask
                    down_bid = down_book.best_bid
            
            # If no WS data, fall back to REST
            if up_ask >= 0.99 and down_ask >= 0.99:
                summary = await poly_client.get_market_summary(session.market)
                up_data = summary.get('Up', {})
                down_data = summary.get('Down', {})
                up_ask = up_data.get('ask', 1.0)
                down_ask = down_data.get('ask', 1.0)
                up_bid = up_data.get('bid', 0.0)
                down_bid = down_data.get('bid', 0.0)
            
            # Calculate edges (including taker fees)
            fee_up = taker_fee(up_ask)
            fee_down = taker_fee(down_ask)
            edge_up = p_up_fair - up_ask - fee_up
            edge_down = p_down_fair - down_ask - fee_down
            
            best_edge = max(edge_up, edge_down)
            best_side = 'Up' if edge_up > edge_down else 'Down'
            
            # Log periodically
            t_left = int(session.time_remaining)
            if t_left % 30 < scan_interval + 0.5:
                delta_pct = (btc_price - session.strike_used) / session.strike_used * 100
                logger.info(
                    f"  [{t_left:3d}s] BTC ${btc_price:,.2f} | Δ={delta_pct:+.3f}% | "
                    f"BS: Up={p_up_fair:.3f} Dn={p_down_fair:.3f} | "
                    f"Ask: Up={up_ask:.3f} Dn={down_ask:.3f} | "
                    f"Edge: {best_side}={best_edge:+.4f}"
                )
            
            # Check for trading signal
            if best_edge > self.min_edge and not traded:
                signal = {
                    'side': best_side,
                    'edge': best_edge,
                    'fair': p_up_fair if best_side == 'Up' else p_down_fair,
                    'ask': up_ask if best_side == 'Up' else down_ask,
                    'fee': fee_up if best_side == 'Up' else fee_down,
                    'btc': btc_price,
                    'K': session.strike_used,
                    'T_sec': session.time_remaining,
                }
                session.signals.append(signal)
                
                if self.on_signal:
                    self.on_signal(signal)
                
                # Calculate Kelly bet size
                p_fair = signal['fair']
                ask = signal['ask']
                fee = signal['fee']
                bet_size = kelly_bet_size(
                    p_fair, ask, self.trader.positions.available_balance, fee=fee
                )
                
                if bet_size > 0:
                    session.state = WindowState.TRADING
                    
                    token_idx = 0 if best_side == 'Up' else 1
                    token_id = session.token_ids[token_idx] if token_idx < len(session.token_ids) else ""
                    
                    if token_id:
                        result = self.trader.buy_binary_token(
                            token_id=token_id,
                            outcome=best_side,
                            market_slug=session.slug,
                            fair_price=p_fair,
                            ask_price=ask,
                            bet_size_usd=bet_size,
                            use_limit=True,
                            neg_risk=True,
                        )
                        
                        if result.success:
                            traded = True
                            session.first_trade_at = time.time()
                            session.trades.append(result)
                            self.windows_traded += 1
                            
                            logger.info(
                                f"  🟢 TRADE: {best_side} {result.size_shares:.1f} shares "
                                f"@ {ask:.4f} = ${bet_size:.2f} | "
                                f"edge={best_edge:.4f} ({best_edge*100:.1f}%)"
                            )
                            
                            if self.on_trade:
                                self.on_trade(result, signal)
                        else:
                            logger.warning(f"  ❌ Trade failed: {result.error}")
            
            await asyncio.sleep(scan_interval)
        
        # Phase 4: CLOSING — cancel remaining orders
        session.state = WindowState.CLOSING
        session.closed_at = time.time()
        
        if session.market.get('slug'):
            self.trader.cancel_market_orders(market_slug=session.slug)
        
        # Phase 5: SETTLEMENT
        # Wait for window to fully expire
        if session.time_remaining > 0:
            await asyncio.sleep(session.time_remaining + 2)
        
        # Determine settlement (BTC above or below K?)
        final_btc = self.feeds.chainlink_price or self.feeds.btc_price
        btc_went_up = final_btc > session.strike_used
        
        # Settle positions
        for trade in session.trades:
            if trade.success and trade.token_id:
                won = (trade.outcome == 'Up' and btc_went_up) or \
                      (trade.outcome == 'Down' and not btc_went_up)
                self.trader.positions.record_settlement(trade.token_id, won)
                
                # Record in risk manager
                pnl = trade.size_shares - trade.size_usd if won else -trade.size_usd
                session.pnl += pnl
                self.total_pnl += pnl
                
                self.trader.risk.record_trade(TradeRecord(
                    timestamp=time.time(),
                    market_slug=session.slug,
                    side=trade.outcome,
                    entry_price=trade.price,
                    size_usd=trade.size_usd,
                    p_true=0,  # TODO
                    edge=0,    # TODO
                    result='win' if won else 'loss',
                    pnl=pnl,
                ))
                
                logger.info(
                    f"  {'✅' if won else '❌'} Settlement: {trade.outcome} "
                    f"{'WON' if won else 'LOST'} | P&L: ${pnl:+.2f} | "
                    f"BTC final: ${final_btc:,.2f} vs K: ${session.strike_used:,.2f}"
                )
        
        if not traded:
            session.state = WindowState.SKIPPED
        else:
            session.state = WindowState.SETTLED
        
        # Clean up WebSocket subscriptions for expired tokens
        if session.token_ids:
            self.feeds.clob.remove_tokens(session.token_ids)
        
        # Archive session
        self.session_history.append(session)
        
        # Keep history bounded
        if len(self.session_history) > 1000:
            self.session_history = self.session_history[-500:]
    
    def _print_summary(self):
        """Print session summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 WINDOW MANAGER SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"  Windows scanned: {self.windows_scanned}")
        logger.info(f"  Windows traded:  {self.windows_traded}")
        logger.info(f"  Total P&L:       ${self.total_pnl:+.2f}")
        
        if self.windows_traded > 0:
            wins = sum(1 for s in self.session_history if s.pnl > 0)
            losses = sum(1 for s in self.session_history if s.pnl < 0)
            logger.info(f"  Win/Loss:        {wins}W / {losses}L")
            logger.info(f"  Win rate:        {wins/(wins+losses)*100:.1f}%" if wins + losses > 0 else "  Win rate:        N/A")
        
        logger.info(f"  Positions: {self.trader.positions.summary()}")
        logger.info(f"  Risk:      {self.trader.risk.summary()}")
    
    def summary(self) -> dict:
        """Current state for external consumers."""
        return {
            'running': self._running,
            'current_window': self.current_session.slug if self.current_session else None,
            'current_state': self.current_session.state.value if self.current_session else None,
            'time_remaining': self.current_session.time_remaining if self.current_session else 0,
            'windows_scanned': self.windows_scanned,
            'windows_traded': self.windows_traded,
            'total_pnl': self.total_pnl,
            'trader': self.trader.summary(),
            'feeds': self.feeds.status(),
        }
