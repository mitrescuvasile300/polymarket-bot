"""
Real-time position and inventory tracking.

Tracks:
- Open positions (token holdings)
- Open orders that could fill
- Total exposure vs available balance
- P&L per position and aggregate
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single token position."""
    token_id: str
    outcome: str         # 'Up' or 'Down'
    market_slug: str
    shares: float = 0.0
    avg_entry: float = 0.0  # Average cost per share (incl. fees)
    total_cost: float = 0.0
    entry_time: float = 0.0
    window_end: float = 0.0  # Expiry timestamp
    
    @property
    def is_expired(self) -> bool:
        return time.time() >= self.window_end
    
    @property
    def max_payout(self) -> float:
        """Maximum payout if position wins ($1 per share)."""
        return self.shares
    
    @property
    def max_profit(self) -> float:
        """Maximum profit if position wins."""
        return self.shares - self.total_cost
    
    def mark_to_market(self, current_price: float) -> float:
        """Unrealized P&L at current market price."""
        return self.shares * current_price - self.total_cost
    
    def settle(self, won: bool) -> float:
        """
        Settle the position at expiry.
        
        Returns:
            P&L from settlement
        """
        if won:
            pnl = self.shares * 1.0 - self.total_cost  # $1 per share
        else:
            pnl = -self.total_cost
        return pnl


@dataclass
class OpenOrder:
    """Tracks an open order that hasn't filled yet."""
    order_id: str
    token_id: str
    outcome: str
    market_slug: str
    side: str            # 'BUY' or 'SELL'
    price: float
    size_shares: float
    size_usd: float      # Locked USDC for BUY orders
    placed_at: float
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.placed_at


@dataclass 
class PositionTracker:
    """
    Tracks all positions, open orders, and available balance.
    
    Prevents:
    - Overlapping orders on the same window
    - Exceeding USDC balance
    - Stale orders from previous windows
    """
    
    initial_balance: float = 0.0
    usdc_balance: float = 0.0
    positions: dict = field(default_factory=dict)      # token_id -> Position
    open_orders: dict = field(default_factory=dict)     # order_id -> OpenOrder
    settled_pnl: float = 0.0
    total_trades: int = 0
    
    @property
    def locked_in_orders(self) -> float:
        """USDC locked in open BUY orders."""
        return sum(
            o.size_usd for o in self.open_orders.values()
            if o.side == 'BUY'
        )
    
    @property
    def available_balance(self) -> float:
        """USDC available for new orders (balance minus locked)."""
        return max(0.0, self.usdc_balance - self.locked_in_orders)
    
    @property
    def total_exposure(self) -> float:
        """Total cost basis of all open positions."""
        return sum(p.total_cost for p in self.positions.values())
    
    @property
    def total_value(self) -> float:
        """Balance + position cost basis (at-risk value)."""
        return self.usdc_balance + self.total_exposure
    
    def has_position_for_window(self, window_start: int) -> bool:
        """Check if we already have a position for this window."""
        slug_prefix = f"btc-updown-5m-{window_start}"
        for p in self.positions.values():
            if p.market_slug.startswith(slug_prefix):
                return True
        return False
    
    def has_open_order_for_window(self, window_start: int) -> bool:
        """Check if we have an open order for this window."""
        slug_prefix = f"btc-updown-5m-{window_start}"
        for o in self.open_orders.values():
            if o.market_slug.startswith(slug_prefix):
                return True
        return False
    
    def can_place_order(self, size_usd: float) -> tuple[bool, str]:
        """Check if we can afford a new order."""
        if size_usd <= 0:
            return False, "Invalid size"
        if size_usd > self.available_balance:
            return False, f"Insufficient balance: need ${size_usd:.2f}, have ${self.available_balance:.2f}"
        return True, "OK"
    
    def record_order_placed(self, order: OpenOrder):
        """Record a new open order."""
        self.open_orders[order.order_id] = order
        logger.info(
            f"Order placed: {order.order_id[:8]}... "
            f"{order.side} {order.size_shares:.1f} {order.outcome} "
            f"@ {order.price:.4f} on {order.market_slug}"
        )
    
    def record_order_filled(self, order_id: str, fill_price: float, fill_shares: float):
        """
        Record that an order was filled (fully or partially).
        
        Creates or updates a Position and adjusts balance.
        """
        order = self.open_orders.get(order_id)
        if not order:
            logger.warning(f"Fill for unknown order: {order_id}")
            return
        
        cost = fill_shares * fill_price  # Approximate (real cost from CLOB response)
        
        if order.side == 'BUY':
            # Deduct USDC, create/update position
            self.usdc_balance -= cost
            
            existing = self.positions.get(order.token_id)
            if existing:
                # Average in
                total_shares = existing.shares + fill_shares
                total_cost = existing.total_cost + cost
                existing.shares = total_shares
                existing.total_cost = total_cost
                existing.avg_entry = total_cost / total_shares if total_shares > 0 else 0
            else:
                self.positions[order.token_id] = Position(
                    token_id=order.token_id,
                    outcome=order.outcome,
                    market_slug=order.market_slug,
                    shares=fill_shares,
                    avg_entry=fill_price,
                    total_cost=cost,
                    entry_time=time.time(),
                )
            
            self.total_trades += 1
            logger.info(
                f"Fill: BUY {fill_shares:.1f} {order.outcome} "
                f"@ {fill_price:.4f} = ${cost:.2f}"
            )
        
        elif order.side == 'SELL':
            # Add USDC, reduce position
            self.usdc_balance += cost
            
            existing = self.positions.get(order.token_id)
            if existing:
                existing.shares -= fill_shares
                existing.total_cost -= cost
                if existing.shares <= 0.001:
                    del self.positions[order.token_id]
            
            logger.info(
                f"Fill: SELL {fill_shares:.1f} {order.outcome} "
                f"@ {fill_price:.4f} = +${cost:.2f}"
            )
        
        # Remove filled order
        del self.open_orders[order_id]
    
    def record_order_cancelled(self, order_id: str):
        """Record order cancellation — frees locked balance."""
        if order_id in self.open_orders:
            order = self.open_orders[order_id]
            logger.info(f"Order cancelled: {order_id[:8]}... ({order.outcome})")
            del self.open_orders[order_id]
    
    def record_settlement(self, token_id: str, won: bool):
        """
        Record position settlement at window expiry.
        
        If won: receive $1 per share
        If lost: position worth $0
        """
        position = self.positions.get(token_id)
        if not position:
            return
        
        pnl = position.settle(won)
        
        if won:
            self.usdc_balance += position.shares  # $1 per share
        
        self.settled_pnl += pnl
        
        logger.info(
            f"Settlement: {position.outcome} on {position.market_slug} — "
            f"{'WON' if won else 'LOST'} — P&L: ${pnl:+.2f}"
        )
        
        del self.positions[token_id]
    
    def cleanup_expired(self):
        """Remove expired positions (unsettled — shouldn't happen in production)."""
        expired = [
            tid for tid, p in self.positions.items()
            if p.is_expired
        ]
        for tid in expired:
            logger.warning(f"Cleaning up expired unsettled position: {tid[:16]}...")
            del self.positions[tid]
    
    def cleanup_stale_orders(self, max_age_seconds: float = 600):
        """Cancel tracking of orders older than max_age (should be cancelled on exchange too)."""
        stale = [
            oid for oid, o in self.open_orders.items()
            if o.age_seconds > max_age_seconds
        ]
        for oid in stale:
            logger.warning(f"Cleaning up stale order: {oid[:8]}... (age: {self.open_orders[oid].age_seconds:.0f}s)")
            del self.open_orders[oid]
    
    def summary(self) -> dict:
        """Current state summary."""
        return {
            'usdc_balance': self.usdc_balance,
            'locked_in_orders': self.locked_in_orders,
            'available_balance': self.available_balance,
            'num_positions': len(self.positions),
            'num_open_orders': len(self.open_orders),
            'total_exposure': self.total_exposure,
            'settled_pnl': self.settled_pnl,
            'total_trades': self.total_trades,
            'positions': [
                {
                    'outcome': p.outcome,
                    'slug': p.market_slug,
                    'shares': p.shares,
                    'avg_entry': p.avg_entry,
                    'cost': p.total_cost,
                    'max_profit': p.max_profit,
                }
                for p in self.positions.values()
            ],
            'open_orders': [
                {
                    'id': o.order_id[:8],
                    'side': o.side,
                    'outcome': o.outcome,
                    'price': o.price,
                    'size': o.size_shares,
                    'age_s': int(o.age_seconds),
                }
                for o in self.open_orders.values()
            ],
        }
