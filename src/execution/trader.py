"""
Order placement, cancellation, and monitoring.

Wraps py-clob-client with:
- DRY_RUN mode for paper trading
- Automatic fee rate resolution
- Order result tracking
- Position tracker integration
- Risk manager integration
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs,
    MarketOrderArgs,
    OrderType,
    PartialCreateOrderOptions,
    OpenOrderParams,
)

from config.settings import MIN_SHARES
from src.execution.positions import PositionTracker, OpenOrder
from src.execution.heartbeat import HeartbeatManager
from src.risk.manager import RiskManager, TradeRecord

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderResult:
    """Result of an order attempt."""
    success: bool
    order_id: str = ""
    side: str = ""
    token_id: str = ""
    outcome: str = ""
    market_slug: str = ""
    price: float = 0.0
    size_shares: float = 0.0
    size_usd: float = 0.0
    order_type: str = ""
    is_dry_run: bool = False
    error: str = ""
    raw_response: dict = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class Trader:
    """
    High-level trading interface.
    
    Features:
    - DRY_RUN mode: logs orders without sending
    - Integrates with PositionTracker and RiskManager
    - Automatic heartbeat management for open orders
    - Order monitoring and status checking
    """
    
    def __init__(
        self,
        client: ClobClient,
        position_tracker: Optional[PositionTracker] = None,
        risk_manager: Optional[RiskManager] = None,
        dry_run: bool = None,
    ):
        self.client = client
        self.positions = position_tracker or PositionTracker()
        self.risk = risk_manager or RiskManager()
        self.dry_run = dry_run if dry_run is not None else os.getenv("DRY_RUN", "true").lower() == "true"
        self.heartbeat: Optional[HeartbeatManager] = None
        
        # Trade history
        self.order_history: list[OrderResult] = []
        self._order_count = 0
    
    def enable_heartbeat(self):
        """Start heartbeat manager (required for open orders)."""
        if self.dry_run:
            logger.info("Heartbeat skipped in DRY_RUN mode")
            return
        self.heartbeat = HeartbeatManager(self.client)
        self.heartbeat.start()
    
    def disable_heartbeat(self):
        """Stop heartbeat manager."""
        if self.heartbeat:
            self.heartbeat.stop()
    
    # ================================================================
    # Limit Orders (GTC) — for market making
    # ================================================================
    
    def place_limit_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size_shares: float,
        outcome: str = "",
        market_slug: str = "",
        post_only: bool = True,
        neg_risk: bool = True,
    ) -> OrderResult:
        """
        Place a GTC limit order.
        
        For market making: use post_only=True to ensure maker fees (0%).
        
        Args:
            token_id: CLOB token ID
            side: BUY or SELL
            price: Limit price (0.01 - 0.99)
            size_shares: Number of shares
            outcome: 'Up' or 'Down' (for logging)
            market_slug: Market identifier (for logging)
            post_only: If True, order rejected if it would cross the spread
            neg_risk: Whether market uses neg-risk contract
        
        Returns:
            OrderResult
        """
        size_usd = size_shares * price
        
        # Pre-flight checks
        if size_shares < MIN_SHARES:
            return OrderResult(
                success=False,
                error=f"Below minimum: {size_shares} < {MIN_SHARES} shares",
                side=side.value,
                token_id=token_id,
            )
        
        # Risk check
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            return OrderResult(success=False, error=f"Risk: {reason}")
        
        # Position check (for BUY)
        if side == OrderSide.BUY:
            can_afford, reason = self.positions.can_place_order(size_usd)
            if not can_afford:
                return OrderResult(success=False, error=f"Balance: {reason}")
        
        # DRY RUN
        if self.dry_run:
            self._order_count += 1
            result = OrderResult(
                success=True,
                order_id=f"DRY-{self._order_count:06d}",
                side=side.value,
                token_id=token_id,
                outcome=outcome,
                market_slug=market_slug,
                price=price,
                size_shares=size_shares,
                size_usd=size_usd,
                order_type="GTC",
                is_dry_run=True,
            )
            logger.info(
                f"[DRY RUN] LIMIT {side.value} {size_shares:.1f} {outcome} "
                f"@ {price:.4f} = ${size_usd:.2f} | {market_slug}"
            )
            self.order_history.append(result)
            
            # Track in position tracker as if filled immediately (paper trading)
            self.positions.record_order_placed(OpenOrder(
                order_id=result.order_id,
                token_id=token_id,
                outcome=outcome,
                market_slug=market_slug,
                side=side.value,
                price=price,
                size_shares=size_shares,
                size_usd=size_usd,
                placed_at=time.time(),
            ))
            # In dry run, simulate immediate fill
            self.positions.record_order_filled(result.order_id, price, size_shares)
            
            return result
        
        # LIVE ORDER
        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size_shares,
                side=side.value,
            )
            
            options = PartialCreateOrderOptions(neg_risk=neg_risk)
            signed_order = self.client.create_order(order_args, options)
            response = self.client.post_order(
                signed_order,
                orderType=OrderType.GTC,
                post_only=post_only,
            )
            
            order_id = ""
            if isinstance(response, dict):
                order_id = response.get("orderID", response.get("id", ""))
            
            result = OrderResult(
                success=True,
                order_id=order_id,
                side=side.value,
                token_id=token_id,
                outcome=outcome,
                market_slug=market_slug,
                price=price,
                size_shares=size_shares,
                size_usd=size_usd,
                order_type="GTC",
                raw_response=response if isinstance(response, dict) else {'raw': str(response)},
            )
            
            logger.info(
                f"LIMIT {side.value} {size_shares:.1f} {outcome} "
                f"@ {price:.4f} = ${size_usd:.2f} | {market_slug} | id={order_id[:12]}"
            )
            
            # Track open order
            self.positions.record_order_placed(OpenOrder(
                order_id=order_id,
                token_id=token_id,
                outcome=outcome,
                market_slug=market_slug,
                side=side.value,
                price=price,
                size_shares=size_shares,
                size_usd=size_usd,
                placed_at=time.time(),
            ))
            
            self.order_history.append(result)
            return result
            
        except Exception as e:
            result = OrderResult(
                success=False,
                error=str(e),
                side=side.value,
                token_id=token_id,
                outcome=outcome,
                market_slug=market_slug,
                price=price,
                size_shares=size_shares,
            )
            logger.error(f"Order failed: {e}")
            self.order_history.append(result)
            return result
    
    # ================================================================
    # Market Orders (FOK) — for latency arbitrage / taker trades
    # ================================================================
    
    def place_market_order(
        self,
        token_id: str,
        side: OrderSide,
        amount: float,
        outcome: str = "",
        market_slug: str = "",
        worst_price: float = 0.0,
        neg_risk: bool = True,
    ) -> OrderResult:
        """
        Place a Fill-Or-Kill market order.
        
        For taker strategy: immediate execution or nothing.
        
        Args:
            token_id: CLOB token ID
            side: BUY or SELL
            amount: USDC amount (BUY) or shares (SELL)
            outcome: 'Up' or 'Down' (for logging)
            market_slug: Market identifier
            worst_price: Maximum price willing to pay (0 = market)
            neg_risk: Whether market uses neg-risk contract
        
        Returns:
            OrderResult
        """
        # Risk check
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            return OrderResult(success=False, error=f"Risk: {reason}")
        
        if side == OrderSide.BUY:
            can_afford, reason = self.positions.can_place_order(amount)
            if not can_afford:
                return OrderResult(success=False, error=f"Balance: {reason}")
        
        # DRY RUN
        if self.dry_run:
            self._order_count += 1
            result = OrderResult(
                success=True,
                order_id=f"DRY-MKT-{self._order_count:06d}",
                side=side.value,
                token_id=token_id,
                outcome=outcome,
                market_slug=market_slug,
                price=worst_price,
                size_usd=amount if side == OrderSide.BUY else 0,
                size_shares=amount if side == OrderSide.SELL else (amount / worst_price if worst_price > 0 else 0),
                order_type="FOK",
                is_dry_run=True,
            )
            logger.info(
                f"[DRY RUN] MARKET {side.value} ${amount:.2f} {outcome} "
                f"@ worst={worst_price:.4f} | {market_slug}"
            )
            self.order_history.append(result)
            
            # Simulate fill in paper trading
            if worst_price > 0:
                fill_shares = amount / worst_price if side == OrderSide.BUY else amount
                oid = result.order_id
                self.positions.record_order_placed(OpenOrder(
                    order_id=oid, token_id=token_id, outcome=outcome,
                    market_slug=market_slug, side=side.value, price=worst_price,
                    size_shares=fill_shares, size_usd=amount, placed_at=time.time(),
                ))
                self.positions.record_order_filled(oid, worst_price, fill_shares)
            
            return result
        
        # LIVE ORDER
        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=side.value,
                price=worst_price if worst_price > 0 else None,
                order_type=OrderType.FOK,
            )
            
            options = PartialCreateOrderOptions(neg_risk=neg_risk)
            signed_order = self.client.create_market_order(order_args, options)
            response = self.client.post_order(signed_order, orderType=OrderType.FOK)
            
            order_id = ""
            if isinstance(response, dict):
                order_id = response.get("orderID", response.get("id", ""))
            
            result = OrderResult(
                success=True,
                order_id=order_id,
                side=side.value,
                token_id=token_id,
                outcome=outcome,
                market_slug=market_slug,
                price=worst_price,
                size_usd=amount,
                order_type="FOK",
                raw_response=response if isinstance(response, dict) else {'raw': str(response)},
            )
            
            logger.info(
                f"MARKET {side.value} ${amount:.2f} {outcome} "
                f"@ worst={worst_price:.4f} | {market_slug} | id={order_id[:12]}"
            )
            
            self.order_history.append(result)
            return result
            
        except Exception as e:
            result = OrderResult(
                success=False,
                error=str(e),
                side=side.value,
                token_id=token_id,
            )
            logger.error(f"Market order failed: {e}")
            self.order_history.append(result)
            return result
    
    # ================================================================
    # Order Management
    # ================================================================
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Cancel order {order_id[:12]}")
            self.positions.record_order_cancelled(order_id)
            return True
        
        try:
            self.client.cancel(order_id)
            self.positions.record_order_cancelled(order_id)
            logger.info(f"Cancelled order {order_id[:12]}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed for {order_id[:12]}: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if self.dry_run:
            order_ids = list(self.positions.open_orders.keys())
            for oid in order_ids:
                self.positions.record_order_cancelled(oid)
            logger.info(f"[DRY RUN] Cancelled all {len(order_ids)} orders")
            return True
        
        try:
            self.client.cancel_all()
            # Clear local tracking
            order_ids = list(self.positions.open_orders.keys())
            for oid in order_ids:
                self.positions.record_order_cancelled(oid)
            logger.info(f"Cancelled all orders ({len(order_ids)} tracked)")
            return True
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return False
    
    def cancel_market_orders(self, market_slug: str = "", token_id: str = "") -> bool:
        """Cancel all orders for a specific market or token."""
        if self.dry_run:
            cancelled = []
            for oid, order in list(self.positions.open_orders.items()):
                if (market_slug and order.market_slug == market_slug) or \
                   (token_id and order.token_id == token_id):
                    self.positions.record_order_cancelled(oid)
                    cancelled.append(oid)
            logger.info(f"[DRY RUN] Cancelled {len(cancelled)} market orders")
            return True
        
        try:
            self.client.cancel_market_orders(market=market_slug, asset_id=token_id)
            # Clean up local tracking
            for oid, order in list(self.positions.open_orders.items()):
                if (market_slug and order.market_slug == market_slug) or \
                   (token_id and order.token_id == token_id):
                    self.positions.record_order_cancelled(oid)
            return True
        except Exception as e:
            logger.error(f"Cancel market orders failed: {e}")
            return False
    
    def get_open_orders(self, market: str = "") -> list[dict]:
        """Fetch open orders from CLOB (live check)."""
        if self.dry_run:
            return [o.__dict__ for o in self.positions.open_orders.values()]
        
        try:
            params = OpenOrderParams(market=market) if market else None
            response = self.client.get_orders(params)
            return response if isinstance(response, list) else []
        except Exception as e:
            logger.error(f"Get orders failed: {e}")
            return []
    
    def get_order_status(self, order_id: str) -> dict:
        """Check status of a specific order."""
        if self.dry_run:
            return {'order_id': order_id, 'status': 'FILLED (dry run)'}
        
        try:
            return self.client.get_order(order_id)
        except Exception as e:
            return {'error': str(e)}
    
    # ================================================================
    # High-level trading actions
    # ================================================================
    
    def buy_binary_token(
        self,
        token_id: str,
        outcome: str,
        market_slug: str,
        fair_price: float,
        ask_price: float,
        bet_size_usd: float,
        use_limit: bool = True,
        neg_risk: bool = True,
    ) -> OrderResult:
        """
        Buy a binary token (Up or Down) based on BS mispricing.
        
        For maker strategy: posts limit order at ask price (or slightly below)
        For taker strategy: sends FOK at current ask
        
        Args:
            token_id: CLOB token ID
            outcome: 'Up' or 'Down'
            market_slug: Market identifier
            fair_price: Our BS fair value
            ask_price: Current best ask
            bet_size_usd: Kelly-sized bet in USDC
            use_limit: True for GTC limit, False for FOK market
            neg_risk: Whether market uses neg-risk contract
        
        Returns:
            OrderResult
        """
        shares = bet_size_usd / ask_price
        
        if shares < MIN_SHARES:
            return OrderResult(
                success=False,
                error=f"Bet ${bet_size_usd:.2f} too small for {shares:.1f} shares (min {MIN_SHARES})",
            )
        
        if use_limit:
            return self.place_limit_order(
                token_id=token_id,
                side=OrderSide.BUY,
                price=ask_price,
                size_shares=shares,
                outcome=outcome,
                market_slug=market_slug,
                post_only=False,  # We want to fill, not just sit on book
                neg_risk=neg_risk,
            )
        else:
            return self.place_market_order(
                token_id=token_id,
                side=OrderSide.BUY,
                amount=bet_size_usd,
                outcome=outcome,
                market_slug=market_slug,
                worst_price=ask_price,
                neg_risk=neg_risk,
            )
    
    # ================================================================
    # Status & Reporting
    # ================================================================
    
    def summary(self) -> dict:
        """Complete trader status."""
        return {
            'dry_run': self.dry_run,
            'total_orders': len(self.order_history),
            'successful_orders': sum(1 for o in self.order_history if o.success),
            'failed_orders': sum(1 for o in self.order_history if not o.success),
            'positions': self.positions.summary(),
            'risk': self.risk.summary(),
            'heartbeat': self.heartbeat.status() if self.heartbeat else None,
        }
