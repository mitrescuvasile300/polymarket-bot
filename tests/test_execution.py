"""
Tests for the execution module: positions, trader (dry run), heartbeat.
"""

import os
import sys
import time
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.execution.positions import PositionTracker, Position, OpenOrder
from src.execution.trader import Trader, OrderSide, OrderResult
from src.risk.manager import RiskManager


class TestPositionTracker(unittest.TestCase):
    """Test position and inventory tracking."""
    
    def setUp(self):
        self.tracker = PositionTracker(
            initial_balance=200.0,
            usdc_balance=200.0,
        )
    
    def test_initial_state(self):
        self.assertEqual(self.tracker.usdc_balance, 200.0)
        self.assertEqual(self.tracker.available_balance, 200.0)
        self.assertEqual(self.tracker.locked_in_orders, 0.0)
        self.assertEqual(self.tracker.total_exposure, 0.0)
    
    def test_can_place_order(self):
        ok, msg = self.tracker.can_place_order(100.0)
        self.assertTrue(ok)
        
        ok, msg = self.tracker.can_place_order(300.0)
        self.assertFalse(ok)
        self.assertIn("Insufficient", msg)
        
        ok, msg = self.tracker.can_place_order(0)
        self.assertFalse(ok)
    
    def test_order_lifecycle_buy(self):
        """Test: place order → fill → settlement."""
        # Place order
        order = OpenOrder(
            order_id="TEST-001",
            token_id="token_up_123",
            outcome="Up",
            market_slug="btc-updown-5m-1000",
            side="BUY",
            price=0.80,
            size_shares=25.0,
            size_usd=20.0,
            placed_at=time.time(),
        )
        self.tracker.record_order_placed(order)
        
        self.assertEqual(len(self.tracker.open_orders), 1)
        self.assertEqual(self.tracker.locked_in_orders, 20.0)
        self.assertEqual(self.tracker.available_balance, 180.0)
        
        # Fill order
        self.tracker.record_order_filled("TEST-001", 0.80, 25.0)
        
        self.assertEqual(len(self.tracker.open_orders), 0)
        self.assertEqual(len(self.tracker.positions), 1)
        self.assertEqual(self.tracker.usdc_balance, 180.0)  # 200 - 20
        
        pos = self.tracker.positions["token_up_123"]
        self.assertEqual(pos.shares, 25.0)
        self.assertEqual(pos.avg_entry, 0.80)
        self.assertEqual(pos.total_cost, 20.0)
    
    def test_settlement_win(self):
        """Test winning settlement."""
        # Create position directly
        self.tracker.positions["token_up"] = Position(
            token_id="token_up",
            outcome="Up",
            market_slug="btc-updown-5m-1000",
            shares=25.0,
            avg_entry=0.80,
            total_cost=20.0,
            entry_time=time.time(),
        )
        self.tracker.usdc_balance = 180.0  # After buying
        
        # Settle as win
        self.tracker.record_settlement("token_up", won=True)
        
        self.assertEqual(len(self.tracker.positions), 0)
        self.assertEqual(self.tracker.usdc_balance, 205.0)  # 180 + 25 ($1/share)
        self.assertEqual(self.tracker.settled_pnl, 5.0)     # 25 - 20 = $5 profit
    
    def test_settlement_loss(self):
        """Test losing settlement."""
        self.tracker.positions["token_up"] = Position(
            token_id="token_up",
            outcome="Up",
            market_slug="btc-updown-5m-1000",
            shares=25.0,
            avg_entry=0.80,
            total_cost=20.0,
            entry_time=time.time(),
        )
        self.tracker.usdc_balance = 180.0
        
        # Settle as loss
        self.tracker.record_settlement("token_up", won=False)
        
        self.assertEqual(len(self.tracker.positions), 0)
        self.assertEqual(self.tracker.usdc_balance, 180.0)  # No payout
        self.assertEqual(self.tracker.settled_pnl, -20.0)   # Lost full cost
    
    def test_window_duplicate_check(self):
        """Prevent duplicate positions in same window."""
        self.tracker.positions["token_x"] = Position(
            token_id="token_x",
            outcome="Up",
            market_slug="btc-updown-5m-1707000000",
            shares=10,
            avg_entry=0.8,
            total_cost=8.0,
        )
        
        self.assertTrue(self.tracker.has_position_for_window(1707000000))
        self.assertFalse(self.tracker.has_position_for_window(1707000300))
    
    def test_cancel_frees_locked_balance(self):
        """Cancelling an order frees locked USDC."""
        order = OpenOrder(
            order_id="CANCEL-001",
            token_id="tok",
            outcome="Down",
            market_slug="btc-updown-5m-1000",
            side="BUY",
            price=0.70,
            size_shares=14.0,
            size_usd=10.0,
            placed_at=time.time(),
        )
        self.tracker.record_order_placed(order)
        self.assertEqual(self.tracker.available_balance, 190.0)
        
        self.tracker.record_order_cancelled("CANCEL-001")
        self.assertEqual(self.tracker.available_balance, 200.0)
        self.assertEqual(len(self.tracker.open_orders), 0)
    
    def test_position_averaging(self):
        """Adding to existing position averages correctly."""
        # First buy
        order1 = OpenOrder("O1", "tok1", "Up", "slug1", "BUY", 0.80, 25, 20, time.time())
        self.tracker.record_order_placed(order1)
        self.tracker.record_order_filled("O1", 0.80, 25.0)
        
        # Second buy at different price
        order2 = OpenOrder("O2", "tok1", "Up", "slug1", "BUY", 0.90, 10, 9, time.time())
        self.tracker.record_order_placed(order2)
        self.tracker.record_order_filled("O2", 0.90, 10.0)
        
        pos = self.tracker.positions["tok1"]
        self.assertEqual(pos.shares, 35.0)
        self.assertAlmostEqual(pos.total_cost, 29.0)
        self.assertAlmostEqual(pos.avg_entry, 29.0 / 35.0, places=4)


class TestTraderDryRun(unittest.TestCase):
    """Test trader in DRY_RUN mode (no real orders)."""
    
    def setUp(self):
        # Trader with no real client (dry run)
        self.trader = Trader(
            client=None,  # No real client needed for dry run
            position_tracker=PositionTracker(initial_balance=200, usdc_balance=200),
            risk_manager=RiskManager(starting_bankroll=200, current_bankroll=200),
            dry_run=True,
        )
    
    def test_limit_order_dry_run(self):
        """Dry run limit order creates position."""
        result = self.trader.place_limit_order(
            token_id="token_up_abc",
            side=OrderSide.BUY,
            price=0.80,
            size_shares=25.0,
            outcome="Up",
            market_slug="btc-updown-5m-1000",
        )
        
        self.assertTrue(result.success)
        self.assertTrue(result.is_dry_run)
        self.assertIn("DRY", result.order_id)
        self.assertEqual(result.price, 0.80)
        self.assertEqual(result.size_shares, 25.0)
        
        # Position should be created (simulated fill)
        self.assertEqual(len(self.trader.positions.positions), 1)
    
    def test_risk_prevents_trade(self):
        """Risk manager blocks trade when halted."""
        self.trader.risk.is_halted = True
        self.trader.risk.halt_reason = "Test halt"
        
        result = self.trader.place_limit_order(
            token_id="tok",
            side=OrderSide.BUY,
            price=0.80,
            size_shares=25.0,
        )
        
        self.assertFalse(result.success)
        self.assertIn("Risk", result.error)
    
    def test_balance_prevents_trade(self):
        """Insufficient balance blocks trade."""
        result = self.trader.place_limit_order(
            token_id="tok",
            side=OrderSide.BUY,
            price=0.80,
            size_shares=500.0,  # $400 > $200 balance
        )
        
        self.assertFalse(result.success)
        self.assertIn("Balance", result.error)
    
    def test_minimum_shares_enforced(self):
        """Order below minimum shares is rejected."""
        result = self.trader.place_limit_order(
            token_id="tok",
            side=OrderSide.BUY,
            price=0.80,
            size_shares=2.0,  # Below MIN_SHARES=5
        )
        
        self.assertFalse(result.success)
        self.assertIn("minimum", result.error.lower())
    
    def test_cancel_all_dry_run(self):
        """Cancel all in dry run."""
        # Place some orders
        self.trader.place_limit_order("t1", OrderSide.BUY, 0.8, 10, "Up", "slug1")
        self.trader.place_limit_order("t2", OrderSide.BUY, 0.7, 10, "Down", "slug2")
        
        ok = self.trader.cancel_all_orders()
        self.assertTrue(ok)
    
    def test_buy_binary_token(self):
        """High-level buy_binary_token works in dry run."""
        result = self.trader.buy_binary_token(
            token_id="tok_up",
            outcome="Up",
            market_slug="btc-updown-5m-1000",
            fair_price=0.85,
            ask_price=0.80,
            bet_size_usd=10.0,
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "Up")
        self.assertTrue(result.is_dry_run)
    
    def test_summary(self):
        """Trader summary includes all components."""
        summary = self.trader.summary()
        self.assertIn('dry_run', summary)
        self.assertIn('positions', summary)
        self.assertIn('risk', summary)
        self.assertTrue(summary['dry_run'])


class TestPosition(unittest.TestCase):
    """Test Position dataclass methods."""
    
    def test_max_profit(self):
        pos = Position(
            token_id="t",
            outcome="Up",
            market_slug="s",
            shares=100,
            avg_entry=0.80,
            total_cost=80.0,
        )
        self.assertEqual(pos.max_payout, 100.0)
        self.assertEqual(pos.max_profit, 20.0)
    
    def test_mark_to_market(self):
        pos = Position(
            token_id="t",
            outcome="Up",
            market_slug="s",
            shares=100,
            avg_entry=0.80,
            total_cost=80.0,
        )
        # Price went up
        self.assertAlmostEqual(pos.mark_to_market(0.90), 10.0)
        # Price went down
        self.assertAlmostEqual(pos.mark_to_market(0.70), -10.0)
    
    def test_settle_win(self):
        pos = Position(token_id="t", outcome="Up", market_slug="s",
                       shares=100, total_cost=80)
        pnl = pos.settle(won=True)
        self.assertEqual(pnl, 20.0)
    
    def test_settle_loss(self):
        pos = Position(token_id="t", outcome="Up", market_slug="s",
                       shares=100, total_cost=80)
        pnl = pos.settle(won=False)
        self.assertEqual(pnl, -80.0)


if __name__ == '__main__':
    unittest.main()
