#!/usr/bin/env python3
"""
Polymarket BTC 5-Min Binary Options Trading Bot
=================================================

Main entry point that ties together all components:
- Authentication (CLOB L1 + L2)
- WebSocket feeds (Binance, RTDS, CLOB book)
- Window lifecycle manager
- Risk management

Usage:
    # Paper trading (default)
    DRY_RUN=true python main.py --duration 60

    # Live trading (requires funded wallet + API creds)
    DRY_RUN=false POLYMARKET_PRIVATE_KEY=0x... python main.py --duration 60

    # Scanner only (no trading)
    python src/scanner/btc_scanner.py --duration 5
"""

import argparse
import asyncio
import logging
import os
import sys
import signal

from config.settings import BANKROLL, MIN_EDGE
from src.execution.auth import PolymarketAuth
from src.execution.trader import Trader
from src.execution.positions import PositionTracker
from src.execution.window_manager import WindowLifecycleManager
from src.data.websocket_feeds import PriceFeedManager
from src.risk.manager import RiskManager

# ============================================================
# Logging setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('bot')


def setup_signal_handlers(manager: WindowLifecycleManager):
    """Graceful shutdown on SIGINT/SIGTERM."""
    def handle_signal(signum, frame):
        logger.info(f"\n⚠️ Signal {signum} received — shutting down gracefully...")
        manager.stop()
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


async def run_bot(args):
    """Main bot loop."""
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
    
    logger.info(f"{'='*60}")
    logger.info(f"🤖 Polymarket BTC 5-Min Bot")
    logger.info(f"{'='*60}")
    logger.info(f"  Mode:     {'📝 PAPER TRADING' if dry_run else '💰 LIVE TRADING'}")
    logger.info(f"  Bankroll: ${args.bankroll:.2f}")
    logger.info(f"  Min edge: {args.min_edge:.1%}")
    logger.info(f"  Duration: {args.duration} minutes")
    logger.info(f"{'='*60}")
    
    # ============================================================
    # 1. Authentication
    # ============================================================
    client = None
    
    if not dry_run:
        auth = PolymarketAuth()
        status = auth.initialize()
        
        if not status.is_l2:
            logger.error(f"Authentication failed: {status.error}")
            logger.info("Falling back to DRY_RUN mode")
            dry_run = True
        else:
            logger.info(f"✅ Authenticated as {status.address}")
            health = auth.health_check()
            logger.info(f"   Server: {'OK' if health['server_ok'] else 'DOWN'}")
            logger.info(f"   Time: {health.get('server_time', 'N/A')}")
            
            balance = auth.get_balance_allowance()
            logger.info(f"   Balance: {balance}")
            
            client = auth.client
    
    # ============================================================
    # 2. Initialize components
    # ============================================================
    positions = PositionTracker(
        initial_balance=args.bankroll,
        usdc_balance=args.bankroll,
    )
    
    risk = RiskManager(
        starting_bankroll=args.bankroll,
        current_bankroll=args.bankroll,
    )
    
    trader = Trader(
        client=client,
        position_tracker=positions,
        risk_manager=risk,
        dry_run=dry_run,
    )
    
    feeds = PriceFeedManager()
    
    manager = WindowLifecycleManager(
        trader=trader,
        feeds=feeds,
        min_edge=args.min_edge,
        bankroll=args.bankroll,
        on_signal=lambda s: logger.info(f"  📡 Signal: {s['side']} edge={s['edge']:.4f}"),
        on_trade=lambda r, s: logger.info(f"  💰 Traded: {r.outcome} ${r.size_usd:.2f}"),
    )
    
    setup_signal_handlers(manager)
    
    # ============================================================
    # 3. Start feeds
    # ============================================================
    logger.info("Starting WebSocket feeds...")
    await feeds.start(use_binance_direct=True, use_rtds=True)
    
    # Wait a moment for initial prices
    await asyncio.sleep(3)
    logger.info(f"  BTC: ${feeds.btc_price:,.2f} | Chainlink: ${feeds.chainlink_price:,.2f}")
    
    # ============================================================
    # 4. Start heartbeat (live only)
    # ============================================================
    if not dry_run:
        trader.enable_heartbeat()
    
    # ============================================================
    # 5. Run the main loop
    # ============================================================
    try:
        await manager.run(
            duration_minutes=args.duration,
            scan_interval=args.scan_interval,
        )
    finally:
        # Clean shutdown
        logger.info("Shutting down...")
        trader.cancel_all_orders()
        trader.disable_heartbeat()
        await feeds.stop()
        
        # Final summary
        summary = manager.summary()
        logger.info(f"\n📊 Final Summary:")
        logger.info(f"  Windows: {summary['windows_scanned']} scanned, {summary['windows_traded']} traded")
        logger.info(f"  P&L: ${summary['total_pnl']:+.2f}")
        logger.info(f"  Positions: {trader.positions.summary()}")


def main():
    parser = argparse.ArgumentParser(description='Polymarket BTC 5-Min Trading Bot')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration in minutes (default: 60)')
    parser.add_argument('--bankroll', type=float, default=BANKROLL,
                        help=f'Bankroll in USDC (default: {BANKROLL})')
    parser.add_argument('--min-edge', type=float, default=MIN_EDGE,
                        help=f'Minimum edge threshold (default: {MIN_EDGE})')
    parser.add_argument('--scan-interval', type=float, default=1.0,
                        help='Seconds between price checks (default: 1.0)')
    args = parser.parse_args()
    
    asyncio.run(run_bot(args))


if __name__ == '__main__':
    main()
