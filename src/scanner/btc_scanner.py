#!/usr/bin/env python3
"""
Polymarket BTC 5-Minute Binary Options Scanner
================================================
Monitors live BTC 5-min markets, calculates Black-Scholes fair value,
and identifies +EV trading opportunities.

Usage:
    python src/scanner/btc_scanner.py [--duration MINUTES] [--interval SECONDS]
"""

import asyncio
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import aiohttp
from config.settings import (
    BANKROLL, MIN_EDGE, SCAN_INTERVAL, STOP_BEFORE_EXPIRY, LOG_DIR
)
from src.pricing.black_scholes import bs_binary_price, taker_fee, SECONDS_PER_YEAR
from src.pricing.kelly import kelly_bet_size, edge_summary
from src.data.price_feeds import BTCPriceFeed
from src.data.polymarket_api import PolymarketClient
from src.data.window_tracker import WindowTracker


class LiveScanner:
    """Real-time scanner for Polymarket BTC 5-min markets."""
    
    def __init__(self, bankroll: float = BANKROLL):
        self.btc = BTCPriceFeed()
        self.window_tracker = WindowTracker()
        self.bankroll = bankroll
        self.scan_count = 0
        self.opportunities: list[dict] = []
        self.all_scans: list[dict] = []
    
    async def analyze_market(
        self,
        client: PolymarketClient,
        market: dict,
        btc_price: float,
    ) -> dict | None:
        """Analyze a single market for trading opportunities."""
        now = time.time()
        window_start = market['window_start']
        window_end = market['window_end']
        
        # Skip if expired or not started
        if now >= window_end - STOP_BEFORE_EXPIRY:
            return None
        if now < window_start:
            return None
        
        T_remaining = (window_end - now) / SECONDS_PER_YEAR
        time_left_sec = window_end - now
        
        # Get market prices
        prices = await client.get_market_summary(market)
        up = prices.get('Up', {})
        down = prices.get('Down', {})
        
        if not up or not down:
            return None
        
        # Strike price K = BTC at window open
        # Use WindowTracker for best estimate
        up_mid = up.get('mid')
        K, k_source = self.window_tracker.get_best_strike(
            window_start,
            btc_price,
            market_up_mid=up_mid,
            sigma=self.btc.vol_annualized,
            time_remaining_sec=time_left_sec,
        )
        
        # Better K estimation: if we know σ and the market price, invert BS
        # For now, use the approximation. The scanner will flag when market
        # deviates from BS regardless of K accuracy.
        sigma = self.btc.vol_annualized
        
        # BS fair values
        p_up_fair, p_down_fair = bs_binary_price(btc_price, K, sigma, T_remaining)
        
        # Taker edges (buy at ask)
        up_ask = up.get('ask', 1.0)
        down_ask = down.get('ask', 1.0)
        fee_up = taker_fee(up_ask)
        fee_down = taker_fee(down_ask)
        
        edge_up = p_up_fair - up_ask - fee_up
        edge_down = p_down_fair - down_ask - fee_down
        
        # Maker edges (post at bid, zero fees)
        edge_up_maker = p_up_fair - up.get('bid', 0)
        edge_down_maker = p_down_fair - down.get('bid', 0)
        
        # Kelly sizing
        kelly_up = kelly_bet_size(p_up_fair, up_ask, self.bankroll)
        kelly_down = kelly_bet_size(p_down_fair, down_ask, self.bankroll)
        
        # Arbitrage check (buy both sides)
        arb_cost = up_ask + down_ask
        arb_profit = (1.0 - arb_cost) if arb_cost < 1.0 else 0.0
        
        return {
            'slug': market['slug'],
            'time_left': time_left_sec,
            'btc_price': btc_price,
            'strike': K,
            'strike_source': k_source,
            'btc_delta_pct': (btc_price - K) / K * 100 if K > 0 else 0,
            'sigma': sigma,
            'bs_up': p_up_fair,
            'bs_down': p_down_fair,
            'up': up,
            'down': down,
            'fee_up': fee_up,
            'fee_down': fee_down,
            'edge_up_taker': edge_up,
            'edge_down_taker': edge_down,
            'edge_up_maker': edge_up_maker,
            'edge_down_maker': edge_down_maker,
            'kelly_up': kelly_up,
            'kelly_down': kelly_down,
            'arb_profit': arb_profit,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
    
    def format_analysis(self, a: dict) -> str:
        """Pretty-print analysis result."""
        lines = []
        t = int(a['time_left'])
        up = a['up']
        down = a['down']
        
        delta_pct = a.get('btc_delta_pct', 0)
        k_src = a.get('strike_source', '?')
        
        lines.append(f"\n{'='*72}")
        lines.append(f"📊 {a['slug']} | ⏱️{t:3d}s | BTC ${a['btc_price']:,.2f} | K=${a.get('strike',0):,.2f} ({k_src})")
        lines.append(f"   Δ={delta_pct:+.3f}% | σ={a['sigma']:.1%}")
        lines.append(f"{'─'*72}")
        
        # Price table
        lines.append(f"  {'':8s} {'BS Fair':>8s} {'Bid':>8s} {'Ask':>8s} {'Mid':>8s} {'Spread':>8s} {'Depth':>8s}")
        up_mid_str = f"{up.get('mid', 0):.4f}" if up.get('mid') else "  N/A "
        dn_mid_str = f"{down.get('mid', 0):.4f}" if down.get('mid') else "  N/A "
        lines.append(f"  {'Up':8s} {a['bs_up']:>8.4f} {up.get('bid',0):>8.4f} {up.get('ask',1):>8.4f} {up_mid_str:>8s} {up.get('spread',0):>8.4f} {up.get('bid_depth_3',0):>8.0f}")
        lines.append(f"  {'Down':8s} {a['bs_down']:>8.4f} {down.get('bid',0):>8.4f} {down.get('ask',1):>8.4f} {dn_mid_str:>8s} {down.get('spread',0):>8.4f} {down.get('bid_depth_3',0):>8.0f}")
        
        # Edge analysis
        lines.append(f"\n  Taker:  Up {a['edge_up_taker']:+.4f} ({a['edge_up_taker']*100:+.1f}%) fee={a['fee_up']:.4f}  |  Down {a['edge_down_taker']:+.4f} ({a['edge_down_taker']*100:+.1f}%) fee={a['fee_down']:.4f}")
        lines.append(f"  Maker:  Up {a['edge_up_maker']:+.4f} ({a['edge_up_maker']*100:+.1f}%)            |  Down {a['edge_down_maker']:+.4f} ({a['edge_down_maker']*100:+.1f}%)")
        
        if a['kelly_up'] > 0:
            lines.append(f"  Kelly:  Up ${a['kelly_up']:.2f}")
        if a['kelly_down'] > 0:
            lines.append(f"  Kelly:  Down ${a['kelly_down']:.2f}")
        
        # Signals
        if a['arb_profit'] > 0:
            lines.append(f"  🔥 ARB: Buy both @ {up.get('ask',0)+down.get('ask',0):.4f} → profit {a['arb_profit']*100:.2f}%")
        
        # Wide spread warning
        if up.get('spread', 0) > 0.30 or down.get('spread', 0) > 0.30:
            lines.append(f"  ⚠️ WIDE SPREAD ({up.get('spread',0):.2f} / {down.get('spread',0):.2f}) — market likely illiquid or just opened")
        
        best_taker = max(a['edge_up_taker'], a['edge_down_taker'])
        if best_taker > MIN_EDGE:
            side = 'UP' if a['edge_up_taker'] > a['edge_down_taker'] else 'DOWN'
            price = up.get('ask', 1) if side == 'UP' else down.get('ask', 1)
            kelly = a['kelly_up'] if side == 'UP' else a['kelly_down']
            lines.append(f"  🟢 SIGNAL: Buy {side} @ {price:.4f} | edge={best_taker:.4f} | Kelly=${kelly:.2f}")
        else:
            lines.append(f"  ⚪ No signal (best taker edge: {best_taker:+.4f}, need >{MIN_EDGE:.4f})")
        
        return '\n'.join(lines)
    
    async def run_cycle(self, session: aiohttp.ClientSession, client: PolymarketClient):
        """Run one scan cycle."""
        self.scan_count += 1
        now_dt = datetime.now(timezone.utc)
        
        # Fetch BTC price
        btc_price = await self.btc.fetch_price(session)
        if btc_price <= 0:
            print(f"[{now_dt.strftime('%H:%M:%S')}] ⚠️ No BTC price")
            return
        
        # Track window strikes
        self.window_tracker.update_price(btc_price)
        
        stats = self.btc.get_5min_stats()
        print(f"\n[{now_dt.strftime('%H:%M:%S')} UTC] Scan #{self.scan_count} | BTC ${btc_price:,.2f} | σ={stats['vol_annualized']:.1%} | ±${stats['expected_move_usd']:.0f}/5min | src={stats['source']} | opps={len(self.opportunities)}")
        
        # Discover markets
        markets = await client.discover_current_markets()
        if not markets:
            print("  No active markets found")
            return
        
        for market in markets:
            analysis = await self.analyze_market(client, market, btc_price)
            if analysis:
                print(self.format_analysis(analysis))
                self.all_scans.append(analysis)
                
                best_edge = max(analysis['edge_up_taker'], analysis['edge_down_taker'])
                if best_edge > MIN_EDGE or analysis['arb_profit'] > 0:
                    self.opportunities.append(analysis)
    
    async def run(self, duration_minutes: int = 5, interval: int = SCAN_INTERVAL):
        """Run the scanner for a specified duration."""
        print(f"🚀 Polymarket BTC 5-Min Scanner")
        print(f"   Bankroll: ${self.bankroll:.2f} | Min Edge: {MIN_EDGE:.0%} | Quarter-Kelly")
        print(f"   Duration: {duration_minutes}min | Interval: {interval}s")
        print(f"   Price sources: Binance (primary) → Kraken (fallback)")
        
        end_time = time.time() + (duration_minutes * 60)
        
        async with aiohttp.ClientSession() as session:
            client = PolymarketClient(session)
            
            # Prime the price feed
            await self.btc.fetch_price(session)
            
            while time.time() < end_time:
                try:
                    await self.run_cycle(session, client)
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                
                await asyncio.sleep(interval)
        
        self._print_summary()
        self._save_log()
    
    def _print_summary(self):
        """Print scan session summary."""
        print(f"\n{'='*72}")
        print(f"📈 SESSION SUMMARY")
        print(f"{'='*72}")
        print(f"  Scans: {self.scan_count}")
        print(f"  Markets analyzed: {len(self.all_scans)}")
        print(f"  +EV opportunities: {len(self.opportunities)}")
        print(f"  BTC: ${self.btc.price:,.2f}")
        print(f"  Realized vol: {self.btc.vol_annualized:.1%}")
        
        if self.opportunities:
            print(f"\n  Top opportunities:")
            sorted_opps = sorted(
                self.opportunities,
                key=lambda x: max(x['edge_up_taker'], x['edge_down_taker']),
                reverse=True
            )
            for opp in sorted_opps[:10]:
                best = max(opp['edge_up_taker'], opp['edge_down_taker'])
                side = 'UP' if opp['edge_up_taker'] > opp['edge_down_taker'] else 'DOWN'
                print(f"    {opp['slug']} | {side} edge={best:+.4f} ({best*100:+.1f}%)")
    
    def _save_log(self):
        """Save scan results to log file."""
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(LOG_DIR, f"scan_{ts}.json")
        
        data = {
            'scan_count': self.scan_count,
            'total_opportunities': len(self.opportunities),
            'final_btc': self.btc.price,
            'realized_vol': self.btc.vol_annualized,
            'opportunities': self.opportunities,
            'all_scans': self.all_scans[-50:],  # Last 50 scans
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n  📁 Log saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Polymarket BTC 5-Min Scanner')
    parser.add_argument('--duration', type=int, default=5, help='Duration in minutes (default: 5)')
    parser.add_argument('--interval', type=int, default=SCAN_INTERVAL, help=f'Scan interval in seconds (default: {SCAN_INTERVAL})')
    parser.add_argument('--bankroll', type=float, default=BANKROLL, help=f'Bankroll in USDC (default: {BANKROLL})')
    args = parser.parse_args()
    
    scanner = LiveScanner(bankroll=args.bankroll)
    asyncio.run(scanner.run(duration_minutes=args.duration, interval=args.interval))


if __name__ == '__main__':
    main()
