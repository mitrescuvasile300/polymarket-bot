#!/usr/bin/env python3
"""
Backtesting framework for BTC 5-minute binary options strategy.

Uses historical Binance 1-minute klines to simulate 5-minute windows.
Each window: compare BTC price at start vs end → Up or Down wins.

Tests our Black-Scholes pricing + Kelly sizing against historical data.

Usage:
    python src/backtesting/backtest_btc_5min.py [--days DAYS] [--bankroll USD]
"""

import asyncio
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import aiohttp
from src.pricing.black_scholes import bs_binary_price, taker_fee, SECONDS_PER_YEAR
from src.pricing.kelly import kelly_bet_size, kelly_fraction
from config.settings import (
    BANKROLL, MIN_EDGE, KELLY_FRACTION, MAX_POSITION_PCT,
    DAILY_STOP_LOSS, CONSECUTIVE_LOSS_LIMIT, DATA_DIR, LOG_DIR
)


@dataclass
class BacktestTrade:
    """Record of a simulated trade."""
    window_ts: int
    btc_open: float
    btc_at_entry: float
    btc_close: float
    sigma: float
    time_into_window: float  # Seconds after window open when we "enter"
    side: str               # 'Up' or 'Down'
    entry_price: float      # Market price we buy at (simulated)
    bs_fair: float          # Our BS fair value
    edge: float             # Net edge after fees
    fee: float
    bet_size: float
    result: str             # 'win' or 'loss'
    pnl: float
    bankroll_after: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: list = field(default_factory=list)
    starting_bankroll: float = 0.0
    final_bankroll: float = 0.0
    total_windows: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_bankroll: float = 0.0
    sharpe_ratio: float = 0.0


async def fetch_binance_klines(
    session: aiohttp.ClientSession,
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start_ms: int = 0,
    end_ms: int = 0,
    limit: int = 1000,
) -> list[dict]:
    """
    Fetch historical klines from Binance.
    
    Each kline: [open_time, open, high, low, close, volume, close_time, ...]
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms
    
    try:
        async with session.get(
            "https://api.binance.com/api/v3/klines",
            params=params,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status == 200:
                raw = await resp.json()
                return [{
                    'open_time': int(k[0]) // 1000,
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': int(k[6]) // 1000,
                } for k in raw]
            else:
                text = await resp.text()
                print(f"  Binance error {resp.status}: {text[:200]}")
    except Exception as e:
        print(f"  Binance fetch error: {e}")
    
    return []


async def fetch_kraken_ohlc(
    session: aiohttp.ClientSession,
    pair: str = "XBTUSD",
    interval: int = 1,  # minutes
    since: int = 0,
) -> list[dict]:
    """Fetch historical OHLC from Kraken (fallback if Binance blocked)."""
    params = {"pair": pair, "interval": interval}
    if since:
        params["since"] = since
    
    try:
        async with session.get(
            "https://api.kraken.com/0/public/OHLC",
            params=params,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                result = data.get('result', {})
                # Remove 'last' key
                ohlc_data = None
                for key in result:
                    if key != 'last':
                        ohlc_data = result[key]
                        break
                
                if ohlc_data:
                    return [{
                        'open_time': int(k[0]),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[6]),
                    } for k in ohlc_data]
    except Exception as e:
        print(f"  Kraken fetch error: {e}")
    
    return []


def build_5min_windows(klines_1m: list[dict]) -> list[dict]:
    """
    Build synthetic 5-minute windows from 1-minute klines.
    Each window: start price (open of first candle) vs end price (close of 5th candle).
    """
    # Sort by time
    klines_1m.sort(key=lambda k: k['open_time'])
    
    windows = []
    # Group into 5-minute blocks aligned to 300s boundaries
    by_window = {}
    for k in klines_1m:
        window_ts = (k['open_time'] // 300) * 300
        if window_ts not in by_window:
            by_window[window_ts] = []
        by_window[window_ts].append(k)
    
    for window_ts in sorted(by_window.keys()):
        candles = sorted(by_window[window_ts], key=lambda k: k['open_time'])
        if len(candles) < 4:  # Need at least 4 of 5 minutes
            continue
        
        open_price = candles[0]['open']
        close_price = candles[-1]['close']
        high = max(c['high'] for c in candles)
        low = min(c['low'] for c in candles)
        volume = sum(c['volume'] for c in candles)
        
        result = 'Up' if close_price > open_price else 'Down'
        
        # Also store intermediate prices for simulating mid-window entry
        mid_candles = candles[len(candles)//2:]  # Second half candles
        mid_price = mid_candles[0]['open'] if mid_candles else open_price
        
        windows.append({
            'window_ts': window_ts,
            'open': open_price,
            'close': close_price,
            'high': high,
            'low': low,
            'volume': volume,
            'result': result,
            'mid_price': mid_price,
            'candles': candles,
        })
    
    return windows


def compute_rolling_vol(windows: list[dict], lookback: int = 100) -> list[float]:
    """Compute rolling annualized volatility from 5-min returns."""
    vols = []
    for i in range(len(windows)):
        start = max(0, i - lookback)
        recent = windows[start:i+1]
        
        if len(recent) < 10:
            vols.append(0.45)  # Default
            continue
        
        returns = []
        for j in range(1, len(recent)):
            p0 = recent[j-1]['close']
            p1 = recent[j]['close']
            if p0 > 0 and p1 > 0:
                returns.append(math.log(p1 / p0))
        
        if len(returns) < 5:
            vols.append(0.45)
            continue
        
        variance = sum(r**2 for r in returns) / len(returns)
        # Annualize: 5-min periods, 105,120 per year
        periods_per_year = SECONDS_PER_YEAR / 300
        annual_vol = math.sqrt(variance * periods_per_year)
        annual_vol = max(0.15, min(1.50, annual_vol))
        vols.append(annual_vol)
    
    return vols


def run_backtest(
    windows: list[dict],
    bankroll: float = BANKROLL,
    entry_delay_seconds: int = 120,  # Enter 2 minutes into the window
    min_edge: float = MIN_EDGE,
) -> BacktestResult:
    """
    Run backtest on synthetic 5-minute windows.
    
    Strategy: At entry_delay_seconds after window open, compute BS fair value.
    If edge > min_edge, place a trade. Evaluate at window close.
    """
    result = BacktestResult(starting_bankroll=bankroll)
    current_bankroll = bankroll
    peak = bankroll
    max_dd = 0.0
    consecutive_losses = 0
    
    # Compute rolling volatility
    vols = compute_rolling_vol(windows)
    
    for i, w in enumerate(windows):
        result.total_windows += 1
        sigma = vols[i]
        
        # Simulate mid-window entry
        btc_open = w['open']         # K (strike)
        btc_at_entry = w['mid_price']  # S at entry time
        btc_close = w['close']
        actual_result = w['result']
        
        # Time remaining at entry
        time_remaining_sec = 300 - entry_delay_seconds
        T_years = time_remaining_sec / SECONDS_PER_YEAR
        
        if T_years <= 0 or btc_open <= 0:
            continue
        
        # Black-Scholes fair values
        p_up_fair, p_down_fair = bs_binary_price(btc_at_entry, btc_open, sigma, T_years)
        
        # Simulate market prices (fair value + noise/spread)
        # In reality, market is efficient ±2-5% of fair value
        # We simulate a market that's slightly mispriced
        market_spread = 0.04  # 4 cent spread
        up_ask = p_up_fair + market_spread / 2
        down_ask = p_down_fair + market_spread / 2
        
        # Check for edges
        fee_up = taker_fee(up_ask)
        fee_down = taker_fee(down_ask)
        edge_up = p_up_fair - up_ask - fee_up
        edge_down = p_down_fair - down_ask - fee_down
        
        # Decide trade
        best_side = None
        best_edge = 0
        best_price = 0
        best_fair = 0
        best_fee = 0
        
        if edge_up > min_edge and edge_up > edge_down:
            best_side = 'Up'
            best_edge = edge_up
            best_price = up_ask
            best_fair = p_up_fair
            best_fee = fee_up
        elif edge_down > min_edge:
            best_side = 'Down'
            best_edge = edge_down
            best_price = down_ask
            best_fair = p_down_fair
            best_fee = fee_down
        
        if best_side is None:
            continue
        
        # Risk checks
        if consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            break
        if current_bankroll < 5:
            break
        
        # Kelly sizing
        bet = kelly_bet_size(best_fair, best_price, current_bankroll)
        if bet <= 0:
            continue
        
        # Execute trade
        won = (best_side == actual_result)
        if won:
            pnl = bet * (1.0 / best_price - 1) - bet * best_fee / best_price
            consecutive_losses = 0
        else:
            pnl = -bet
            consecutive_losses += 1
        
        current_bankroll += pnl
        
        # Track drawdown
        if current_bankroll > peak:
            peak = current_bankroll
        dd = (peak - current_bankroll) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        
        trade = BacktestTrade(
            window_ts=w['window_ts'],
            btc_open=btc_open,
            btc_at_entry=btc_at_entry,
            btc_close=btc_close,
            sigma=sigma,
            time_into_window=entry_delay_seconds,
            side=best_side,
            entry_price=best_price,
            bs_fair=best_fair,
            edge=best_edge,
            fee=best_fee,
            bet_size=bet,
            result='win' if won else 'loss',
            pnl=pnl,
            bankroll_after=current_bankroll,
        )
        result.trades.append(trade)
        
        if won:
            result.wins += 1
        else:
            result.losses += 1
    
    result.total_trades = len(result.trades)
    result.final_bankroll = current_bankroll
    result.total_pnl = current_bankroll - bankroll
    result.max_drawdown = max_dd
    result.peak_bankroll = peak
    
    # Sharpe ratio (simplified)
    if result.trades:
        returns = [t.pnl / t.bet_size if t.bet_size > 0 else 0 for t in result.trades]
        avg_ret = sum(returns) / len(returns)
        std_ret = math.sqrt(sum((r - avg_ret)**2 for r in returns) / len(returns)) if len(returns) > 1 else 1
        result.sharpe_ratio = avg_ret / std_ret * math.sqrt(len(returns)) if std_ret > 0 else 0
    
    return result


def print_backtest_report(result: BacktestResult):
    """Print formatted backtest report."""
    print(f"\n{'='*72}")
    print(f"📊 BACKTEST REPORT")
    print(f"{'='*72}")
    print(f"  Windows analyzed: {result.total_windows}")
    print(f"  Trades executed:  {result.total_trades}")
    print(f"  Win rate:         {result.wins}/{result.total_trades} ({result.wins/result.total_trades*100:.1f}%)" if result.total_trades > 0 else "  No trades")
    print(f"\n  Starting bankroll: ${result.starting_bankroll:.2f}")
    print(f"  Final bankroll:    ${result.final_bankroll:.2f}")
    print(f"  Total P&L:         ${result.total_pnl:+.2f} ({result.total_pnl/result.starting_bankroll*100:+.1f}%)")
    print(f"  Peak bankroll:     ${result.peak_bankroll:.2f}")
    print(f"  Max drawdown:      {result.max_drawdown:.1%}")
    print(f"  Sharpe ratio:      {result.sharpe_ratio:.2f}")
    
    if result.trades:
        pnls = [t.pnl for t in result.trades]
        avg_win = sum(p for p in pnls if p > 0) / max(1, sum(1 for p in pnls if p > 0))
        avg_loss = sum(p for p in pnls if p < 0) / max(1, sum(1 for p in pnls if p < 0))
        print(f"  Avg win:           ${avg_win:+.2f}")
        print(f"  Avg loss:          ${avg_loss:+.2f}")
        print(f"  Profit factor:     {abs(sum(p for p in pnls if p > 0) / min(-0.01, sum(p for p in pnls if p < 0))):.2f}")
        
        # Equity curve snapshot
        print(f"\n  Equity curve (first 20 trades):")
        for t in result.trades[:20]:
            bar = '█' * int(t.bankroll_after / result.starting_bankroll * 20)
            emoji = '✅' if t.result == 'win' else '❌'
            print(f"    {emoji} {t.side:4s} @ {t.entry_price:.3f} edge={t.edge:+.3f} pnl=${t.pnl:+.2f} → ${t.bankroll_after:.2f} {bar}")


async def main():
    parser = argparse.ArgumentParser(description='Backtest BTC 5-min binary options strategy')
    parser.add_argument('--days', type=int, default=3, help='Days of historical data (default: 3)')
    parser.add_argument('--bankroll', type=float, default=BANKROLL, help=f'Starting bankroll (default: {BANKROLL})')
    parser.add_argument('--min-edge', type=float, default=MIN_EDGE, help=f'Minimum edge to trade (default: {MIN_EDGE})')
    parser.add_argument('--entry-delay', type=int, default=120, help='Entry delay in seconds (default: 120)')
    parser.add_argument('--source', choices=['binance', 'kraken'], default='kraken', help='Data source (default: kraken)')
    args = parser.parse_args()
    
    print(f"📥 Fetching {args.days} days of 1-minute BTC data from {args.source}...")
    
    async with aiohttp.ClientSession() as session:
        now = int(time.time())
        start = now - (args.days * 86400)
        
        if args.source == 'binance':
            # Fetch from Binance (may be blocked in some regions)
            all_klines = []
            cursor = start * 1000
            while cursor < now * 1000:
                klines = await fetch_binance_klines(
                    session,
                    start_ms=cursor,
                    end_ms=min(cursor + 1000 * 60000, now * 1000),
                )
                if not klines:
                    break
                all_klines.extend(klines)
                cursor = (klines[-1]['open_time'] + 60) * 1000
                print(f"  Fetched {len(all_klines)} candles...")
                await asyncio.sleep(0.5)  # Rate limit
        else:
            # Fetch from Kraken
            all_klines = await fetch_kraken_ohlc(session, since=start)
            print(f"  Fetched {len(all_klines)} candles from Kraken")
    
    if not all_klines:
        print("❌ No data fetched. Try a different source or check connectivity.")
        return
    
    # Build 5-minute windows
    windows = build_5min_windows(all_klines)
    print(f"📊 Built {len(windows)} five-minute windows")
    
    if len(windows) < 20:
        print("❌ Not enough data for meaningful backtest")
        return
    
    # Show data range
    first_dt = datetime.fromtimestamp(windows[0]['window_ts'], tz=timezone.utc)
    last_dt = datetime.fromtimestamp(windows[-1]['window_ts'], tz=timezone.utc)
    up_count = sum(1 for w in windows if w['result'] == 'Up')
    print(f"   Range: {first_dt.strftime('%Y-%m-%d %H:%M')} to {last_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"   Up: {up_count} ({up_count/len(windows)*100:.1f}%) | Down: {len(windows)-up_count} ({(len(windows)-up_count)/len(windows)*100:.1f}%)")
    
    # Run backtest
    print(f"\n🔬 Running backtest (bankroll=${args.bankroll}, min_edge={args.min_edge}, entry_delay={args.entry_delay}s)...")
    result = run_backtest(
        windows,
        bankroll=args.bankroll,
        entry_delay_seconds=args.entry_delay,
        min_edge=args.min_edge,
    )
    
    print_backtest_report(result)
    
    # Save results
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(LOG_DIR, f"backtest_{ts}.json")
    
    save_data = {
        'params': {
            'days': args.days,
            'bankroll': args.bankroll,
            'min_edge': args.min_edge,
            'entry_delay': args.entry_delay,
            'source': args.source,
        },
        'summary': {
            'windows': result.total_windows,
            'trades': result.total_trades,
            'wins': result.wins,
            'losses': result.losses,
            'win_rate': result.wins / result.total_trades if result.total_trades > 0 else 0,
            'starting_bankroll': result.starting_bankroll,
            'final_bankroll': result.final_bankroll,
            'pnl': result.total_pnl,
            'pnl_pct': result.total_pnl / result.starting_bankroll * 100 if result.starting_bankroll > 0 else 0,
            'max_drawdown': result.max_drawdown,
            'sharpe': result.sharpe_ratio,
        },
        'trades': [{
            'ts': t.window_ts,
            'side': t.side,
            'entry': t.entry_price,
            'bs_fair': t.bs_fair,
            'edge': t.edge,
            'bet': t.bet_size,
            'result': t.result,
            'pnl': t.pnl,
            'bankroll': t.bankroll_after,
        } for t in result.trades],
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n📁 Results saved: {filepath}")


if __name__ == '__main__':
    asyncio.run(main())
