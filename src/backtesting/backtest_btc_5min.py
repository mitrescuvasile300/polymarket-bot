#!/usr/bin/env python3
"""
Backtesting framework for BTC 5-minute binary options strategy.

Uses historical Kraken 1-minute OHLC to simulate 5-minute windows.
Each window: compare BTC price at start vs end → Up or Down wins.

Tests our Black-Scholes pricing + Kelly sizing against historical data.

KEY INSIGHT FROM LIVE SCANNING:
- ATM spreads are massive ($0.01-$0.99) — no edge at midmarket
- Edge appears ONLY when BTC moves decisively from K (window open price)
- At extreme prices (p > 0.80), fees drop to <0.64% and mispricings exist
- This backtest simulates: monitor BTC movement, enter when directional

Usage:
    python src/backtesting/backtest_btc_5min.py [--days DAYS] [--bankroll USD]
"""

import asyncio
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import aiohttp
from src.pricing.black_scholes import bs_binary_price, taker_fee, SECONDS_PER_YEAR
from src.pricing.kelly import kelly_bet_size, kelly_fraction
from config.settings import (
    BANKROLL, MIN_EDGE, KELLY_FRACTION, MAX_POSITION_PCT,
    DAILY_STOP_LOSS, CONSECUTIVE_LOSS_LIMIT, DATA_DIR, LOG_DIR
)


# ============================================================
# Data structures
# ============================================================

@dataclass
class BacktestTrade:
    window_ts: int
    btc_open: float
    btc_at_entry: float
    btc_close: float
    sigma: float
    time_into_window: float
    side: str
    entry_price: float
    bs_fair: float
    edge: float
    fee: float
    bet_size: float
    result: str
    pnl: float
    bankroll_after: float


@dataclass
class BacktestResult:
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
    windows_with_movement: int = 0
    skipped_no_edge: int = 0
    skipped_risk: int = 0


# ============================================================
# Data fetching
# ============================================================

async def fetch_kraken_ohlc_page(
    session: aiohttp.ClientSession,
    pair: str = "XBTUSD",
    interval: int = 1,
    since: int = 0,
) -> tuple[list[dict], int]:
    """Fetch a single page of OHLC from Kraken. Returns (candles, last_timestamp)."""
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
                last_ts = result.get('last', 0)
                for key in result:
                    if key != 'last':
                        candles = [{
                            'open_time': int(k[0]),
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4]),
                            'volume': float(k[6]),
                        } for k in result[key]]
                        return candles, last_ts
    except Exception as e:
        print(f"  Kraken fetch error: {e}")
    return [], 0


async def fetch_kraken_ohlc(
    session: aiohttp.ClientSession,
    pair: str = "XBTUSD",
    interval: int = 1,
    since: int = 0,
    end_ts: int = 0,
) -> list[dict]:
    """
    Fetch historical OHLC from Kraken with pagination.
    
    Kraken returns max 720 candles per request. We paginate using the 
    'last' field from each response as the 'since' for the next request.
    """
    all_candles = []
    cursor = since
    target_end = end_ts or int(time.time())
    max_pages = 20  # Safety limit
    
    for page in range(max_pages):
        candles, last_ts = await fetch_kraken_ohlc_page(session, pair, interval, cursor)
        
        if not candles:
            break
        
        # Filter to requested range
        filtered = [c for c in candles if c['open_time'] <= target_end]
        all_candles.extend(filtered)
        
        # Check if we've covered the range
        latest_ts = max(c['open_time'] for c in candles)
        if latest_ts >= target_end or last_ts == 0 or last_ts <= cursor:
            break
        
        cursor = last_ts
        print(f"  Page {page + 1}: {len(all_candles)} candles total (up to {datetime.fromtimestamp(latest_ts, tz=timezone.utc).strftime('%m-%d %H:%M')})")
        await asyncio.sleep(1)  # Rate limit (Kraken allows ~1 req/s for public)
    
    # Deduplicate by open_time
    seen = set()
    unique = []
    for c in all_candles:
        if c['open_time'] not in seen:
            seen.add(c['open_time'])
            unique.append(c)
    
    return sorted(unique, key=lambda c: c['open_time'])


# ============================================================
# Window construction
# ============================================================

def build_5min_windows(klines_1m: list[dict]) -> list[dict]:
    """Build synthetic 5-minute windows from 1-minute klines."""
    klines_1m.sort(key=lambda k: k['open_time'])
    
    by_window = {}
    for k in klines_1m:
        window_ts = (k['open_time'] // 300) * 300
        if window_ts not in by_window:
            by_window[window_ts] = []
        by_window[window_ts].append(k)
    
    windows = []
    for window_ts in sorted(by_window.keys()):
        candles = sorted(by_window[window_ts], key=lambda k: k['open_time'])
        if len(candles) < 4:
            continue
        
        open_price = candles[0]['open']
        close_price = candles[-1]['close']
        high = max(c['high'] for c in candles)
        low = min(c['low'] for c in candles)
        volume = sum(c['volume'] for c in candles)
        result = 'Up' if close_price > open_price else 'Down'
        
        # Store per-minute prices for intra-window simulation
        minute_prices = [(c['open_time'], c['open'], c['close'], c['high'], c['low']) for c in candles]
        
        windows.append({
            'window_ts': window_ts,
            'open': open_price,
            'close': close_price,
            'high': high,
            'low': low,
            'volume': volume,
            'result': result,
            'delta_pct': (close_price - open_price) / open_price * 100,
            'max_delta_pct': max(abs(high - open_price), abs(low - open_price)) / open_price * 100,
            'minute_prices': minute_prices,
        })
    
    return windows


def compute_rolling_vol(windows: list[dict], lookback: int = 100) -> list[float]:
    """Compute rolling annualized volatility from 5-min returns."""
    vols = []
    for i in range(len(windows)):
        start = max(0, i - lookback)
        recent = windows[start:i + 1]
        
        if len(recent) < 10:
            vols.append(0.45)
            continue
        
        returns = []
        for j in range(1, len(recent)):
            p0 = recent[j - 1]['close']
            p1 = recent[j]['close']
            if p0 > 0 and p1 > 0:
                returns.append(math.log(p1 / p0))
        
        if len(returns) < 5:
            vols.append(0.45)
            continue
        
        # Bessel-corrected variance (subtract mean, N-1 denominator)
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        periods_per_year = SECONDS_PER_YEAR / 300
        annual_vol = math.sqrt(variance * periods_per_year)
        vols.append(max(0.15, min(1.50, annual_vol)))
    
    return vols


# ============================================================
# Market price simulation
# ============================================================

def simulate_market_price(
    bs_fair_current: float,
    bs_fair_stale: float,
    spread_model: str = "realistic",
    market_efficiency: float = 0.85,
) -> tuple[float, float]:
    """
    Simulate realistic market bid/ask with market lag.
    
    The KEY insight: edge comes from latency.
    - bs_fair_current: our BS fair value using latest BTC price
    - bs_fair_stale: BS fair value the market is currently pricing at
      (based on BTC price from a few seconds/minutes ago)
    
    market_efficiency (0-1):
    - 1.0 = perfectly efficient (market = current BS, no edge ever)
    - 0.85 = market tracks ~85% of current BS + 15% stale
    - 0.70 = slow market, lots of stale pricing
    
    Returns (bid, ask) based on stale market mid.
    """
    # Market mid is a blend of current and stale BS
    market_mid = market_efficiency * bs_fair_current + (1 - market_efficiency) * bs_fair_stale
    
    # Spread depends on price level
    if spread_model == "realistic":
        if market_mid > 0.90 or market_mid < 0.10:
            half_spread = 0.01
        elif market_mid > 0.80 or market_mid < 0.20:
            half_spread = 0.02
        elif market_mid > 0.70 or market_mid < 0.30:
            half_spread = 0.03
        elif market_mid > 0.60 or market_mid < 0.40:
            half_spread = 0.04
        else:
            half_spread = 0.05
    elif spread_model == "tight":
        half_spread = 0.01
    else:
        half_spread = 0.05
    
    bid = max(0.01, market_mid - half_spread)
    ask = min(0.99, market_mid + half_spread)
    
    return (bid, ask)


# ============================================================
# Core backtest engine
# ============================================================

def run_backtest(
    windows: list[dict],
    bankroll: float = BANKROLL,
    min_edge: float = MIN_EDGE,
    strategy: str = "directional",
    spread_model: str = "realistic",
    min_btc_delta_pct: float = 0.03,
    market_efficiency: float = 0.85,  # How fast market tracks BTC
) -> BacktestResult:
    """
    Run backtest simulating intra-window entries.
    
    Strategy 'directional':
    - Wait for BTC to move >min_btc_delta_pct from window open
    - Calculate BS fair value at that point
    - If edge > min_edge (after fees), enter the trade
    - Evaluate at window close
    
    This simulates the real behavior: we watch BTC move, then trade
    when the binary is at an extreme price (where fees are low).
    """
    result = BacktestResult(starting_bankroll=bankroll)
    current_bankroll = bankroll
    peak = bankroll
    max_dd = 0.0
    consecutive_losses = 0
    
    vols = compute_rolling_vol(windows)
    
    for i, w in enumerate(windows):
        result.total_windows += 1
        sigma = vols[i]
        K = w['open']  # Strike = window open price
        
        if K <= 0:
            continue
        
        # Check if BTC moved enough during this window
        if w['max_delta_pct'] < min_btc_delta_pct:
            result.skipped_no_edge += 1
            continue
        
        result.windows_with_movement += 1
        
        # Simulate intra-window: find the best entry point
        best_trade = None
        
        prev_S = K  # Previous minute's BTC price (starts at open)
        
        for minute_idx, (ts, m_open, m_close, m_high, m_low) in enumerate(w['minute_prices']):
            # Skip first minute (need direction) and last minute (too close to expiry)
            if minute_idx < 1 or minute_idx >= len(w['minute_prices']) - 1:
                prev_S = m_close
                continue
            
            seconds_elapsed = (minute_idx + 1) * 60
            time_remaining = 300 - seconds_elapsed
            T_years = time_remaining / SECONDS_PER_YEAR
            
            if T_years <= 0:
                prev_S = m_close
                continue
            
            # Current BTC price (we see this in real-time)
            S = m_close
            delta_pct = (S - K) / K * 100
            
            if abs(delta_pct) < min_btc_delta_pct:
                prev_S = m_close
                continue
            
            # OUR BS fair value (using latest BTC price)
            p_up_fair, p_down_fair = bs_binary_price(S, K, sigma, T_years)
            
            # STALE BS fair value (what the market was pricing a moment ago)
            # Market makers update every few seconds, but there's always lag
            p_up_stale, p_down_stale = bs_binary_price(prev_S, K, sigma, T_years)
            
            # Decide direction
            if delta_pct > 0:
                side = 'Up'
                bs_fair = p_up_fair
                bs_stale = p_up_stale
            else:
                side = 'Down'
                bs_fair = p_down_fair
                bs_stale = p_down_stale
            
            # Strategy filter
            if strategy == "extreme_only" and bs_fair < 0.75:
                prev_S = m_close
                continue
            
            # Simulate market with lag
            bid, ask = simulate_market_price(
                bs_fair, bs_stale, spread_model, market_efficiency
            )
            fee = taker_fee(ask)
            
            # Edge: our fair value vs what market is offering
            edge = bs_fair - ask - fee
            
            if edge > min_edge:
                bet = kelly_bet_size(bs_fair, ask, current_bankroll)
                if bet > 0:
                    best_trade = {
                        'minute_idx': minute_idx,
                        'S': S,
                        'side': side,
                        'bs_fair': bs_fair,
                        'bs_stale': bs_stale,
                        'ask': ask,
                        'fee': fee,
                        'edge': edge,
                        'bet': bet,
                        'time_remaining': time_remaining,
                        'delta_pct': delta_pct,
                    }
                    break  # Take first valid entry
            
            prev_S = m_close
        
        if best_trade is None:
            result.skipped_no_edge += 1
            continue
        
        # Risk checks
        if consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            result.skipped_risk += 1
            break
        if current_bankroll < 5:
            result.skipped_risk += 1
            break
        daily_pnl = current_bankroll - result.starting_bankroll
        if daily_pnl < -DAILY_STOP_LOSS * result.starting_bankroll:
            result.skipped_risk += 1
            break
        
        t = best_trade
        
        # Determine outcome
        # FIX: real cost per share = ask + fee (fee charged on top of price)
        cost_per_share = t['ask'] + t['fee']
        shares = t['bet'] / cost_per_share
        
        won = (t['side'] == w['result'])
        if won:
            # Winning shares pay $1.00, we paid (ask + fee) per share
            pnl = shares * (1.0 - cost_per_share)
            consecutive_losses = 0
        else:
            # Losing shares pay $0.00, we lose entire cost
            pnl = -t['bet']
            consecutive_losses += 1
        
        current_bankroll += pnl
        
        if current_bankroll > peak:
            peak = current_bankroll
        dd = (peak - current_bankroll) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        
        trade = BacktestTrade(
            window_ts=w['window_ts'],
            btc_open=K,
            btc_at_entry=t['S'],
            btc_close=w['close'],
            sigma=sigma,
            time_into_window=300 - t['time_remaining'],
            side=t['side'],
            entry_price=t['ask'],
            bs_fair=t['bs_fair'],
            edge=t['edge'],
            fee=t['fee'],
            bet_size=t['bet'],
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
    
    if result.trades:
        returns = [t.pnl / t.bet_size if t.bet_size > 0 else 0 for t in result.trades]
        avg_ret = sum(returns) / len(returns)
        if len(returns) > 1:
            std_ret = math.sqrt(sum((r - avg_ret) ** 2 for r in returns) / (len(returns) - 1))
        else:
            std_ret = 1
        result.sharpe_ratio = avg_ret / std_ret * math.sqrt(len(returns)) if std_ret > 0 else 0
    
    return result


# ============================================================
# Reporting
# ============================================================

def print_report(result: BacktestResult, params: dict):
    """Print formatted backtest report."""
    print(f"\n{'='*72}")
    print(f"📊 BACKTEST REPORT — {params.get('strategy', 'directional')} strategy")
    print(f"{'='*72}")
    print(f"  Data:    {result.total_windows} windows | {result.windows_with_movement} with movement (>{params.get('min_delta', 0.03):.2f}%)")
    print(f"  Trades:  {result.total_trades} executed | {result.skipped_no_edge} skipped (no edge) | {result.skipped_risk} skipped (risk)")
    
    if result.total_trades > 0:
        wr = result.wins / result.total_trades * 100
        print(f"  Record:  {result.wins}W / {result.losses}L ({wr:.1f}% win rate)")
    else:
        print(f"  Record:  No trades executed")
    
    print(f"\n  {'Starting bankroll:':<22s} ${result.starting_bankroll:>10.2f}")
    print(f"  {'Final bankroll:':<22s} ${result.final_bankroll:>10.2f}")
    print(f"  {'Total P&L:':<22s} ${result.total_pnl:>+10.2f} ({result.total_pnl / result.starting_bankroll * 100:+.1f}%)")
    print(f"  {'Peak bankroll:':<22s} ${result.peak_bankroll:>10.2f}")
    print(f"  {'Max drawdown:':<22s} {result.max_drawdown:>10.1%}")
    print(f"  {'Sharpe ratio:':<22s} {result.sharpe_ratio:>10.2f}")
    
    if result.trades:
        pnls = [t.pnl for t in result.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        gross_win = sum(wins)
        gross_loss = sum(losses)
        profit_factor = abs(gross_win / gross_loss) if gross_loss != 0 else float('inf')
        
        print(f"\n  {'Avg win:':<22s} ${avg_win:>+10.2f}")
        print(f"  {'Avg loss:':<22s} ${avg_loss:>+10.2f}")
        print(f"  {'Profit factor:':<22s} {profit_factor:>10.2f}")
        print(f"  {'Avg edge at entry:':<22s} {sum(t.edge for t in result.trades) / len(result.trades) * 100:>10.2f}%")
        print(f"  {'Avg entry price:':<22s} ${sum(t.entry_price for t in result.trades) / len(result.trades):>10.3f}")
        
        # Edge distribution
        edges = [t.edge for t in result.trades]
        print(f"\n  Edge distribution:")
        for threshold in [0.03, 0.05, 0.08, 0.10, 0.15]:
            count = sum(1 for e in edges if e >= threshold)
            print(f"    ≥{threshold:.0%}: {count} trades ({count / len(edges) * 100:.0f}%)")
        
        # Equity curve
        print(f"\n  Equity curve (last 30 trades):")
        for t in result.trades[-30:]:
            dt = datetime.fromtimestamp(t.window_ts, tz=timezone.utc).strftime('%m-%d %H:%M')
            emoji = '✅' if t.result == 'win' else '❌'
            bar_len = max(1, int(t.bankroll_after / result.starting_bankroll * 20))
            bar = '█' * bar_len
            btc_delta = (t.btc_at_entry - t.btc_open) / t.btc_open * 100
            print(f"    {dt} {emoji} {t.side:4s} @{t.entry_price:.3f} bs={t.bs_fair:.3f} Δ={btc_delta:+.2f}% pnl=${t.pnl:+.2f} → ${t.bankroll_after:.2f} {bar}")


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description='Backtest BTC 5-min binary options')
    parser.add_argument('--days', type=int, default=3, help='Days of data (default: 3)')
    parser.add_argument('--bankroll', type=float, default=BANKROLL, help=f'Bankroll (default: {BANKROLL})')
    parser.add_argument('--min-edge', type=float, default=MIN_EDGE, help=f'Min edge (default: {MIN_EDGE})')
    parser.add_argument('--min-delta', type=float, default=0.03, help='Min BTC move %% to enter (default: 0.03)')
    parser.add_argument('--strategy', choices=['directional', 'any', 'extreme_only'], default='directional')
    parser.add_argument('--spread', choices=['tight', 'realistic', 'wide'], default='realistic')
    args = parser.parse_args()
    
    print(f"📥 Fetching {args.days} days of 1-minute BTC/USD from Kraken...")
    
    async with aiohttp.ClientSession() as session:
        now = int(time.time())
        start = now - (args.days * 86400)
        all_klines = await fetch_kraken_ohlc(session, since=start)
        print(f"  Fetched {len(all_klines)} candles")
    
    if not all_klines:
        print("❌ No data. Check connectivity.")
        return
    
    windows = build_5min_windows(all_klines)
    print(f"📊 Built {len(windows)} five-minute windows")
    
    if len(windows) < 20:
        print("❌ Not enough data")
        return
    
    # Data stats
    first_dt = datetime.fromtimestamp(windows[0]['window_ts'], tz=timezone.utc)
    last_dt = datetime.fromtimestamp(windows[-1]['window_ts'], tz=timezone.utc)
    up_count = sum(1 for w in windows if w['result'] == 'Up')
    avg_delta = sum(abs(w['delta_pct']) for w in windows) / len(windows)
    big_moves = sum(1 for w in windows if w['max_delta_pct'] >= args.min_delta)
    
    print(f"   Range: {first_dt.strftime('%Y-%m-%d %H:%M')} → {last_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"   Up: {up_count} ({up_count / len(windows) * 100:.1f}%) | Down: {len(windows) - up_count}")
    print(f"   Avg |Δ|: {avg_delta:.3f}% | Windows with >{args.min_delta:.2f}% move: {big_moves} ({big_moves / len(windows) * 100:.0f}%)")
    
    # Run multiple strategy variants
    strategies_to_test = [
        # Vary market efficiency (key parameter: how fast does the market react?)
        {'min_edge': 0.02, 'min_delta': 0.03, 'spread': 'realistic', 'efficiency': 0.70, 'label': 'Slow market (eff=70%, 2% edge)'},
        {'min_edge': 0.02, 'min_delta': 0.03, 'spread': 'realistic', 'efficiency': 0.80, 'label': 'Moderate lag (eff=80%, 2% edge)'},
        {'min_edge': 0.02, 'min_delta': 0.03, 'spread': 'realistic', 'efficiency': 0.90, 'label': 'Fast market (eff=90%, 2% edge)'},
        {'min_edge': 0.03, 'min_delta': 0.05, 'spread': 'realistic', 'efficiency': 0.80, 'label': 'Big moves only (eff=80%, 3% edge, 5¢+)'},
        {'min_edge': 0.01, 'min_delta': 0.02, 'spread': 'tight', 'efficiency': 0.70, 'label': 'Aggressive + slow mkt (eff=70%, 1% edge)'},
    ]
    
    print(f"\n🔬 Running {len(strategies_to_test)} strategy variants...")
    
    all_results = []
    for s in strategies_to_test:
        result = run_backtest(
            windows,
            bankroll=args.bankroll,
            min_edge=s['min_edge'],
            strategy=args.strategy,
            spread_model=s['spread'],
            min_btc_delta_pct=s['min_delta'],
            market_efficiency=s.get('efficiency', 0.85),
        )
        
        wr = result.wins / result.total_trades * 100 if result.total_trades > 0 else 0
        pnl_pct = result.total_pnl / args.bankroll * 100
        
        print(f"\n  {s['label']}:")
        print(f"    Trades: {result.total_trades} | WR: {wr:.0f}% | P&L: ${result.total_pnl:+.2f} ({pnl_pct:+.1f}%) | DD: {result.max_drawdown:.1%} | Sharpe: {result.sharpe_ratio:.2f}")
        
        all_results.append((s, result))
    
    # Show detailed report for the best strategy
    best_s, best_r = max(all_results, key=lambda x: x[1].total_pnl)
    print(f"\n\n{'🏆' * 5} BEST STRATEGY: {best_s['label']} {'🏆' * 5}")
    print_report(best_r, {**best_s, 'strategy': args.strategy})
    
    # Save all results
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(LOG_DIR, f"backtest_{ts}.json")
    
    save_data = {
        'run_time': datetime.now(timezone.utc).isoformat(),
        'data_range': f"{first_dt.isoformat()} to {last_dt.isoformat()}",
        'total_windows': len(windows),
        'strategies': [{
            'params': s,
            'summary': {
                'trades': r.total_trades,
                'wins': r.wins,
                'losses': r.losses,
                'win_rate': r.wins / r.total_trades if r.total_trades > 0 else 0,
                'pnl': r.total_pnl,
                'pnl_pct': r.total_pnl / args.bankroll * 100,
                'max_drawdown': r.max_drawdown,
                'sharpe': r.sharpe_ratio,
                'final_bankroll': r.final_bankroll,
            },
            'trades': [{
                'ts': t.window_ts,
                'btc_open': t.btc_open,
                'btc_entry': t.btc_at_entry,
                'btc_close': t.btc_close,
                'side': t.side,
                'entry': t.entry_price,
                'bs_fair': t.bs_fair,
                'edge': t.edge,
                'bet': t.bet_size,
                'result': t.result,
                'pnl': t.pnl,
                'bankroll': t.bankroll_after,
            } for t in r.trades],
        } for s, r in all_results],
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n📁 All results saved: {filepath}")


if __name__ == '__main__':
    asyncio.run(main())
