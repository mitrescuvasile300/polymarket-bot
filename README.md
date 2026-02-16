# Polymarket BTC 5-Min Binary Options Trading Bot

Automated trading bot for Polymarket BTC 5-minute binary options markets. Uses Black-Scholes pricing, Kelly Criterion position sizing, and real-time WebSocket feeds.

## Architecture

```
src/
├── pricing/          # Black-Scholes, Kelly Criterion, fee model
│   ├── black_scholes.py    # BS fair value for binary options
│   └── kelly.py            # Position sizing with fee adjustment
├── data/             # Market data and price feeds
│   ├── polymarket_api.py   # REST client (Gamma + CLOB)
│   ├── price_feeds.py      # BTC price + rolling vol
│   ├── window_tracker.py   # Strike (K) caching
│   └── websocket_feeds.py  # Real-time WebSocket (Binance, RTDS, CLOB book)
├── scanner/          # Market monitoring
│   └── btc_scanner.py      # Live opportunity scanner
├── execution/        # Trading engine
│   ├── auth.py             # Polymarket CLOB auth (L1 + L2)
│   ├── trader.py           # Order placement, cancellation, DRY_RUN mode
│   ├── positions.py        # Real-time position & inventory tracking
│   ├── heartbeat.py        # Keep-alive for open orders (10s timeout!)
│   └── window_manager.py   # Automated 5-min window lifecycle
├── risk/             # Risk management
│   └── manager.py          # Circuit breakers, position limits
└── backtesting/      # Historical simulation
    └── backtest_btc_5min.py

config/settings.py    # All tunable parameters
main.py               # Main entry point
```

## Quick Start

### Scanner Only (no risk, no wallet needed)
```bash
python src/scanner/btc_scanner.py --duration 5
```

### Paper Trading (DRY_RUN)
```bash
DRY_RUN=true python main.py --duration 60 --bankroll 215
```

### Live Trading
```bash
export POLYMARKET_PRIVATE_KEY=0x...
export DRY_RUN=false
python main.py --duration 60 --bankroll 215 --min-edge 0.03
```

## Strategy

**Market Making at Extreme Prices** (p > 0.80 or p < 0.20):
- Zero maker fees + earn rebates
- Black-Scholes pricing: `P_up ≈ N(ln(S/K) / σ√T)`
- Quarter-Kelly sizing: `f* = 0.25 × (p - c - fee) / (1 - c)`
- Cancel if BTC moves >0.05% from last quote

**Taker Trades** when BS mispricing > 3%:
- Fill-Or-Kill market orders for guaranteed fills
- Dynamic taker fee model: `fee = 0.25 × (p × (1-p))²`

## Key Components

### WebSocket Feeds (`src/data/websocket_feeds.py`)
- **Binance Trade Stream**: Sub-10ms BTC price updates
- **Polymarket RTDS**: Binance + Chainlink oracle (settlement source!)
- **CLOB Book Stream**: Real-time order book changes

### Window Lifecycle Manager (`src/execution/window_manager.py`)
Automated 5-minute cycle:
1. **Pre-open** (T-20s): Discover market, subscribe to book
2. **Open** (T+0s): Capture Chainlink K via RTDS
3. **Active** (T+0 to T+270s): Monitor for +EV, execute trades
4. **Close** (T+270s): Cancel all orders
5. **Settlement**: Record P&L, move to next window

### Risk Management
- Max 10% bankroll per trade (Quarter-Kelly)
- Daily stop loss: 15%
- Circuit breaker: 5 consecutive losses
- Heartbeat every 5s (Polymarket cancels all orders if missed for 10s)

## Tests
```bash
python3.12 -m unittest discover -v tests/
# 37 tests: 18 pricing + 19 execution
```

## Wallet
- **Address**: `0x5E46213E74652895b284922cd9c71F06c57f238d`
- **Network**: Polygon (Chain ID 137)
- **Status**: Needs USDC funding via bridge from SOL

## Dependencies
```
py-clob-client>=0.22   # Polymarket official Python SDK
aiohttp>=3.9           # Async HTTP + WebSocket
websockets>=12.0       # Additional WS support
scipy>=1.12            # Normal CDF for Black-Scholes
python-dotenv>=1.0     # Environment config
```
