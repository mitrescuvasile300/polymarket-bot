# Polymarket BTC 5-Minute Binary Options Trading Bot

Algorithmic trading bot for Polymarket's BTC 5-minute binary options markets.

## Architecture

```
polymarket-bot/
├── src/
│   ├── scanner/          # Live market scanner & opportunity detection
│   ├── pricing/          # Black-Scholes binary pricing & Greeks
│   ├── data/             # Market data feeds (Binance, CLOB, RTDS)
│   ├── execution/        # Order placement & management
│   ├── risk/             # Kelly criterion, position limits, circuit breakers
│   └── backtesting/      # Historical backtesting framework
├── config/               # Configuration files
├── data/historical/      # Historical price & trade data
├── logs/                 # Trading logs
└── tests/                # Unit & integration tests
```

## How It Works

### The Market
Polymarket creates a new BTC binary option every 5 minutes, 24/7 (288 markets/day).
Each asks: **"Will BTC be higher or lower at the end of this 5-minute window?"**
- "Up" token pays $1.00 if BTC goes up, $0.00 if down
- "Down" token pays $1.00 if BTC goes down, $0.00 if up
- Settlement via Chainlink Data Streams on Polygon

### The Strategy
1. **Black-Scholes Pricing**: Calculate theoretical fair value using `P_up = N(ln(S/K) / σ√T)`
2. **Edge Detection**: Compare fair value to market prices, accounting for dynamic fees
3. **Kelly Sizing**: Optimal bet sizing via `f* = (p - c) / (1 - c)` (quarter-Kelly)
4. **Market Making Focus**: Zero maker fees + daily rebates at extreme prices (p > 0.80 or p < 0.20)

### Fee Structure
| Price | Taker Fee | Strategy |
|-------|-----------|----------|
| $0.50 | 1.56% | ❌ Avoid (fees > edge) |
| $0.80 | 0.64% | ⚠️ Only with strong edge |
| $0.90 | 0.20% | ✅ Sweet spot |
| $0.95 | 0.06% | ✅ Near-free |
| Maker | 0.00% | ✅ + rebates |

## Quick Start

### 1. Scanner (observe only, no trading)
```bash
python src/scanner/btc_scanner.py
```

### 2. Backtesting
```bash
python src/backtesting/backtest_btc_5min.py
```

### 3. Live Trading (requires wallet setup)
```bash
# Set environment variables
export POLYMARKET_PRIVATE_KEY=...
export POLYMARKET_FUNDER_ADDRESS=...
python src/execution/live_trader.py
```

## API Endpoints

| API | URL | Purpose |
|-----|-----|---------|
| CLOB | `https://clob.polymarket.com` | Trading, order book |
| Gamma | `https://gamma-api.polymarket.com` | Market discovery |
| RTDS WS | `wss://rtds.polymarket.com` | Binance + Chainlink prices |
| Binance WS | `wss://stream.binance.com:9443/ws/btcusdt@trade` | Price discovery |

## Key Formulas

**Black-Scholes Binary Call:**
```
P_up = N(d₂)
d₂ = ln(S/K) / (σ√T)
T = seconds_remaining / 31,536,000
```

**Dynamic Taker Fee:**
```
fee = 0.25 × (p × (1-p))²
```

**Kelly Criterion (Binary):**
```
f* = (p_true - market_price) / (1 - market_price)
bet = bankroll × f* × 0.25  # quarter-Kelly
```

## References
- [Polymarket CLOB API Docs](https://docs.polymarket.com)
- [py-clob-client](https://github.com/Polymarket/py-clob-client)
- [polymarket-client-sdk (Rust)](https://crates.io/crates/polymarket-client-sdk)
- arXiv:2508.03474 — $40M guaranteed arbitrage paper
