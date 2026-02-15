"""
Bot configuration settings.
All tunable parameters in one place.
"""
import os

# ============================================================
# API Endpoints
# ============================================================
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS = "wss://rtds.polymarket.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@trade"
BINANCE_REST = "https://api.binance.com/api/v3"
KRAKEN_REST = "https://api.kraken.com/0/public"

# ============================================================
# Trading Parameters
# ============================================================
BANKROLL = float(os.getenv("BANKROLL", "215.0"))       # USDC
KELLY_FRACTION = 0.25          # Quarter-Kelly
MIN_EDGE = 0.03                # 3% minimum edge to trade
MAX_POSITION_PCT = 0.10        # Max 10% of bankroll per trade
MIN_SHARES = 5                 # Polymarket minimum order

# ============================================================
# Risk Management
# ============================================================
DAILY_STOP_LOSS = 0.15         # Stop if down 15% on the day
CONSECUTIVE_LOSS_LIMIT = 5     # Circuit breaker after N consecutive losses
MAX_DAILY_TRADES = 50          # Cap daily trades

# ============================================================
# Volatility
# ============================================================
VOL_LOOKBACK = 100             # Rolling vol window (data points)
VOL_MIN = 0.15                 # Minimum annualized vol (15%)
VOL_MAX = 1.50                 # Maximum annualized vol (150%)
VOL_DEFAULT = 0.45             # Default vol if insufficient data

# ============================================================
# Market Parameters
# ============================================================
WINDOW_SECONDS = 300           # 5 minutes
MARKETS_PER_DAY = 288          # 24*60/5
STOP_BEFORE_EXPIRY = 30        # Stop trading 30s before window close
SCAN_INTERVAL = 5              # Seconds between scans

# ============================================================
# Authentication (set via environment variables)
# ============================================================
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
POLYGON_CHAIN_ID = 137

# ============================================================
# Deployment
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "historical")
