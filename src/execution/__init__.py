"""
Execution module: authentication, trading, position tracking, and window lifecycle.

Components:
- auth: Polymarket CLOB authentication (L1 + L2)
- trader: Order placement, cancellation, monitoring
- positions: Real-time position and inventory tracking
- heartbeat: Keep-alive management to prevent order cancellation
- window_manager: Automated 5-min window cycling
"""

from .auth import PolymarketAuth
from .trader import Trader, OrderResult, OrderSide
from .positions import PositionTracker, Position
from .heartbeat import HeartbeatManager
from .window_manager import WindowLifecycleManager

__all__ = [
    'PolymarketAuth',
    'Trader',
    'OrderResult',
    'OrderSide',
    'PositionTracker',
    'Position',
    'HeartbeatManager',
    'WindowLifecycleManager',
]
