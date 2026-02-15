"""
Risk management: circuit breakers, position limits, and daily tracking.
"""

import time
from dataclasses import dataclass, field
from config.settings import (
    DAILY_STOP_LOSS, CONSECUTIVE_LOSS_LIMIT, MAX_DAILY_TRADES, MAX_POSITION_PCT
)


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: float
    market_slug: str
    side: str           # 'Up' or 'Down'
    entry_price: float
    size_usd: float
    p_true: float       # Our BS estimate
    edge: float         # Net edge at entry
    result: str = ""    # 'win', 'loss', or ''
    pnl: float = 0.0


@dataclass
class RiskManager:
    """
    Manages risk constraints and trade tracking.
    
    Circuit breakers:
    - Daily stop loss (% of starting bankroll)
    - Consecutive loss limit
    - Max daily trades
    - Max position size
    """
    
    starting_bankroll: float = 215.0
    current_bankroll: float = 215.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    is_halted: bool = False
    halt_reason: str = ""
    trades: list = field(default_factory=list)
    
    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed."""
        if self.is_halted:
            return False, f"HALTED: {self.halt_reason}"
        
        # Daily stop loss
        if self.daily_pnl < -DAILY_STOP_LOSS * self.starting_bankroll:
            self.is_halted = True
            self.halt_reason = f"Daily stop loss hit ({self.daily_pnl:.2f})"
            return False, self.halt_reason
        
        # Consecutive losses
        if self.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            self.is_halted = True
            self.halt_reason = f"Consecutive loss limit ({self.consecutive_losses})"
            return False, self.halt_reason
        
        # Max daily trades
        if self.trades_today >= MAX_DAILY_TRADES:
            return False, f"Max daily trades reached ({self.trades_today})"
        
        # Bankroll check
        if self.current_bankroll < 5:  # Minimum for 1 trade
            self.is_halted = True
            self.halt_reason = "Bankroll depleted"
            return False, self.halt_reason
        
        return True, "OK"
    
    def validate_position_size(self, size_usd: float) -> tuple[bool, str]:
        """Validate a proposed position size."""
        max_size = self.current_bankroll * MAX_POSITION_PCT
        if size_usd > max_size:
            return False, f"Size ${size_usd:.2f} exceeds max ${max_size:.2f}"
        if size_usd <= 0:
            return False, "Zero size"
        return True, "OK"
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade and update risk state."""
        self.trades.append(trade)
        self.trades_today += 1
        self.daily_pnl += trade.pnl
        self.current_bankroll += trade.pnl
        
        if trade.result == 'loss':
            self.consecutive_losses += 1
        elif trade.result == 'win':
            self.consecutive_losses = 0
    
    def reset_daily(self):
        """Reset daily counters (call at start of each day)."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.is_halted = False
        self.halt_reason = ""
        self.starting_bankroll = self.current_bankroll
    
    def summary(self) -> dict:
        """Get risk manager summary."""
        wins = sum(1 for t in self.trades if t.result == 'win')
        losses = sum(1 for t in self.trades if t.result == 'loss')
        total = wins + losses
        
        return {
            'bankroll': self.current_bankroll,
            'starting_bankroll': self.starting_bankroll,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.starting_bankroll * 100 if self.starting_bankroll > 0 else 0,
            'trades_today': self.trades_today,
            'total_trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / total * 100 if total > 0 else 0,
            'consecutive_losses': self.consecutive_losses,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
        }
