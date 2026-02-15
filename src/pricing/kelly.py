"""
Kelly Criterion for binary option position sizing.

For a binary contract priced at c where true probability is p:
    f* = (p - c) / (1 - c)

The numerator (p - c) is the edge.
The denominator (1 - c) normalizes by maximum profit per dollar.
"""

from config.settings import KELLY_FRACTION, MAX_POSITION_PCT, MIN_SHARES


def kelly_fraction(p_true: float, market_price: float) -> float:
    """
    Full Kelly fraction for a binary bet.
    
    Args:
        p_true: Our estimated true probability
        market_price: Current market price (cost per share)
    
    Returns:
        Fraction of bankroll to bet (before applying quarter-Kelly)
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if p_true <= market_price:
        return 0.0
    
    return (p_true - market_price) / (1.0 - market_price)


def kelly_bet_size(
    p_true: float,
    market_price: float,
    bankroll: float,
    fraction: float = KELLY_FRACTION,
    max_pct: float = MAX_POSITION_PCT,
) -> float:
    """
    Calculate bet size using fractional Kelly.
    
    Returns dollar amount to bet, or 0 if below minimum.
    """
    full_kelly = kelly_fraction(p_true, market_price)
    bet_fraction = min(full_kelly * fraction, max_pct)
    bet_size = bankroll * bet_fraction
    
    # Check minimum order (5 shares × price per share)
    min_cost = MIN_SHARES * market_price
    if bet_size < min_cost:
        return 0.0
    
    return bet_size


def expected_value(p_true: float, market_price: float, fee: float = 0.0) -> float:
    """
    Expected value per dollar bet on a binary outcome.
    
    EV = p × (1/c - 1) - (1-p) - fee/c
    Simplified: EV = (p - c - fee) / c
    """
    if market_price <= 0:
        return 0.0
    
    gross_ev = (p_true - market_price) / market_price
    net_ev = gross_ev - fee / market_price
    return net_ev


def edge_summary(p_true: float, market_price: float, fee: float, bankroll: float) -> dict:
    """Complete edge analysis for a potential trade."""
    edge = p_true - market_price - fee
    ev = expected_value(p_true, market_price, fee)
    full_k = kelly_fraction(p_true, market_price)
    bet = kelly_bet_size(p_true, market_price, bankroll)
    
    return {
        'p_true': p_true,
        'market_price': market_price,
        'fee': fee,
        'gross_edge': p_true - market_price,
        'net_edge': edge,
        'ev_per_dollar': ev,
        'full_kelly': full_k,
        'quarter_kelly': full_k * KELLY_FRACTION,
        'bet_size': bet,
        'max_profit': bet * (1.0 / market_price - 1) if bet > 0 and market_price > 0 else 0,
        'max_loss': bet,
    }
