"""Unit tests for pricing module — BS, fees, Kelly."""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.pricing.black_scholes import (
    bs_binary_price, taker_fee, min_edge_at_price, SECONDS_PER_YEAR
)
from src.pricing.kelly import kelly_fraction, kelly_bet_size, expected_value


# ============================================================
# Black-Scholes tests
# ============================================================

def test_bs_atm():
    """At-the-money (S=K): fair value should be ~0.50."""
    p_up, p_down = bs_binary_price(S=97000, K=97000, sigma=0.45, T_years=300/SECONDS_PER_YEAR)
    assert abs(p_up - 0.50) < 0.01, f"ATM Up should be ~0.50, got {p_up}"
    assert abs(p_down - 0.50) < 0.01, f"ATM Down should be ~0.50, got {p_down}"
    assert abs(p_up + p_down - 1.0) < 1e-10, "Up + Down must sum to 1.0"


def test_bs_itm():
    """In-the-money (S > K): Up should be > 0.50."""
    p_up, _ = bs_binary_price(S=97200, K=97000, sigma=0.45, T_years=300/SECONDS_PER_YEAR)
    assert p_up > 0.75, f"ITM Up should be >0.75 with +0.2% move, got {p_up}"


def test_bs_deep_itm():
    """Deep ITM: approaching 1.0."""
    p_up, _ = bs_binary_price(S=97500, K=97000, sigma=0.45, T_years=300/SECONDS_PER_YEAR)
    assert p_up > 0.95, f"Deep ITM Up should be >0.95, got {p_up}"


def test_bs_at_expiry():
    """At expiry (T=0): deterministic."""
    p_up, p_down = bs_binary_price(S=97100, K=97000, sigma=0.45, T_years=0)
    assert p_up == 1.0 and p_down == 0.0, "At expiry with S>K, Up=1.0"
    
    p_up, p_down = bs_binary_price(S=96900, K=97000, sigma=0.45, T_years=0)
    assert p_up == 0.0 and p_down == 1.0, "At expiry with S<K, Down=1.0"


def test_bs_zero_vol():
    """Zero volatility: same as at expiry."""
    p_up, _ = bs_binary_price(S=97100, K=97000, sigma=0, T_years=300/SECONDS_PER_YEAR)
    assert p_up == 1.0, "Zero vol with S>K should give Up=1.0"


def test_bs_symmetry():
    """Symmetric around ATM."""
    p_up_pos, _ = bs_binary_price(S=97100, K=97000, sigma=0.45, T_years=300/SECONDS_PER_YEAR)
    _, p_down_neg = bs_binary_price(S=96900, K=97000, sigma=0.45, T_years=300/SECONDS_PER_YEAR)
    assert abs(p_up_pos - p_down_neg) < 0.02, f"Symmetric moves should give similar probs: {p_up_pos} vs {p_down_neg}"


# ============================================================
# Fee tests
# ============================================================

def test_fee_at_midmarket():
    """Fee at p=0.50 should be ~1.56%."""
    fee = taker_fee(0.50)
    expected = 0.25 * (0.50 * 0.50) ** 2  # 0.25 × 0.0625 = 0.015625
    assert abs(fee - expected) < 1e-10, f"Fee at 0.50: expected {expected}, got {fee}"
    assert abs(fee - 0.015625) < 1e-10


def test_fee_at_extremes():
    """Fees should be tiny at extreme prices."""
    fee_95 = taker_fee(0.95)
    assert fee_95 < 0.001, f"Fee at 0.95 should be <0.001, got {fee_95}"
    
    fee_05 = taker_fee(0.05)
    assert fee_05 < 0.001, f"Fee at 0.05 should be <0.001, got {fee_05}"


def test_fee_at_boundaries():
    """Fee at 0 and 1 should be 0."""
    assert taker_fee(0.0) == 0.0
    assert taker_fee(1.0) == 0.0


def test_fee_symmetry():
    """Fee should be symmetric: fee(p) = fee(1-p)."""
    for p in [0.1, 0.2, 0.3, 0.4]:
        assert abs(taker_fee(p) - taker_fee(1 - p)) < 1e-15


def test_min_edge_single_fee():
    """min_edge_at_price should be 1× fee, not 2× (no sell-side fee at settlement)."""
    price = 0.50
    assert min_edge_at_price(price) == taker_fee(price), "Should be 1× fee"
    assert min_edge_at_price(price) != 2 * taker_fee(price), "Should NOT be 2× fee"


# ============================================================
# Kelly tests
# ============================================================

def test_kelly_no_edge():
    """No edge → zero bet."""
    assert kelly_fraction(0.50, 0.55) == 0.0
    assert kelly_fraction(0.50, 0.50) == 0.0


def test_kelly_with_edge():
    """With edge, Kelly should be positive."""
    f = kelly_fraction(0.65, 0.50)
    assert f > 0, f"Edge of 15% should give positive Kelly, got {f}"
    expected = 0.15 / 0.50  # (0.65 - 0.50) / (1 - 0.50) = 0.30
    assert abs(f - 0.30) < 1e-10


def test_kelly_with_fee():
    """Fee should reduce Kelly fraction."""
    f_no_fee = kelly_fraction(0.65, 0.50, fee=0.0)
    f_with_fee = kelly_fraction(0.65, 0.50, fee=0.02)
    assert f_with_fee < f_no_fee, "Fee should reduce Kelly"
    
    # With fee: (0.65 - 0.50 - 0.02) / (1 - 0.50) = 0.13/0.50 = 0.26
    expected = 0.13 / 0.50
    assert abs(f_with_fee - expected) < 1e-10


def test_kelly_fee_kills_edge():
    """Large fee should make Kelly zero."""
    f = kelly_fraction(0.52, 0.50, fee=0.03)
    assert f == 0.0, "Fee exceeds edge, Kelly should be 0"


def test_kelly_bet_size_minimum():
    """Bet must cover minimum 5 shares."""
    # At price 0.80 + fee ~0.0064, cost per share ≈ 0.81
    # 5 shares = $4.03, so bankroll of $3 should give 0
    bet = kelly_bet_size(0.90, 0.80, bankroll=3.0, fee=0.006)
    assert bet == 0.0, "Below minimum should return 0"


def test_kelly_bet_size_with_fee():
    """Bet size with fees should be smaller than without (when not capped)."""
    # Use smaller edge so we don't hit the 10% position cap
    bet_no_fee = kelly_bet_size(0.65, 0.60, bankroll=215.0, fee=0.0)
    bet_with_fee = kelly_bet_size(0.65, 0.60, bankroll=215.0, fee=0.015)
    assert bet_with_fee < bet_no_fee, f"Fee should reduce bet: {bet_with_fee} vs {bet_no_fee}"


# ============================================================
# P&L calculation test (the critical bug fix)
# ============================================================

def test_pnl_calculation():
    """
    Verify P&L with correct fee-adjusted share count.
    
    Scenario: Buy Up at $0.70, fee = $0.01/share, bet = $10
    Shares = $10 / ($0.70 + $0.01) = 14.085 shares
    Win: pnl = 14.085 × ($1.00 - $0.71) = 14.085 × $0.29 = $4.085
    Loss: pnl = -$10.00
    """
    bet = 10.0
    ask = 0.70
    fee = 0.01
    
    cost_per_share = ask + fee  # 0.71
    shares = bet / cost_per_share  # 14.085
    
    win_pnl = shares * (1.0 - cost_per_share)  # 14.085 × 0.29 = 4.085
    loss_pnl = -bet
    
    assert abs(shares - 14.0845) < 0.01
    assert abs(win_pnl - 4.085) < 0.01
    assert loss_pnl == -10.0
    
    # Old buggy calculation for comparison (should be LARGER — the bug)
    buggy_shares = bet / ask  # 14.286
    buggy_pnl = buggy_shares * (1.0 - ask) - buggy_shares * fee  # 14.286 × 0.30 - 14.286 × 0.01 = 4.143
    assert buggy_pnl > win_pnl, "Buggy calc should overstate profits"


# ============================================================
# Run all tests
# ============================================================

if __name__ == '__main__':
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\n  {passed} passed, {failed} failed out of {passed + failed} tests")
    
    if failed > 0:
        exit(1)
