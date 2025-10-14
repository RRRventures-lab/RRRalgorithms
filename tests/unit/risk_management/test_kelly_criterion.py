from risk.sizing.kelly_criterion import (
import numpy as np
import os
import pytest
import sys

"""
Unit Tests for Kelly Criterion Position Sizing

Tests position sizing calculations, risk management, and edge cases.
Critical for capital preservation and optimal bet sizing.
"""


# Add risk management to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../worktrees/risk-management/src'))

    calculate_kelly_percentage,
    calculate_fractional_kelly,
    calculate_position_size,
    optimize_kelly_monte_carlo
)


class TestKellyCalculations:
    """Test basic Kelly Criterion calculations"""

    def test_positive_expectancy(self):
        """Test Kelly with positive expectancy"""
        win_rate = 0.60  # 60% win rate
        avg_win = 2.0    # Average win is 2x
        avg_loss = 1.0   # Average loss is 1x

        kelly_pct = calculate_kelly_percentage(win_rate, avg_win, avg_loss)

        assert kelly_pct > 0  # Should be positive
        assert kelly_pct < 1  # Should not exceed 100%
        assert 0.1 < kelly_pct < 0.4  # Should be reasonable range

    def test_negative_expectancy(self):
        """Test Kelly with negative expectancy"""
        win_rate = 0.40  # 40% win rate
        avg_win = 1.0    # Average win is 1x
        avg_loss = 2.0   # Average loss is 2x (bad!)

        kelly_pct = calculate_kelly_percentage(win_rate, avg_win, avg_loss)

        assert kelly_pct <= 0  # Should be zero or negative (don't trade!)

    def test_even_odds(self):
        """Test Kelly with 50-50 odds and equal payoffs"""
        win_rate = 0.50
        avg_win = 1.0
        avg_loss = 1.0

        kelly_pct = calculate_kelly_percentage(win_rate, avg_win, avg_loss)

        assert kelly_pct == 0  # Should not trade (no edge)

    def test_coin_flip_with_edge(self):
        """Test classic coin flip with edge example"""
        win_rate = 0.60  # 60% to win
        avg_win = 1.0    # Win 1:1
        avg_loss = 1.0   # Lose 1:1

        kelly_pct = calculate_kelly_percentage(win_rate, avg_win, avg_loss)

        expected_kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        assert abs(kelly_pct - expected_kelly) < 0.001

    def test_high_win_rate_large_wins(self):
        """Test high win rate with large wins"""
        win_rate = 0.70
        avg_win = 3.0
        avg_loss = 1.0

        kelly_pct = calculate_kelly_percentage(win_rate, avg_win, avg_loss)

        assert kelly_pct > 0.5  # Should be aggressive
        assert kelly_pct < 1.0  # But not > 100%


class TestFractionalKelly:
    """Test fractional Kelly implementation"""

    def test_half_kelly(self):
        """Test half-Kelly for risk reduction"""
        win_rate = 0.60
        avg_win = 2.0
        avg_loss = 1.0

        full_kelly = calculate_kelly_percentage(win_rate, avg_win, avg_loss)
        half_kelly = calculate_fractional_kelly(win_rate, avg_win, avg_loss, fraction=0.5)

        assert abs(half_kelly - full_kelly * 0.5) < 0.001

    def test_quarter_kelly(self):
        """Test quarter-Kelly for conservative sizing"""
        win_rate = 0.60
        avg_win = 2.0
        avg_loss = 1.0

        full_kelly = calculate_kelly_percentage(win_rate, avg_win, avg_loss)
        quarter_kelly = calculate_fractional_kelly(win_rate, avg_win, avg_loss, fraction=0.25)

        assert abs(quarter_kelly - full_kelly * 0.25) < 0.001

    def test_default_fraction(self):
        """Test default fraction (should be 0.25)"""
        win_rate = 0.60
        avg_win = 2.0
        avg_loss = 1.0

        fractional = calculate_fractional_kelly(win_rate, avg_win, avg_loss)
        full_kelly = calculate_kelly_percentage(win_rate, avg_win, avg_loss)

        assert abs(fractional - full_kelly * 0.25) < 0.001

    def test_negative_kelly_with_fraction(self):
        """Test fractional Kelly with negative expectancy"""
        win_rate = 0.30
        avg_win = 1.0
        avg_loss = 2.0

        fractional = calculate_fractional_kelly(win_rate, avg_win, avg_loss, fraction=0.5)

        assert fractional <= 0  # Should still be non-positive


class TestPositionSizeCalculation:
    """Test position size calculations"""

    def test_position_size_with_capital(self):
        """Test calculating position size with capital"""
        capital = 100000  # $100k
        kelly_pct = 0.10  # 10% Kelly

        position_size = calculate_position_size(capital, kelly_pct)

        assert position_size == 10000  # $10k position

    def test_position_size_with_max_limit(self):
        """Test position size respects maximum limit"""
        capital = 100000
        kelly_pct = 0.50  # 50% Kelly
        max_position_pct = 0.20  # But limit to 20%

        position_size = calculate_position_size(capital, kelly_pct, max_position_pct)

        assert position_size == 20000  # Capped at 20% = $20k

    def test_position_size_with_min_limit(self):
        """Test position size respects minimum limit"""
        capital = 100000
        kelly_pct = 0.005  # 0.5% Kelly
        min_position_pct = 0.01  # But minimum is 1%

        position_size = calculate_position_size(capital, kelly_pct, min_position_pct)

        assert position_size == 1000  # Raised to 1% = $1k

    def test_zero_kelly(self):
        """Test position size with zero Kelly (no trade)"""
        capital = 100000
        kelly_pct = 0.0

        position_size = calculate_position_size(capital, kelly_pct)

        assert position_size == 0

    def test_negative_kelly(self):
        """Test position size with negative Kelly (no trade)"""
        capital = 100000
        kelly_pct = -0.10

        position_size = calculate_position_size(capital, kelly_pct)

        assert position_size == 0


class TestKellyEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_win_rate(self):
        """Test with 0% win rate"""
        kelly_pct = calculate_kelly_percentage(0.0, 2.0, 1.0)
        assert kelly_pct <= 0

    def test_hundred_percent_win_rate(self):
        """Test with 100% win rate"""
        kelly_pct = calculate_kelly_percentage(1.0, 2.0, 1.0)
        assert kelly_pct > 0
        assert kelly_pct <= 1.0

    def test_zero_average_loss(self):
        """Test with zero average loss (can't lose)"""
        kelly_pct = calculate_kelly_percentage(0.60, 2.0, 0.0)
        assert kelly_pct > 0  # Should be very high

    def test_very_small_edge(self):
        """Test with very small edge"""
        win_rate = 0.51  # Barely above 50%
        avg_win = 1.0
        avg_loss = 1.0

        kelly_pct = calculate_kelly_percentage(win_rate, avg_win, avg_loss)

        assert 0 < kelly_pct < 0.05  # Should be very small

    def test_large_capital_small_kelly(self):
        """Test large capital with small Kelly"""
        capital = 10_000_000  # $10M
        kelly_pct = 0.001  # 0.1%

        position_size = calculate_position_size(capital, kelly_pct)

        assert position_size == 10_000  # $10k


class TestKellyMonteCarloOptimization:
    """Test Monte Carlo optimization for Kelly"""

    def test_monte_carlo_basic(self):
        """Test basic Monte Carlo optimization"""
        trade_history = [
            {'win': True, 'pnl': 2000},
            {'win': True, 'pnl': 1500},
            {'win': False, 'pnl': -1000},
            {'win': True, 'pnl': 1800},
            {'win': False, 'pnl': -900},
            {'win': True, 'pnl': 2200},
            {'win': True, 'pnl': 1600},
            {'win': False, 'pnl': -1100},
            {'win': True, 'pnl': 1900},
            {'win': True, 'pnl': 2100}
        ]

        optimal_kelly = optimize_kelly_monte_carlo(
            trade_history,
            initial_capital=100000,
            num_simulations=1000
        )

        assert 0 < optimal_kelly < 1
        assert optimal_kelly < 0.5  # Should be conservative

    def test_monte_carlo_all_wins(self):
        """Test Monte Carlo with all winning trades"""
        trade_history = [{'win': True, 'pnl': 1000} for _ in range(20)]

        optimal_kelly = optimize_kelly_monte_carlo(
            trade_history,
            initial_capital=100000,
            num_simulations=500
        )

        assert optimal_kelly > 0.1  # Should be more aggressive

    def test_monte_carlo_mostly_losses(self):
        """Test Monte Carlo with mostly losing trades"""
        trade_history = [
            {'win': True, 'pnl': 500} if i % 5 == 0 else {'win': False, 'pnl': -200}
            for i in range(20)
        ]

        optimal_kelly = optimize_kelly_monte_carlo(
            trade_history,
            initial_capital=100000,
            num_simulations=500
        )

        assert optimal_kelly < 0.1  # Should be very conservative or zero


class TestKellyWithRealWorldScenarios:
    """Test Kelly with realistic trading scenarios"""

    def test_crypto_day_trading(self):
        """Test Kelly for typical crypto day trading"""
        win_rate = 0.55  # 55% win rate (good for day trading)
        avg_win = 1.5    # 1.5% average win
        avg_loss = 1.0   # 1% average loss

        kelly_pct = calculate_kelly_percentage(win_rate, avg_win, avg_loss)

        # Use fractional Kelly for safety
        position_pct = calculate_fractional_kelly(win_rate, avg_win, avg_loss, fraction=0.25)

        assert 0.02 < position_pct < 0.15  # Should be 2-15% per trade

    def test_swing_trading(self):
        """Test Kelly for swing trading"""
        win_rate = 0.60  # 60% win rate
        avg_win = 3.0    # 3% average win
        avg_loss = 1.5   # 1.5% average loss

        kelly_pct = calculate_kelly_percentage(win_rate, avg_win, avg_loss)
        position_pct = calculate_fractional_kelly(win_rate, avg_win, avg_loss, fraction=0.25)

        assert 0.05 < position_pct < 0.25  # Should be 5-25% per trade

    def test_high_frequency(self):
        """Test Kelly for high-frequency trading"""
        win_rate = 0.52  # 52% win rate (small edge)
        avg_win = 1.0    # 0.1% average win
        avg_loss = 0.95  # 0.095% average loss

        kelly_pct = calculate_kelly_percentage(win_rate, avg_win, avg_loss)
        position_pct = calculate_fractional_kelly(win_rate, avg_win, avg_loss, fraction=0.5)

        assert 0 < position_pct < 0.10  # Should be very small for HFT


class TestKellyParameterValidation:
    """Test parameter validation"""

    def test_invalid_win_rate_negative(self):
        """Test negative win rate"""
        with pytest.raises(ValueError):
            calculate_kelly_percentage(-0.10, 2.0, 1.0)

    def test_invalid_win_rate_above_one(self):
        """Test win rate > 1"""
        with pytest.raises(ValueError):
            calculate_kelly_percentage(1.5, 2.0, 1.0)

    def test_invalid_negative_avg_win(self):
        """Test negative average win"""
        with pytest.raises(ValueError):
            calculate_kelly_percentage(0.60, -2.0, 1.0)

    def test_invalid_negative_avg_loss(self):
        """Test negative average loss (should be positive)"""
        with pytest.raises(ValueError):
            calculate_kelly_percentage(0.60, 2.0, -1.0)

    def test_invalid_fraction_negative(self):
        """Test negative fractional Kelly"""
        with pytest.raises(ValueError):
            calculate_fractional_kelly(0.60, 2.0, 1.0, fraction=-0.5)

    def test_invalid_fraction_above_one(self):
        """Test fractional Kelly > 1"""
        with pytest.raises(ValueError):
            calculate_fractional_kelly(0.60, 2.0, 1.0, fraction=1.5)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
