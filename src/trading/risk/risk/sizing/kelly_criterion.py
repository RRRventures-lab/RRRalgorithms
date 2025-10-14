from dataclasses import dataclass
from typing import Dict, Optional
import logging
import numpy as np

"""
Kelly Criterion Position Sizing Calculator

Implements the Kelly Criterion formula for optimal position sizing
based on win rate, average win/loss ratios, and account constraints.

Formula: Kelly % = W - [(1-W) / R]
Where:
    W = Win rate (probability of winning)
    R = Win/Loss ratio (avg win / avg loss)

Includes safeguards for practical trading:
- Fractional Kelly (typically 0.25 to 0.5 of full Kelly)
- Maximum position size limits
- Minimum position size thresholds
"""


logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    recommended_size: float  # As fraction of portfolio (0.0 to 1.0)
    recommended_units: Optional[float]  # Number of units/shares to buy
    kelly_fraction: float  # Full Kelly percentage
    fractional_kelly: float  # Applied fractional Kelly
    risk_amount: float  # Dollar amount at risk
    notes: str  # Additional information or warnings


class KellyCriterion:
    """
    Kelly Criterion position sizing calculator

    Calculates optimal position sizes based on historical performance
    and risk parameters.
    """

    def __init__(
        self,
        max_position_size: float = 0.20,
        fractional_kelly: float = 0.25,
        min_position_size: float = 0.01,
        max_kelly_cap: float = 0.30
    ):
        """
        Initialize Kelly Criterion calculator

        Args:
            max_position_size: Maximum position as fraction of portfolio (default 20%)
            fractional_kelly: Fraction of full Kelly to use (default 0.25 = 25%)
            min_position_size: Minimum position size (default 1%)
            max_kelly_cap: Maximum Kelly percentage to cap at (default 30%)
        """
        self.max_position_size = max_position_size
        self.fractional_kelly = fractional_kelly
        self.min_position_size = min_position_size
        self.max_kelly_cap = max_kelly_cap

        logger.info(
            f"Kelly Criterion initialized: max_pos={max_position_size:.1%}, "
            f"fractional={fractional_kelly:.1%}, min_pos={min_position_size:.1%}"
        )

    def calculate_kelly_percentage(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly percentage

        Args:
            win_rate: Probability of winning (0.0 to 1.0)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)

        Returns:
            Kelly percentage as fraction (0.0 to 1.0)
        """
        # Validation
        if not 0 <= win_rate <= 1:
            raise ValueError(f"Win rate must be between 0 and 1, got {win_rate}")

        if avg_win <= 0:
            raise ValueError(f"Average win must be positive, got {avg_win}")

        if avg_loss <= 0:
            raise ValueError(f"Average loss must be positive, got {avg_loss}")

        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss

        # Kelly formula: W - [(1-W) / R]
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Cap at max_kelly_cap
        kelly_pct = min(kelly_pct, self.max_kelly_cap)

        # Kelly can be negative (indicating not to trade)
        kelly_pct = max(kelly_pct, 0.0)

        return kelly_pct

    def calculate_position_size(
        self,
        account_size: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_price: Optional[float] = None,
        stop_loss_pct: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate recommended position size

        Args:
            account_size: Total portfolio value in dollars
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive)
            current_price: Current asset price (optional, for unit calculation)
            stop_loss_pct: Stop loss as percentage (optional, for risk calculation)

        Returns:
            PositionSizeResult with recommendation details
        """
        notes = []

        # Calculate full Kelly
        kelly_pct = self.calculate_kelly_percentage(win_rate, avg_win, avg_loss)

        # Apply fractional Kelly
        fractional = kelly_pct * self.fractional_kelly

        # Apply position size limits
        if fractional > self.max_position_size:
            notes.append(
                f"Kelly ({fractional:.1%}) capped at max position size ({self.max_position_size:.1%})"
            )
            fractional = self.max_position_size

        if fractional < self.min_position_size and fractional > 0:
            notes.append(
                f"Kelly ({fractional:.1%}) below minimum, setting to {self.min_position_size:.1%}"
            )
            fractional = self.min_position_size

        # Warning for low win rates
        if win_rate < 0.40:
            notes.append(f"Warning: Low win rate ({win_rate:.1%}) - consider strategy review")

        # Warning for negative Kelly
        if kelly_pct <= 0:
            notes.append(
                f"Negative Kelly ({kelly_pct:.1%}) - strategy has negative expectancy, DO NOT TRADE"
            )
            fractional = 0.0

        # Calculate dollar amount
        position_value = account_size * fractional

        # Calculate units if price provided
        units = None
        if current_price is not None and current_price > 0:
            units = position_value / current_price

        # Calculate risk amount
        risk_amount = 0.0
        if stop_loss_pct is not None and stop_loss_pct > 0:
            risk_amount = position_value * stop_loss_pct
            risk_pct = (risk_amount / account_size) * 100
            notes.append(f"Risk per trade: ${risk_amount:.2f} ({risk_pct:.2f}% of account)")

        return PositionSizeResult(
            recommended_size=fractional,
            recommended_units=units,
            kelly_fraction=kelly_pct,
            fractional_kelly=fractional,
            risk_amount=risk_amount,
            notes="; ".join(notes) if notes else "Position size looks good"
        )

    def calculate_from_history(
        self,
        trade_history: list,
        account_size: float,
        current_price: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate position size from trade history

        Args:
            trade_history: List of trade P&L values (positive for wins, negative for losses)
            account_size: Current account size
            current_price: Current asset price (optional)

        Returns:
            PositionSizeResult with recommendation
        """
        if not trade_history:
            raise ValueError("Trade history is empty")

        # Separate wins and losses
        wins = [t for t in trade_history if t > 0]
        losses = [abs(t) for t in trade_history if t < 0]

        if not wins and not losses:
            raise ValueError("No wins or losses in trade history")

        # Calculate statistics
        total_trades = len(trade_history)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 1  # Avoid division by zero

        logger.info(
            f"Trade history analysis: {total_trades} trades, "
            f"win_rate={win_rate:.1%}, avg_win=${avg_win:.2f}, avg_loss=${avg_loss:.2f}"
        )

        return self.calculate_position_size(
            account_size=account_size,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_price=current_price
        )

    def optimize_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        simulations: int = 10000
    ) -> Dict[str, float]:
        """
        Run Monte Carlo simulation to find optimal Kelly fraction

        Simulates trading with different Kelly fractions to find
        the balance between growth and drawdown risk.

        Args:
            win_rate: Win rate (0.0 to 1.0)
            avg_win: Average win amount
            avg_loss: Average loss amount
            simulations: Number of Monte Carlo simulations

        Returns:
            Dict with optimal fractional Kelly and statistics
        """
        full_kelly = self.calculate_kelly_percentage(win_rate, avg_win, avg_loss)

        if full_kelly <= 0:
            return {
                "optimal_fraction": 0.0,
                "full_kelly": full_kelly,
                "recommendation": "Do not trade - negative expectancy"
            }

        # Test different fractions
        fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        results = {}

        for fraction in fractions:
            position_size = full_kelly * fraction

            # Simulate trades
            balances = []
            for _ in range(simulations):
                balance = 100.0  # Start with $100
                for _ in range(100):  # 100 trades per simulation
                    # Random outcome based on win rate
                    if np.random.random() < win_rate:
                        balance += balance * position_size * (avg_win / 100)
                    else:
                        balance -= balance * position_size * (avg_loss / 100)

                    if balance <= 0:
                        balance = 0
                        break

                balances.append(balance)

            # Calculate metrics
            median_balance = np.median(balances)
            avg_balance = np.mean(balances)
            max_drawdown = np.percentile(balances, 5)  # 5th percentile as worst case

            results[fraction] = {
                "median_balance": median_balance,
                "avg_balance": avg_balance,
                "max_drawdown": max_drawdown,
                "sharpe_proxy": avg_balance / (np.std(balances) + 1e-10)
            }

        # Find optimal (highest Sharpe proxy)
        optimal_fraction = max(results.keys(), key=lambda k: results[k]["sharpe_proxy"])

        return {
            "optimal_fraction": optimal_fraction,
            "full_kelly": full_kelly,
            "results": results,
            "recommendation": f"Use {optimal_fraction:.0%} of Kelly for optimal risk-adjusted returns"
        }


def main():
    """Example usage"""
    # Initialize calculator
    kelly = KellyCriterion(
        max_position_size=0.20,  # 20% max position
        fractional_kelly=0.25,    # Use 25% of full Kelly
        min_position_size=0.01    # 1% minimum
    )

    # Example 1: Direct calculation
    print("=" * 60)
    print("Example 1: Direct Kelly Calculation")
    print("=" * 60)

    result = kelly.calculate_position_size(
        account_size=100000,
        win_rate=0.55,
        avg_win=200,
        avg_loss=100,
        current_price=50.0,
        stop_loss_pct=0.02
    )

    print(f"Account Size: $100,000")
    print(f"Win Rate: 55%")
    print(f"Avg Win: $200, Avg Loss: $100")
    print(f"\nResults:")
    print(f"  Full Kelly: {result.kelly_fraction:.2%}")
    print(f"  Fractional Kelly (25%): {result.fractional_kelly:.2%}")
    print(f"  Recommended Position: {result.recommended_size:.2%} of portfolio")
    print(f"  Position Value: ${100000 * result.recommended_size:,.2f}")
    print(f"  Units to Buy: {result.recommended_units:.2f}")
    print(f"  Risk Amount: ${result.risk_amount:.2f}")
    print(f"  Notes: {result.notes}")

    # Example 2: From trade history
    print("\n" + "=" * 60)
    print("Example 2: Calculate from Trade History")
    print("=" * 60)

    trade_history = [
        150, -80, 200, -90, 180, -100, 220, -85, 190, -95,
        160, -110, 210, -75, 170, -105, 230, -88, 195, -92
    ]

    result2 = kelly.calculate_from_history(
        trade_history=trade_history,
        account_size=100000,
        current_price=50.0
    )

    print(f"Trade History: {len(trade_history)} trades")
    print(f"\nResults:")
    print(f"  Recommended Position: {result2.recommended_size:.2%}")
    print(f"  Units to Buy: {result2.recommended_units:.2f}")
    print(f"  Notes: {result2.notes}")

    # Example 3: Optimize Kelly fraction
    print("\n" + "=" * 60)
    print("Example 3: Optimize Kelly Fraction")
    print("=" * 60)

    optimization = kelly.optimize_kelly_fraction(
        win_rate=0.55,
        avg_win=200,
        avg_loss=100,
        simulations=1000
    )

    print(f"Full Kelly: {optimization['full_kelly']:.2%}")
    print(f"Optimal Fraction: {optimization['optimal_fraction']:.0%}")
    print(f"Recommendation: {optimization['recommendation']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
