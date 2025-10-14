from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import numpy as np
import pandas as pd

"""
Base class for hypothesis testing framework.

This module provides the core HypothesisTester class that all hypothesis
research agents inherit from. It enforces a consistent testing methodology
across all market inefficiency hypotheses.
"""



@dataclass
class HypothesisMetadata:
    """Metadata for a trading hypothesis."""
    hypothesis_id: str
    title: str
    category: str  # e.g., "on-chain", "microstructure", "sentiment"
    priority_score: int
    created: datetime = field(default_factory=datetime.now)
    status: str = "research"  # research, testing, killed, iterate, scale


@dataclass
class BacktestResults:
    """Results from backtesting a hypothesis."""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    total_return: float
    annual_return: float
    volatility: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    equity_curve: pd.Series
    trades: pd.DataFrame


@dataclass
class StatisticalValidation:
    """Statistical validation results."""
    p_value: float
    t_statistic: float
    correlation: float
    information_coefficient: float
    significant: bool
    notes: str = ""


@dataclass
class HypothesisDecision:
    """Final decision on hypothesis."""
    decision: str  # KILL, ITERATE, SCALE
    confidence: float  # 0-1
    reasoning: List[str]
    next_steps: List[str]
    ready_for_production: bool


@dataclass
class HypothesisReport:
    """Complete report for a hypothesis test."""
    metadata: HypothesisMetadata
    backtest_results: BacktestResults
    statistical_validation: StatisticalValidation
    decision: HypothesisDecision
    execution_time: float
    data_quality_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class HypothesisTester(ABC):
    """
    Base class for all hypothesis testing agents.

    Each hypothesis research agent inherits from this class and implements
    the abstract methods. The class enforces a consistent testing pipeline:

    1. Document hypothesis
    2. Collect historical data
    3. Engineer features
    4. Validate statistically
    5. Backtest strategy
    6. Make KILL/ITERATE/SCALE decision
    """

    def __init__(
        self,
        hypothesis_id: str,
        title: str,
        category: str,
        priority_score: int,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None
    ):
        """
        Initialize hypothesis tester.

        Args:
            hypothesis_id: Unique identifier (e.g., "H001")
            title: Human-readable title
            category: Category (on-chain, microstructure, sentiment, etc.)
            priority_score: Priority score (higher = more important)
            data_dir: Directory for storing collected data
            results_dir: Directory for storing results
        """
        self.metadata = HypothesisMetadata(
            hypothesis_id=hypothesis_id,
            title=title,
            category=category,
            priority_score=priority_score
        )

        # Set up directories
        base_dir = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/research")
        self.data_dir = data_dir or base_dir / "data" / hypothesis_id
        self.results_dir = results_dir or base_dir / "results" / hypothesis_id
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Storage for collected data and results
        self.raw_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.backtest_results: Optional[BacktestResults] = None
        self.statistical_validation: Optional[StatisticalValidation] = None
        self.decision: Optional[HypothesisDecision] = None

    @abstractmethod
    async def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Collect historical data for the hypothesis.

        Args:
            start_date: Start of data collection period
            end_date: End of data collection period

        Returns:
            DataFrame with raw historical data
        """
        pass

    @abstractmethod
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw data.

        Args:
            data: Raw historical data

        Returns:
            DataFrame with engineered features
        """
        pass

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from features.

        Args:
            features: Engineered features

        Returns:
            Series of trading signals (1=long, -1=short, 0=neutral)
        """
        pass

    def validate_hypothesis_statistically(
        self,
        features: pd.DataFrame,
        returns: pd.Series
    ) -> StatisticalValidation:
        """
        Validate hypothesis using statistical tests.

        Args:
            features: Engineered features
            returns: Forward returns

        Returns:
            StatisticalValidation object
        """
        from scipy import stats

        # Get primary feature (should be implemented by subclass)
        if hasattr(self, 'primary_feature'):
            primary_feature = features[self.primary_feature]
        else:
            # Use first feature as fallback
            primary_feature = features.iloc[:, 0]

        # Align indices (important after dropna operations)
        aligned_df = pd.DataFrame({
            'feature': primary_feature,
            'returns': returns
        }).dropna()

        # Calculate correlation
        correlation = aligned_df['feature'].corr(aligned_df['returns'])

        # T-test for significance
        median_val = aligned_df['feature'].median()
        t_stat, p_value = stats.ttest_ind(
            aligned_df[aligned_df['feature'] > median_val]['returns'],
            aligned_df[aligned_df['feature'] <= median_val]['returns'],
            equal_var=False
        )

        # Information coefficient (Spearman rank correlation)
        ic, _ = stats.spearmanr(aligned_df['feature'], aligned_df['returns'])

        # Determine if significant
        significant = (p_value < 0.05) and (abs(correlation) > 0.1)

        return StatisticalValidation(
            p_value=float(p_value),
            t_statistic=float(t_stat),
            correlation=float(correlation),
            information_coefficient=float(ic),
            significant=significant,
            notes=f"Primary feature correlation: {correlation:.3f}"
        )

    def backtest_strategy(
        self,
        features: pd.DataFrame,
        signals: pd.Series,
        prices: pd.Series,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005     # 0.05% slippage
    ) -> BacktestResults:
        """
        Backtest the trading strategy.

        Args:
            features: Engineered features
            signals: Trading signals
            prices: Price data
            commission: Commission per trade (as fraction)
            slippage: Slippage per trade (as fraction)

        Returns:
            BacktestResults object
        """
        # Calculate returns
        returns = prices.pct_change()

        # Calculate strategy returns
        position = signals.shift(1).fillna(0)  # Trade on next bar
        strategy_returns = position * returns

        # Apply costs
        trades = position.diff().abs()
        costs = trades * (commission + slippage)
        net_returns = strategy_returns - costs

        # Calculate equity curve
        equity = (1 + net_returns).cumprod()

        # Calculate metrics
        sharpe = self._calculate_sharpe(net_returns)
        sortino = self._calculate_sortino(net_returns)
        max_dd = self._calculate_max_drawdown(equity)

        # Trade statistics
        winning_trades = net_returns[net_returns > 0]
        losing_trades = net_returns[net_returns < 0]

        win_rate = len(winning_trades) / len(net_returns[net_returns != 0]) if len(net_returns[net_returns != 0]) > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else 0

        # Create trades DataFrame (align indices first)
        trade_mask = net_returns != 0
        trades_data = pd.DataFrame({
            'net_return': net_returns,
            'position': position,
            'price': prices
        }).loc[trade_mask]

        trades_df = pd.DataFrame({
            'timestamp': trades_data.index,
            'return': trades_data['net_return'].values,
            'position': trades_data['position'].values,
            'price': trades_data['price'].values
        })

        return BacktestResults(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=int(trades.sum()),
            total_return=float(equity.iloc[-1] - 1),
            annual_return=float(net_returns.mean() * 252),  # Assuming daily data
            volatility=float(net_returns.std() * np.sqrt(252)),
            profit_factor=float(profit_factor),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            best_trade=float(net_returns.max()),
            worst_trade=float(net_returns.min()),
            equity_curve=equity,
            trades=trades_df
        )

    def make_decision(
        self,
        backtest_results: BacktestResults,
        statistical_validation: StatisticalValidation,
        min_sharpe: float = 1.5,
        min_win_rate: float = 0.6,
        max_p_value: float = 0.05
    ) -> HypothesisDecision:
        """
        Make KILL/ITERATE/SCALE decision based on results.

        Args:
            backtest_results: Backtest results
            statistical_validation: Statistical validation results
            min_sharpe: Minimum Sharpe ratio for SCALE
            min_win_rate: Minimum win rate for SCALE
            max_p_value: Maximum p-value for significance

        Returns:
            HypothesisDecision object
        """
        reasoning = []
        next_steps = []

        # Check SCALE criteria
        scale_criteria = [
            backtest_results.sharpe_ratio >= min_sharpe,
            backtest_results.win_rate >= min_win_rate,
            statistical_validation.p_value <= max_p_value,
            statistical_validation.significant,
            backtest_results.total_trades >= 30  # Minimum sample size
        ]

        # Check KILL criteria
        kill_criteria = [
            backtest_results.sharpe_ratio < 0.5,
            backtest_results.win_rate < 0.5,
            statistical_validation.p_value > 0.1,
            not statistical_validation.significant
        ]

        # Make decision
        if all(scale_criteria):
            decision = "SCALE"
            confidence = 0.9
            reasoning.append(f"Sharpe ratio {backtest_results.sharpe_ratio:.2f} exceeds {min_sharpe}")
            reasoning.append(f"Win rate {backtest_results.win_rate:.2%} exceeds {min_win_rate:.0%}")
            reasoning.append(f"Statistically significant (p={statistical_validation.p_value:.4f})")
            next_steps.append("Implement in production strategy")
            next_steps.append("Start paper trading")
            next_steps.append("Set up monitoring dashboard")
            ready = True

        elif sum(kill_criteria) >= 2:
            decision = "KILL"
            confidence = 0.8
            reasoning.append(f"Sharpe ratio {backtest_results.sharpe_ratio:.2f} too low")
            if backtest_results.win_rate < 0.5:
                reasoning.append(f"Win rate {backtest_results.win_rate:.2%} below breakeven")
            if statistical_validation.p_value > 0.1:
                reasoning.append(f"Not statistically significant (p={statistical_validation.p_value:.4f})")
            next_steps.append("Archive hypothesis")
            next_steps.append("Document learnings")
            ready = False

        else:
            decision = "ITERATE"
            confidence = 0.6
            reasoning.append(f"Shows promise (Sharpe={backtest_results.sharpe_ratio:.2f})")
            reasoning.append("Needs refinement before production")
            next_steps.append("Improve feature engineering")
            next_steps.append("Collect more/better data")
            next_steps.append("Refine entry/exit rules")
            next_steps.append("Re-test in 1-2 weeks")
            ready = False

        return HypothesisDecision(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            next_steps=next_steps,
            ready_for_production=ready
        )

    async def execute_full_pipeline(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback_months: int = 6
    ) -> HypothesisReport:
        """
        Execute the full hypothesis testing pipeline.

        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            lookback_months: Number of months to look back if dates not provided

        Returns:
            Complete HypothesisReport
        """
        start_time = datetime.now()

        # Set default dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=lookback_months * 30)

        print(f"[{self.metadata.hypothesis_id}] Starting pipeline: {self.metadata.title}")

        # Step 1: Collect data
        print(f"[{self.metadata.hypothesis_id}] Collecting data from {start_date.date()} to {end_date.date()}...")
        self.raw_data = await self.collect_historical_data(start_date, end_date)
        data_quality = self._assess_data_quality(self.raw_data)

        # Step 2: Engineer features
        print(f"[{self.metadata.hypothesis_id}] Engineering features...")
        self.features = self.engineer_features(self.raw_data)

        # Step 3: Generate signals
        print(f"[{self.metadata.hypothesis_id}] Generating trading signals...")
        signals = self.generate_signals(self.features)

        # Step 4: Statistical validation
        print(f"[{self.metadata.hypothesis_id}] Running statistical validation...")
        returns = self.raw_data['close'].pct_change().shift(-1)  # Forward returns
        self.statistical_validation = self.validate_hypothesis_statistically(
            self.features,
            returns
        )

        # Step 5: Backtest
        print(f"[{self.metadata.hypothesis_id}] Backtesting strategy...")
        self.backtest_results = self.backtest_strategy(
            self.features,
            signals,
            self.raw_data['close']
        )

        # Step 6: Make decision
        print(f"[{self.metadata.hypothesis_id}] Making decision...")
        self.decision = self.make_decision(
            self.backtest_results,
            self.statistical_validation
        )

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Create report
        report = HypothesisReport(
            metadata=self.metadata,
            backtest_results=self.backtest_results,
            statistical_validation=self.statistical_validation,
            decision=self.decision,
            execution_time=execution_time,
            data_quality_score=data_quality
        )

        # Save report
        self._save_report(report)

        print(f"[{self.metadata.hypothesis_id}] Pipeline complete!")
        print(f"[{self.metadata.hypothesis_id}] Decision: {self.decision.decision} (confidence: {self.decision.confidence:.0%})")
        print(f"[{self.metadata.hypothesis_id}] Sharpe: {self.backtest_results.sharpe_ratio:.2f} | Win Rate: {self.backtest_results.win_rate:.1%}")

        return report

    # Helper methods

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return float(excess_returns.mean() / excess_returns.std() * np.sqrt(252))

    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return float(excess_returns.mean() / downside_returns.std() * np.sqrt(252))

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return float(drawdown.min())

    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """
        Assess data quality (0-1 score).

        Checks:
        - Missing values
        - Duplicate timestamps
        - Data coverage
        """
        total_score = 1.0

        # Penalize missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        total_score -= missing_pct * 0.5

        # Penalize duplicates
        if 'timestamp' in data.columns:
            duplicate_pct = data['timestamp'].duplicated().sum() / len(data)
            total_score -= duplicate_pct * 0.3

        # Penalize low coverage
        if len(data) < 100:
            total_score -= 0.2

        return max(0.0, min(1.0, total_score))

    def _save_report(self, report: HypothesisReport) -> None:
        """Save report to disk."""
        report_file = self.results_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert to dict (simplified - would need proper serialization)
        report_dict = {
            "hypothesis_id": report.metadata.hypothesis_id,
            "title": report.metadata.title,
            "decision": report.decision.decision,
            "confidence": report.decision.confidence,
            "sharpe_ratio": report.backtest_results.sharpe_ratio,
            "win_rate": report.backtest_results.win_rate,
            "p_value": report.statistical_validation.p_value,
            "execution_time": report.execution_time,
            "timestamp": report.timestamp.isoformat()
        }

        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)

        print(f"[{self.metadata.hypothesis_id}] Report saved to {report_file}")
