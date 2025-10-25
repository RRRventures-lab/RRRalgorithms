from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
import numpy as np
import pandas as pd

"""
Results Aggregation and Ranking System
=======================================

Aggregates backtest results across strategies and ranks them
for deployment selection.
"""

logger = logging.getLogger(__name__)


@dataclass
class StrategyRanking:
    """Ranked strategy with all metrics"""
    rank: int
    strategy_id: str
    strategy_name: str
    detector_type: str

    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_return: float
    annualized_return: float

    # Risk metrics
    max_drawdown: float
    volatility: float
    var_95: float

    # Trade statistics
    win_rate: float
    profit_factor: float
    total_trades: int

    # Validation
    p_value: float
    validation_grade: str
    confidence_level: float

    # Monte Carlo
    mc_prob_profit: float
    mc_risk_of_ruin: float
    mc_mean_return: float

    # Overall score
    composite_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ResultsAggregator:
    """
    Aggregates and ranks backtest results

    Ranking methodology:
    - Composite score combining multiple metrics
    - Weighted by statistical significance
    - Penalized by risk metrics
    - Boosted by consistency
    """

    def __init__(self,
                 output_dir: Path,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize results aggregator

        Args:
            output_dir: Directory for saving results
            weights: Custom weights for scoring (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default scoring weights
        self.weights = weights or {
            'sharpe_ratio': 0.25,
            'win_rate': 0.15,
            'profit_factor': 0.15,
            'mc_prob_profit': 0.15,
            'statistical_significance': 0.15,
            'max_drawdown': -0.10,  # Negative weight (penalty)
            'mc_risk_of_ruin': -0.05   # Negative weight (penalty)
        }

        self.rankings: List[StrategyRanking] = []

    def aggregate_results(self,
                         backtest_results: List[Dict[str, Any]],
                         validation_results: List[Any],  # ValidationResult
                         mc_results: List[Any]) -> List[StrategyRanking]:  # MonteCarloResult
        """
        Aggregate all results and create rankings

        Args:
            backtest_results: List of backtest metrics
            validation_results: List of ValidationResult
            mc_results: List of MonteCarloResult

        Returns:
            List of StrategyRanking sorted by composite score
        """
        logger.info(f"Aggregating results from {len(backtest_results)} strategies")

        # Create lookup dictionaries
        validation_lookup = {v.strategy_id: v for v in validation_results}
        mc_lookup = {m.strategy_id: m for m in mc_results}

        rankings = []

        for bt_result in backtest_results:
            strategy_id = bt_result['strategy_id']

            # Get corresponding validation and MC results
            validation = validation_lookup.get(strategy_id)
            mc_result = mc_lookup.get(strategy_id)

            if not validation or not mc_result:
                logger.warning(f"Missing validation or MC results for {strategy_id}")
                continue

            # Skip if not validated
            if not validation.is_valid:
                continue

            # Calculate composite score
            composite_score = self._calculate_composite_score(
                bt_result['metrics'],
                validation,
                mc_result
            )

            # Create ranking entry
            ranking = StrategyRanking(
                rank=0,  # Will be assigned after sorting
                strategy_id=strategy_id,
                strategy_name=bt_result.get('strategy_name', strategy_id),
                detector_type=bt_result.get('detector_type', 'unknown'),
                sharpe_ratio=bt_result['metrics'].sharpe_ratio,
                sortino_ratio=bt_result['metrics'].sortino_ratio,
                calmar_ratio=bt_result['metrics'].calmar_ratio or 0,
                total_return=bt_result['metrics'].total_return,
                annualized_return=bt_result['metrics'].annualized_return,
                max_drawdown=bt_result['metrics'].max_drawdown,
                volatility=bt_result['metrics'].volatility,
                var_95=bt_result['metrics'].var_95,
                win_rate=bt_result['metrics'].win_rate,
                profit_factor=bt_result['metrics'].profit_factor,
                total_trades=bt_result['metrics'].total_trades,
                p_value=validation.p_value_adjusted,
                validation_grade=validation.get_grade(),
                confidence_level=validation.confidence_level,
                mc_prob_profit=mc_result.probability_of_profit,
                mc_risk_of_ruin=mc_result.risk_of_ruin,
                mc_mean_return=mc_result.mean_return,
                composite_score=composite_score
            )

            rankings.append(ranking)

        # Sort by composite score
        rankings.sort(key=lambda r: r.composite_score, reverse=True)

        # Assign ranks
        for i, ranking in enumerate(rankings, 1):
            ranking.rank = i

        self.rankings = rankings

        logger.info(f"Created rankings for {len(rankings)} validated strategies")

        return rankings

    def _calculate_composite_score(self,
                                  metrics: Any,  # BacktestMetrics
                                  validation: Any,  # ValidationResult
                                  mc_result: Any) -> float:  # MonteCarloResult
        """
        Calculate composite score for ranking

        Score components:
        - Performance metrics (Sharpe, win rate, profit factor)
        - Statistical significance
        - Monte Carlo confidence
        - Risk penalties
        """
        score = 0.0

        # Normalize Sharpe ratio (cap at 5.0)
        normalized_sharpe = min(metrics.sharpe_ratio / 5.0, 1.0)
        score += self.weights['sharpe_ratio'] * normalized_sharpe

        # Win rate
        score += self.weights['win_rate'] * metrics.win_rate

        # Profit factor (normalized to 1.0 at PF=3.0)
        normalized_pf = min(metrics.profit_factor / 3.0, 1.0)
        score += self.weights['profit_factor'] * normalized_pf

        # Monte Carlo probability of profit
        score += self.weights['mc_prob_profit'] * mc_result.probability_of_profit

        # Statistical significance (1 - p_value)
        sig_score = 1.0 - min(validation.p_value_adjusted, 1.0)
        score += self.weights['statistical_significance'] * sig_score

        # Penalties

        # Max drawdown penalty (worse drawdown = more penalty)
        dd_penalty = abs(metrics.max_drawdown)
        score += self.weights['max_drawdown'] * dd_penalty

        # Risk of ruin penalty
        score += self.weights['mc_risk_of_ruin'] * mc_result.risk_of_ruin

        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))

        return score

    def get_top_strategies(self, n: int = 10) -> List[StrategyRanking]:
        """Get top N strategies"""
        return self.rankings[:n]

    def get_strategies_by_detector(self, detector_type: str) -> List[StrategyRanking]:
        """Get all strategies for a specific detector type"""
        return [r for r in self.rankings if r.detector_type == detector_type]

    def get_diversified_portfolio(self,
                                 n_strategies: int = 10,
                                 max_per_detector: int = 3) -> List[StrategyRanking]:
        """
        Get diversified portfolio of strategies

        Ensures diversity by limiting strategies per detector type

        Args:
            n_strategies: Total number of strategies
            max_per_detector: Maximum strategies per detector type

        Returns:
            List of StrategyRanking
        """
        portfolio = []
        detector_counts = {}

        for ranking in self.rankings:
            detector = ranking.detector_type

            # Check if we've reached limit for this detector
            if detector_counts.get(detector, 0) >= max_per_detector:
                continue

            # Add to portfolio
            portfolio.append(ranking)
            detector_counts[detector] = detector_counts.get(detector, 0) + 1

            # Stop when we have enough
            if len(portfolio) >= n_strategies:
                break

        logger.info(f"Created diversified portfolio with {len(portfolio)} strategies:")
        for detector, count in detector_counts.items():
            logger.info(f"  {detector}: {count} strategies")

        return portfolio

    def save_results(self, filename: str = "strategy_rankings.json"):
        """Save rankings to JSON file"""
        filepath = self.output_dir / filename

        data = {
            'timestamp': datetime.now().isoformat(),
            'total_strategies': len(self.rankings),
            'weights': self.weights,
            'rankings': [r.to_dict() for r in self.rankings]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved rankings to {filepath}")

        return filepath

    def save_csv(self, filename: str = "strategy_rankings.csv"):
        """Save rankings to CSV file"""
        filepath = self.output_dir / filename

        df = pd.DataFrame([r.to_dict() for r in self.rankings])
        df.to_csv(filepath, index=False)

        logger.info(f"Saved rankings CSV to {filepath}")

        return filepath

    def generate_summary_report(self) -> str:
        """Generate human-readable summary report"""
        if not self.rankings:
            return "No strategies to report"

        report = []
        report.append("\n" + "="*80)
        report.append("STRATEGY RANKING SUMMARY")
        report.append("="*80)
        report.append(f"Total Validated Strategies: {len(self.rankings)}")
        report.append("")

        # Top 10 strategies
        report.append("TOP 10 STRATEGIES")
        report.append("-"*80)

        for i, ranking in enumerate(self.rankings[:10], 1):
            report.append(f"\n{i}. {ranking.strategy_name}")
            report.append(f"   Strategy ID: {ranking.strategy_id}")
            report.append(f"   Detector: {ranking.detector_type}")
            report.append(f"   Composite Score: {ranking.composite_score:.3f}")
            report.append(f"   Grade: {ranking.validation_grade}")
            report.append("")
            report.append(f"   Performance:")
            report.append(f"     Sharpe Ratio: {ranking.sharpe_ratio:.2f}")
            report.append(f"     Annual Return: {ranking.annualized_return:.1%}")
            report.append(f"     Win Rate: {ranking.win_rate:.1%}")
            report.append(f"     Profit Factor: {ranking.profit_factor:.2f}")
            report.append(f"     Max Drawdown: {ranking.max_drawdown:.1%}")
            report.append("")
            report.append(f"   Validation:")
            report.append(f"     P-value: {ranking.p_value:.4f}")
            report.append(f"     Confidence: {ranking.confidence_level:.1%}")
            report.append("")
            report.append(f"   Monte Carlo:")
            report.append(f"     Prob. of Profit: {ranking.mc_prob_profit:.1%}")
            report.append(f"     Risk of Ruin: {ranking.mc_risk_of_ruin:.2%}")

        # Statistics by detector type
        report.append("\n" + "-"*80)
        report.append("STATISTICS BY DETECTOR TYPE")
        report.append("-"*80)

        detector_stats = {}
        for ranking in self.rankings:
            detector = ranking.detector_type
            if detector not in detector_stats:
                detector_stats[detector] = {
                    'count': 0,
                    'avg_sharpe': [],
                    'avg_win_rate': [],
                    'avg_score': []
                }

            detector_stats[detector]['count'] += 1
            detector_stats[detector]['avg_sharpe'].append(ranking.sharpe_ratio)
            detector_stats[detector]['avg_win_rate'].append(ranking.win_rate)
            detector_stats[detector]['avg_score'].append(ranking.composite_score)

        for detector, stats in sorted(detector_stats.items(), key=lambda x: np.mean(x[1]['avg_score']), reverse=True):
            report.append(f"\n{detector}:")
            report.append(f"  Strategies: {stats['count']}")
            report.append(f"  Avg Sharpe: {np.mean(stats['avg_sharpe']):.2f}")
            report.append(f"  Avg Win Rate: {np.mean(stats['avg_win_rate']):.1%}")
            report.append(f"  Avg Score: {np.mean(stats['avg_score']):.3f}")

        return "\n".join(report)

    def export_for_deployment(self,
                            top_n: int = 10,
                            filename: str = "deployment_strategies.json") -> Path:
        """
        Export top strategies in format ready for live deployment

        Args:
            top_n: Number of top strategies to export
            filename: Output filename

        Returns:
            Path to exported file
        """
        filepath = self.output_dir / filename

        top_strategies = self.get_top_strategies(top_n)

        deployment_data = {
            'timestamp': datetime.now().isoformat(),
            'n_strategies': len(top_strategies),
            'strategies': []
        }

        for ranking in top_strategies:
            strategy_data = {
                'strategy_id': ranking.strategy_id,
                'strategy_name': ranking.strategy_name,
                'detector_type': ranking.detector_type,
                'rank': ranking.rank,
                'allocation_weight': self._calculate_allocation_weight(ranking, top_strategies),

                'expected_metrics': {
                    'sharpe_ratio': ranking.sharpe_ratio,
                    'annual_return': ranking.annualized_return,
                    'win_rate': ranking.win_rate,
                    'max_drawdown': ranking.max_drawdown,
                    'volatility': ranking.volatility
                },

                'risk_limits': {
                    'max_position_size': 0.10,  # 10%
                    'stop_loss': abs(ranking.max_drawdown) * 0.5,  # Half of historical max DD
                    'daily_loss_limit': 0.02  # 2% daily loss limit
                },

                'confidence': {
                    'validation_grade': ranking.validation_grade,
                    'p_value': ranking.p_value,
                    'mc_prob_profit': ranking.mc_prob_profit,
                    'mc_risk_of_ruin': ranking.mc_risk_of_ruin
                }
            }

            deployment_data['strategies'].append(strategy_data)

        with open(filepath, 'w') as f:
            json.dump(deployment_data, f, indent=2, default=str)

        logger.info(f"Exported {len(top_strategies)} strategies for deployment to {filepath}")

        return filepath

    def _calculate_allocation_weight(self,
                                   ranking: StrategyRanking,
                                   all_strategies: List[StrategyRanking]) -> float:
        """
        Calculate allocation weight for a strategy in a portfolio

        Uses inverse variance weighting based on risk metrics
        """
        # Simple approach: weight by composite score
        total_score = sum(s.composite_score for s in all_strategies)

        if total_score > 0:
            weight = ranking.composite_score / total_score
        else:
            weight = 1.0 / len(all_strategies)

        return round(weight, 4)
