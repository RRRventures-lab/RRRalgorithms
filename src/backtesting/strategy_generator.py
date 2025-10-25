from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import itertools
import logging
import numpy as np
import pandas as pd

"""
Strategy Generation System
===========================

Generates trading strategies by combining inefficiency detectors
with various parameters and filters.
"""

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    strategy_id: str
    name: str
    description: str

    # Detector configuration
    detector_type: str  # 'latency', 'funding', 'correlation', 'sentiment', 'seasonality', 'orderflow'
    detector_params: Dict[str, Any] = field(default_factory=dict)

    # Signal filters
    min_confidence: float = 0.6
    min_expected_return: float = 0.005  # 0.5%
    max_p_value: float = 0.05

    # Entry/exit rules
    entry_rule: str = "signal"  # 'signal', 'confirmation', 'breakout'
    exit_rule: str = "target"   # 'target', 'stop', 'time', 'reverse'
    hold_time: Optional[int] = None  # Bars to hold position

    # Risk management
    stop_loss_pct: Optional[float] = 0.02  # 2%
    take_profit_pct: Optional[float] = 0.05  # 5%
    position_size: float = 0.1  # 10% of capital

    # Filters
    market_regime_filter: Optional[str] = None  # 'trending', 'ranging', None
    volatility_filter: Optional[str] = None      # 'high', 'low', None
    time_filter: Optional[List[int]] = None      # Trading hours

    # Ensemble
    combine_with: Optional[List[str]] = None  # Other detector types to combine

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'description': self.description,
            'detector_type': self.detector_type,
            'detector_params': self.detector_params,
            'min_confidence': self.min_confidence,
            'min_expected_return': self.min_expected_return,
            'max_p_value': self.max_p_value,
            'entry_rule': self.entry_rule,
            'exit_rule': self.exit_rule,
            'hold_time': self.hold_time,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'position_size': self.position_size,
            'market_regime_filter': self.market_regime_filter,
            'volatility_filter': self.volatility_filter,
            'time_filter': self.time_filter,
            'combine_with': self.combine_with
        }


class StrategyGenerator:
    """
    Generates trading strategy variations

    Creates strategies by:
    1. Combining different inefficiency detectors
    2. Varying parameters
    3. Adding filters and risk management
    4. Creating ensemble strategies
    """

    def __init__(self):
        self.strategies: List[StrategyConfig] = []

        # Detector types available
        self.detector_types = [
            'latency_arbitrage',
            'funding_rate',
            'correlation_anomaly',
            'sentiment_divergence',
            'seasonality',
            'order_flow_toxicity'
        ]

        # Parameter ranges for grid search
        self.param_ranges = {
            'min_confidence': [0.5, 0.6, 0.7, 0.8],
            'min_expected_return': [0.003, 0.005, 0.01, 0.02],
            'max_p_value': [0.01, 0.05, 0.10],
            'stop_loss_pct': [0.01, 0.02, 0.03, 0.05],
            'take_profit_pct': [0.03, 0.05, 0.08, 0.10],
            'position_size': [0.05, 0.10, 0.15, 0.20],
            'hold_time': [5, 10, 20, 50, None]
        }

    def generate_strategies(self, n_strategies: int = 500) -> List[StrategyConfig]:
        """
        Generate N strategy variations

        Strategies include:
        - Single detector strategies with parameter variations
        - Ensemble strategies combining multiple detectors
        - Filtered strategies with regime/volatility filters
        - Time-based strategies

        Args:
            n_strategies: Number of strategies to generate

        Returns:
            List of StrategyConfig objects
        """
        logger.info(f"Generating {n_strategies} strategy variations...")

        strategies = []

        # 1. Single detector strategies with parameter grid search
        single_detector_strategies = self._generate_single_detector_strategies()
        strategies.extend(single_detector_strategies[:n_strategies // 2])

        # 2. Ensemble strategies
        ensemble_strategies = self._generate_ensemble_strategies()
        strategies.extend(ensemble_strategies[:n_strategies // 4])

        # 3. Filtered strategies
        filtered_strategies = self._generate_filtered_strategies()
        strategies.extend(filtered_strategies[:n_strategies // 4])

        # Trim to exact number requested
        strategies = strategies[:n_strategies]

        logger.info(f"Generated {len(strategies)} strategies:")
        logger.info(f"  Single detector: {len(single_detector_strategies)}")
        logger.info(f"  Ensemble: {len(ensemble_strategies)}")
        logger.info(f"  Filtered: {len(filtered_strategies)}")

        self.strategies = strategies
        return strategies

    def _generate_single_detector_strategies(self) -> List[StrategyConfig]:
        """Generate strategies using single detectors with parameter variations"""
        strategies = []
        strategy_counter = 0

        for detector_type in self.detector_types:
            # Generate parameter combinations (grid search)
            param_keys = ['min_confidence', 'min_expected_return', 'max_p_value',
                         'stop_loss_pct', 'take_profit_pct', 'position_size']

            # Sample parameter combinations (not full grid to avoid explosion)
            n_samples = 20  # Sample 20 parameter combinations per detector

            for i in range(n_samples):
                # Random sample from parameter ranges
                params = {
                    'min_confidence': np.random.choice(self.param_ranges['min_confidence']),
                    'min_expected_return': np.random.choice(self.param_ranges['min_expected_return']),
                    'max_p_value': np.random.choice(self.param_ranges['max_p_value']),
                    'stop_loss_pct': np.random.choice(self.param_ranges['stop_loss_pct']),
                    'take_profit_pct': np.random.choice(self.param_ranges['take_profit_pct']),
                    'position_size': np.random.choice(self.param_ranges['position_size']),
                    'hold_time': np.random.choice(self.param_ranges['hold_time'])
                }

                strategy = StrategyConfig(
                    strategy_id=f"single_{detector_type}_{strategy_counter}",
                    name=f"{detector_type.replace('_', ' ').title()} Strategy {strategy_counter}",
                    description=f"Single detector strategy using {detector_type}",
                    detector_type=detector_type,
                    **params
                )

                strategies.append(strategy)
                strategy_counter += 1

        logger.info(f"Generated {len(strategies)} single detector strategies")
        return strategies

    def _generate_ensemble_strategies(self) -> List[StrategyConfig]:
        """Generate ensemble strategies combining multiple detectors"""
        strategies = []
        strategy_counter = 0

        # Generate pairs and triples of detectors
        detector_pairs = list(itertools.combinations(self.detector_types, 2))
        detector_triples = list(itertools.combinations(self.detector_types, 3))

        # Sample some combinations
        sampled_pairs = detector_pairs[:15]  # Take 15 pairs
        sampled_triples = detector_triples[:10]  # Take 10 triples

        for combo in sampled_pairs + sampled_triples:
            primary_detector = combo[0]
            secondary_detectors = list(combo[1:])

            strategy = StrategyConfig(
                strategy_id=f"ensemble_{strategy_counter}",
                name=f"Ensemble {' + '.join([d.replace('_', ' ').title() for d in combo])}",
                description=f"Ensemble combining {', '.join(combo)}",
                detector_type=primary_detector,
                combine_with=secondary_detectors,
                min_confidence=0.7,  # Higher threshold for ensembles
                min_expected_return=0.008,
                max_p_value=0.01,
                stop_loss_pct=0.02,
                take_profit_pct=0.05,
                position_size=0.10
            )

            strategies.append(strategy)
            strategy_counter += 1

        logger.info(f"Generated {len(strategies)} ensemble strategies")
        return strategies

    def _generate_filtered_strategies(self) -> List[StrategyConfig]:
        """Generate strategies with market regime and volatility filters"""
        strategies = []
        strategy_counter = 0

        # Market regime filters
        regime_filters = ['trending', 'ranging', None]
        volatility_filters = ['high', 'low', None]

        for detector_type in self.detector_types:
            for regime in regime_filters:
                for volatility in volatility_filters:
                    if regime is None and volatility is None:
                        continue  # Skip unfiltered (already in single detector)

                    filter_desc = []
                    if regime:
                        filter_desc.append(f"{regime} market")
                    if volatility:
                        filter_desc.append(f"{volatility} volatility")

                    strategy = StrategyConfig(
                        strategy_id=f"filtered_{detector_type}_{strategy_counter}",
                        name=f"{detector_type.replace('_', ' ').title()} - {' + '.join(filter_desc)}",
                        description=f"{detector_type} filtered for {', '.join(filter_desc)}",
                        detector_type=detector_type,
                        market_regime_filter=regime,
                        volatility_filter=volatility,
                        min_confidence=0.65,
                        min_expected_return=0.005,
                        max_p_value=0.05,
                        stop_loss_pct=0.02,
                        take_profit_pct=0.05,
                        position_size=0.10
                    )

                    strategies.append(strategy)
                    strategy_counter += 1

        logger.info(f"Generated {len(strategies)} filtered strategies")
        return strategies

    def create_signal_function(self, strategy: StrategyConfig,
                              detector_signals: Dict[str, pd.DataFrame]) -> Callable:
        """
        Create a signal generation function for a strategy

        Args:
            strategy: Strategy configuration
            detector_signals: Dictionary of detector signals by type

        Returns:
            Function that generates trading signals from data
        """
        def signal_func(data: pd.DataFrame) -> pd.Series:
            """
            Generate trading signals

            Returns:
                Series with values: 1 (long), -1 (short), 0 (neutral)
            """
            signals = pd.Series(0, index=data.index)

            # Get primary detector signals
            if strategy.detector_type not in detector_signals:
                logger.warning(f"No signals found for detector: {strategy.detector_type}")
                return signals

            detector_df = detector_signals[strategy.detector_type]

            # Apply filters
            mask = (
                (detector_df['confidence'] >= strategy.min_confidence) &
                (detector_df['expected_return'] >= strategy.min_expected_return) &
                (detector_df['p_value'] <= strategy.max_p_value)
            )

            # Market regime filter
            if strategy.market_regime_filter and 'market_regime' in detector_df.columns:
                mask &= detector_df['market_regime'].str.contains(strategy.market_regime_filter, na=False)

            # Volatility filter
            if strategy.volatility_filter and 'volatility_regime' in detector_df.columns:
                mask &= detector_df['volatility_regime'].str.contains(strategy.volatility_filter, na=False)

            # Generate signals
            filtered_signals = detector_df[mask]

            for idx in filtered_signals.index:
                if idx in signals.index:
                    direction = filtered_signals.loc[idx, 'direction']
                    if direction == 'long':
                        signals.loc[idx] = 1
                    elif direction == 'short':
                        signals.loc[idx] = -1

            # Apply ensemble logic if specified
            if strategy.combine_with:
                for secondary_detector in strategy.combine_with:
                    if secondary_detector in detector_signals:
                        secondary_df = detector_signals[secondary_detector]
                        secondary_mask = (
                            (secondary_df['confidence'] >= strategy.min_confidence) &
                            (secondary_df['expected_return'] >= strategy.min_expected_return)
                        )
                        # Require confirmation from secondary detector
                        for idx in signals[signals != 0].index:
                            if idx not in secondary_df.index or not secondary_mask.loc[idx]:
                                signals.loc[idx] = 0  # Cancel signal if no confirmation

            # Apply hold time logic
            if strategy.hold_time:
                signals = self._apply_hold_time(signals, strategy.hold_time)

            return signals

        return signal_func

    def _apply_hold_time(self, signals: pd.Series, hold_time: int) -> pd.Series:
        """Apply minimum hold time to signals"""
        adjusted_signals = signals.copy()

        in_position = False
        entry_idx = None
        position_direction = 0

        for i, idx in enumerate(signals.index):
            if not in_position and signals.iloc[i] != 0:
                # Enter position
                in_position = True
                entry_idx = i
                position_direction = signals.iloc[i]

            elif in_position:
                # Check if hold time elapsed
                bars_held = i - entry_idx

                if bars_held < hold_time:
                    # Keep position open
                    adjusted_signals.iloc[i] = position_direction
                else:
                    # Can exit
                    if signals.iloc[i] == 0 or signals.iloc[i] != position_direction:
                        in_position = False
                        adjusted_signals.iloc[i] = 0

        return adjusted_signals

    def get_strategy_by_id(self, strategy_id: str) -> Optional[StrategyConfig]:
        """Get strategy by ID"""
        for strategy in self.strategies:
            if strategy.strategy_id == strategy_id:
                return strategy
        return None

    def get_strategies_by_detector(self, detector_type: str) -> List[StrategyConfig]:
        """Get all strategies using a specific detector"""
        return [s for s in self.strategies if s.detector_type == detector_type]

    def export_strategies(self) -> List[Dict[str, Any]]:
        """Export all strategies as list of dicts"""
        return [s.to_dict() for s in self.strategies]
