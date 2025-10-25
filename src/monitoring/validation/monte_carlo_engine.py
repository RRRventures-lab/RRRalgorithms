from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import logging
import numpy as np
import pandas as pd

#!/usr/bin/env python3

"""
Monte Carlo Simulation Engine

Comprehensive Monte Carlo simulation suite for testing trading system robustness.
Generates 20,000+ scenarios across multiple categories:
- Market regimes (10,000+ scenarios)
- Microstructure (5,000+ scenarios)
- Risk events (3,000+ scenarios)
- Adversarial scenarios (2,000+ scenarios)

Author: AI Psychology Team
Date: 2025-10-11
"""


logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    BUBBLE = "bubble"
    RECOVERY = "recovery"


class ScenarioCategory(Enum):
    """Scenario categories"""
    MARKET_REGIME = "market_regime"
    MICROSTRUCTURE = "microstructure"
    RISK_EVENT = "risk_event"
    ADVERSARIAL = "adversarial"


@dataclass
class SimulationScenario:
    """Single simulation scenario"""
    scenario_id: str
    category: ScenarioCategory
    name: str
    description: str
    parameters: Dict[str, Any]
    expected_behavior: str
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class SimulationResult:
    """Result of running a scenario"""
    scenario_id: str
    scenario_name: str
    passed: bool
    execution_time_ms: float
    metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]
    hallucinations_detected: int
    decisions_rejected: int
    validation_failures: int
    system_stable: bool
    data_corruption: bool
    details: Dict[str, Any]


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for comprehensive system testing

    Generates and runs 20,000+ scenarios to test system robustness
    """

    def __init__(
        self,
        num_market_scenarios: int = 10000,
        num_microstructure_scenarios: int = 5000,
        num_risk_scenarios: int = 3000,
        num_adversarial_scenarios: int = 2000,
        parallel_workers: int = 8,
        random_seed: Optional[int] = None
    ):
        self.num_market_scenarios = num_market_scenarios
        self.num_microstructure_scenarios = num_microstructure_scenarios
        self.num_risk_scenarios = num_risk_scenarios
        self.num_adversarial_scenarios = num_adversarial_scenarios
        self.parallel_workers = parallel_workers

        # Set random seed for reproducibility
        if random_seed:
            np.random.seed(random_seed)

        # Storage
        self.scenarios: List[SimulationScenario] = []
        self.results: List[SimulationResult] = []

        # Statistics
        self.total_scenarios_generated = 0
        self.total_scenarios_run = 0
        self.scenarios_passed = 0
        self.scenarios_failed = 0

        logger.info(f"MonteCarloEngine initialized: {self.total_scenarios()} total scenarios")

    def total_scenarios(self) -> int:
        """Total number of scenarios"""
        return (self.num_market_scenarios +
                self.num_microstructure_scenarios +
                self.num_risk_scenarios +
                self.num_adversarial_scenarios)

    def generate_all_scenarios(self) -> List[SimulationScenario]:
        """Generate all simulation scenarios"""
        logger.info("Generating all scenarios...")

        self.scenarios = []

        # 1. Market regime scenarios
        self.scenarios.extend(self._generate_market_regime_scenarios())

        # 2. Microstructure scenarios
        self.scenarios.extend(self._generate_microstructure_scenarios())

        # 3. Risk event scenarios
        self.scenarios.extend(self._generate_risk_event_scenarios())

        # 4. Adversarial scenarios
        self.scenarios.extend(self._generate_adversarial_scenarios())

        self.total_scenarios_generated = len(self.scenarios)
        logger.info(f"Generated {self.total_scenarios_generated} scenarios")

        return self.scenarios

    def _generate_market_regime_scenarios(self) -> List[SimulationScenario]:
        """
        Generate market regime scenarios (10,000+)

        Tests system behavior across different market conditions
        """
        scenarios = []

        # 1. Bull markets (2000 scenarios)
        for i in range(2000):
            trend_strength = np.random.uniform(0.01, 0.10)  # 1-10% daily return
            volatility = np.random.uniform(0.01, 0.05)
            duration_days = np.random.randint(30, 365)

            scenarios.append(SimulationScenario(
                scenario_id=f"market_bull_{i:04d}",
                category=ScenarioCategory.MARKET_REGIME,
                name=f"Bull Market {i+1}",
                description=f"Bull market with {trend_strength*100:.1f}% daily return, {volatility*100:.1f}% volatility",
                parameters={
                    "regime": MarketRegime.BULL.value,
                    "trend_strength": trend_strength,
                    "volatility": volatility,
                    "duration_days": duration_days,
                    "autocorrelation": np.random.uniform(0.3, 0.7)
                },
                expected_behavior="System should capitalize on trend with appropriate position sizing",
                severity="medium"
            ))

        # 2. Bear markets (2000 scenarios)
        for i in range(2000):
            trend_strength = -np.random.uniform(0.01, 0.08)  # -1% to -8% daily return
            volatility = np.random.uniform(0.02, 0.08)
            duration_days = np.random.randint(30, 180)

            scenarios.append(SimulationScenario(
                scenario_id=f"market_bear_{i:04d}",
                category=ScenarioCategory.MARKET_REGIME,
                name=f"Bear Market {i+1}",
                description=f"Bear market with {trend_strength*100:.1f}% daily return",
                parameters={
                    "regime": MarketRegime.BEAR.value,
                    "trend_strength": trend_strength,
                    "volatility": volatility,
                    "duration_days": duration_days,
                    "panic_selling": np.random.choice([True, False], p=[0.3, 0.7])
                },
                expected_behavior="System should reduce exposure and implement stop-losses",
                severity="high"
            ))

        # 3. Ranging markets (2000 scenarios)
        for i in range(2000):
            range_width = np.random.uniform(0.02, 0.15)
            oscillation_period = np.random.randint(5, 30)
            volatility = np.random.uniform(0.01, 0.04)

            scenarios.append(SimulationScenario(
                scenario_id=f"market_range_{i:04d}",
                category=ScenarioCategory.MARKET_REGIME,
                name=f"Ranging Market {i+1}",
                description=f"Ranging market with {range_width*100:.1f}% range",
                parameters={
                    "regime": MarketRegime.RANGING.value,
                    "range_width": range_width,
                    "oscillation_period": oscillation_period,
                    "volatility": volatility,
                    "breakout_probability": np.random.uniform(0.0, 0.2)
                },
                expected_behavior="System should employ mean-reversion strategies",
                severity="low"
            ))

        # 4. High volatility periods (1500 scenarios)
        for i in range(1500):
            volatility = np.random.uniform(0.05, 0.30)
            garch_params = {
                "omega": np.random.uniform(0.00001, 0.0001),
                "alpha": np.random.uniform(0.05, 0.15),
                "beta": np.random.uniform(0.75, 0.95)
            }

            scenarios.append(SimulationScenario(
                scenario_id=f"market_highvol_{i:04d}",
                category=ScenarioCategory.MARKET_REGIME,
                name=f"High Volatility {i+1}",
                description=f"High volatility period: {volatility*100:.1f}%",
                parameters={
                    "regime": MarketRegime.HIGH_VOLATILITY.value,
                    "volatility": volatility,
                    "garch_params": garch_params,
                    "volatility_clustering": True,
                    "fat_tails": True
                },
                expected_behavior="System should reduce position sizes and widen stops",
                severity="high"
            ))

        # 5. Market crashes (1000 scenarios)
        for i in range(1000):
            crash_magnitude = -np.random.uniform(0.10, 0.50)  # -10% to -50%
            crash_duration_hours = np.random.uniform(1, 48)
            recovery_days = np.random.randint(5, 90)

            scenarios.append(SimulationScenario(
                scenario_id=f"market_crash_{i:04d}",
                category=ScenarioCategory.MARKET_REGIME,
                name=f"Market Crash {i+1}",
                description=f"Market crash: {crash_magnitude*100:.1f}% in {crash_duration_hours:.1f} hours",
                parameters={
                    "regime": MarketRegime.CRASH.value,
                    "crash_magnitude": crash_magnitude,
                    "crash_duration_hours": crash_duration_hours,
                    "recovery_days": recovery_days,
                    "liquidity_crisis": np.random.choice([True, False], p=[0.4, 0.6]),
                    "contagion": np.random.choice([True, False], p=[0.3, 0.7])
                },
                expected_behavior="System should trigger circuit breakers and emergency stops",
                severity="critical"
            ))

        # 6. Bubbles (500 scenarios)
        for i in range(500):
            bubble_growth_rate = np.random.uniform(0.10, 0.50)  # 10-50% per month
            bubble_duration_months = np.random.randint(3, 24)
            pop_probability = np.random.uniform(0.05, 0.30)

            scenarios.append(SimulationScenario(
                scenario_id=f"market_bubble_{i:04d}",
                category=ScenarioCategory.MARKET_REGIME,
                name=f"Bubble {i+1}",
                description=f"Bubble with {bubble_growth_rate*100:.0f}% monthly growth",
                parameters={
                    "regime": MarketRegime.BUBBLE.value,
                    "growth_rate": bubble_growth_rate,
                    "duration_months": bubble_duration_months,
                    "pop_probability": pop_probability,
                    "euphoria_phase": True,
                    "fundamental_disconnect": True
                },
                expected_behavior="System should detect overvaluation and reduce exposure",
                severity="high"
            ))

        # 7. Recovery phases (1000 scenarios)
        for i in range(1000):
            recovery_rate = np.random.uniform(0.02, 0.10)
            recovery_duration = np.random.randint(30, 180)
            false_start_probability = np.random.uniform(0.1, 0.4)

            scenarios.append(SimulationScenario(
                scenario_id=f"market_recovery_{i:04d}",
                category=ScenarioCategory.MARKET_REGIME,
                name=f"Recovery {i+1}",
                description=f"Market recovery at {recovery_rate*100:.1f}% daily",
                parameters={
                    "regime": MarketRegime.RECOVERY.value,
                    "recovery_rate": recovery_rate,
                    "duration_days": recovery_duration,
                    "false_start_probability": false_start_probability,
                    "uncertainty_high": True
                },
                expected_behavior="System should gradually increase exposure",
                severity="medium"
            ))

        logger.info(f"Generated {len(scenarios)} market regime scenarios")
        return scenarios

    def _generate_microstructure_scenarios(self) -> List[SimulationScenario]:
        """
        Generate microstructure scenarios (5,000+)

        Tests system behavior with market microstructure effects
        """
        scenarios = []

        # 1. Bid-ask spread variations (1500 scenarios)
        for i in range(1500):
            base_spread_bps = np.random.uniform(1, 100)  # 0.01% to 1%
            spread_volatility = np.random.uniform(0.1, 2.0)

            scenarios.append(SimulationScenario(
                scenario_id=f"micro_spread_{i:04d}",
                category=ScenarioCategory.MICROSTRUCTURE,
                name=f"Spread Variation {i+1}",
                description=f"Bid-ask spread: {base_spread_bps:.1f} bps",
                parameters={
                    "base_spread_bps": base_spread_bps,
                    "spread_volatility": spread_volatility,
                    "spread_widens_on_volatility": True,
                    "asymmetric_spread": np.random.choice([True, False])
                },
                expected_behavior="System should account for transaction costs",
                severity="medium"
            ))

        # 2. Order book depth (1000 scenarios)
        for i in range(1000):
            depth_levels = np.random.randint(5, 50)
            total_depth_btc = np.random.uniform(10, 1000)
            depth_imbalance = np.random.uniform(-0.5, 0.5)

            scenarios.append(SimulationScenario(
                scenario_id=f"micro_depth_{i:04d}",
                category=ScenarioCategory.MICROSTRUCTURE,
                name=f"Order Book Depth {i+1}",
                description=f"Order book with {total_depth_btc:.0f} BTC depth",
                parameters={
                    "depth_levels": depth_levels,
                    "total_depth_btc": total_depth_btc,
                    "depth_imbalance": depth_imbalance,
                    "book_updates_per_second": np.random.uniform(10, 500)
                },
                expected_behavior="System should split large orders appropriately",
                severity="high"
            ))

        # 3. Slippage scenarios (1000 scenarios)
        for i in range(1000):
            slippage_bps = np.random.uniform(1, 200)
            order_size_btc = np.random.uniform(0.1, 100)

            scenarios.append(SimulationScenario(
                scenario_id=f"micro_slippage_{i:04d}",
                category=ScenarioCategory.MICROSTRUCTURE,
                name=f"Slippage {i+1}",
                description=f"Expected slippage: {slippage_bps:.1f} bps for {order_size_btc:.2f} BTC",
                parameters={
                    "slippage_bps": slippage_bps,
                    "order_size_btc": order_size_btc,
                    "market_impact_coefficient": np.random.uniform(0.1, 1.0)
                },
                expected_behavior="System should minimize slippage through smart execution",
                severity="high"
            ))

        # 4. Latency scenarios (800 scenarios)
        for i in range(800):
            network_latency_ms = np.random.uniform(1, 500)
            latency_variance_ms = np.random.uniform(0, 100)
            packet_loss_rate = np.random.uniform(0, 0.05)

            scenarios.append(SimulationScenario(
                scenario_id=f"micro_latency_{i:04d}",
                category=ScenarioCategory.MICROSTRUCTURE,
                name=f"Latency {i+1}",
                description=f"Network latency: {network_latency_ms:.0f} ms ± {latency_variance_ms:.0f} ms",
                parameters={
                    "network_latency_ms": network_latency_ms,
                    "latency_variance_ms": latency_variance_ms,
                    "packet_loss_rate": packet_loss_rate,
                    "latency_spikes": np.random.choice([True, False], p=[0.3, 0.7])
                },
                expected_behavior="System should handle latency gracefully",
                severity="high"
            ))

        # 5. Quote stuffing / HFT manipulation (700 scenarios)
        for i in range(700):
            quotes_per_second = np.random.uniform(1000, 10000)
            cancel_rate = np.random.uniform(0.90, 0.99)

            scenarios.append(SimulationScenario(
                scenario_id=f"micro_stuffing_{i:04d}",
                category=ScenarioCategory.MICROSTRUCTURE,
                name=f"Quote Stuffing {i+1}",
                description=f"{quotes_per_second:.0f} quotes/sec, {cancel_rate*100:.1f}% cancellation",
                parameters={
                    "quotes_per_second": quotes_per_second,
                    "cancel_rate": cancel_rate,
                    "layering": np.random.choice([True, False]),
                    "spoofing": np.random.choice([True, False])
                },
                expected_behavior="System should detect and ignore manipulative quotes",
                severity="critical"
            ))

        logger.info(f"Generated {len(scenarios)} microstructure scenarios")
        return scenarios

    def _generate_risk_event_scenarios(self) -> List[SimulationScenario]:
        """
        Generate risk event scenarios (3,000+)

        Tests system response to various risk events
        """
        scenarios = []

        # 1. Exchange outages (500 scenarios)
        for i in range(500):
            outage_duration_minutes = np.random.uniform(1, 240)
            partial_outage = np.random.choice([True, False])

            scenarios.append(SimulationScenario(
                scenario_id=f"risk_outage_{i:04d}",
                category=ScenarioCategory.RISK_EVENT,
                name=f"Exchange Outage {i+1}",
                description=f"Exchange outage for {outage_duration_minutes:.0f} minutes",
                parameters={
                    "outage_duration_minutes": outage_duration_minutes,
                    "partial_outage": partial_outage,
                    "data_available": np.random.choice([True, False]),
                    "orders_cancelled": np.random.choice([True, False])
                },
                expected_behavior="System should halt trading and preserve positions",
                severity="critical"
            ))

        # 2. Flash crashes (400 scenarios)
        for i in range(400):
            crash_magnitude = -np.random.uniform(0.10, 0.70)
            crash_duration_seconds = np.random.uniform(10, 600)

            scenarios.append(SimulationScenario(
                scenario_id=f"risk_flash_{i:04d}",
                category=ScenarioCategory.RISK_EVENT,
                name=f"Flash Crash {i+1}",
                description=f"Flash crash: {crash_magnitude*100:.0f}% in {crash_duration_seconds:.0f}s",
                parameters={
                    "crash_magnitude": crash_magnitude,
                    "crash_duration_seconds": crash_duration_seconds,
                    "recovery_percentage": np.random.uniform(0.5, 1.0),
                    "stop_losses_triggered": True
                },
                expected_behavior="System should trigger circuit breakers",
                severity="critical"
            ))

        # 3. Liquidity crises (500 scenarios)
        for i in range(500):
            liquidity_reduction = np.random.uniform(0.50, 0.95)
            duration_hours = np.random.uniform(1, 72)

            scenarios.append(SimulationScenario(
                scenario_id=f"risk_liquidity_{i:04d}",
                category=ScenarioCategory.RISK_EVENT,
                name=f"Liquidity Crisis {i+1}",
                description=f"Liquidity reduced by {liquidity_reduction*100:.0f}%",
                parameters={
                    "liquidity_reduction": liquidity_reduction,
                    "duration_hours": duration_hours,
                    "spread_widening_multiplier": np.random.uniform(2, 20)
                },
                expected_behavior="System should reduce position sizes",
                severity="critical"
            ))

        # 4. Regulatory events (400 scenarios)
        for i in range(400):
            trading_halt = np.random.choice([True, False])
            withdrawal_restrictions = np.random.choice([True, False])

            scenarios.append(SimulationScenario(
                scenario_id=f"risk_regulatory_{i:04d}",
                category=ScenarioCategory.RISK_EVENT,
                name=f"Regulatory Event {i+1}",
                description="Sudden regulatory announcement",
                parameters={
                    "trading_halt": trading_halt,
                    "withdrawal_restrictions": withdrawal_restrictions,
                    "uncertainty_duration_days": np.random.randint(1, 30),
                    "market_panic": np.random.choice([True, False])
                },
                expected_behavior="System should preserve capital and await clarity",
                severity="high"
            ))

        # 5. Black swan events (300 scenarios)
        for i in range(300):
            impact_magnitude = -np.random.uniform(0.30, 0.90)
            contagion_factor = np.random.uniform(0, 1)

            scenarios.append(SimulationScenario(
                scenario_id=f"risk_blackswan_{i:04d}",
                category=ScenarioCategory.RISK_EVENT,
                name=f"Black Swan {i+1}",
                description=f"Unprecedented event with {impact_magnitude*100:.0f}% impact",
                parameters={
                    "impact_magnitude": impact_magnitude,
                    "contagion_factor": contagion_factor,
                    "correlation_breakdown": True,
                    "volatility_explosion": True,
                    "fat_tail_event": True
                },
                expected_behavior="System should emergency stop and protect capital",
                severity="critical"
            ))

        # 6. Data feed failures (400 scenarios)
        for i in range(400):
            data_loss_percentage = np.random.uniform(0.10, 1.0)
            corrupted_data = np.random.choice([True, False])

            scenarios.append(SimulationScenario(
                scenario_id=f"risk_datafeed_{i:04d}",
                category=ScenarioCategory.RISK_EVENT,
                name=f"Data Feed Failure {i+1}",
                description=f"Data loss: {data_loss_percentage*100:.0f}%",
                parameters={
                    "data_loss_percentage": data_loss_percentage,
                    "corrupted_data": corrupted_data,
                    "stale_data": np.random.choice([True, False]),
                    "timestamp_errors": np.random.choice([True, False])
                },
                expected_behavior="System should detect bad data and halt",
                severity="critical"
            ))

        # 7. Execution failures (500 scenarios)
        for i in range(500):
            order_rejection_rate = np.random.uniform(0.10, 0.90)
            partial_fills = np.random.choice([True, False])

            scenarios.append(SimulationScenario(
                scenario_id=f"risk_execution_{i:04d}",
                category=ScenarioCategory.RISK_EVENT,
                name=f"Execution Failure {i+1}",
                description=f"Order rejection rate: {order_rejection_rate*100:.0f}%",
                parameters={
                    "order_rejection_rate": order_rejection_rate,
                    "partial_fills": partial_fills,
                    "timeout_errors": np.random.choice([True, False])
                },
                expected_behavior="System should retry with exponential backoff",
                severity="high"
            ))

        logger.info(f"Generated {len(scenarios)} risk event scenarios")
        return scenarios

    def _generate_adversarial_scenarios(self) -> List[SimulationScenario]:
        """
        Generate adversarial scenarios (2,000+)

        Tests system against adversarial attacks and malicious inputs
        """
        scenarios = []

        # 1. Adversarial price inputs (500 scenarios)
        for i in range(500):
            attack_type = np.random.choice(['gradient_attack', 'evolutionary_attack', 'transfer_attack'])
            perturbation_magnitude = np.random.uniform(0.001, 0.10)

            scenarios.append(SimulationScenario(
                scenario_id=f"adv_price_{i:04d}",
                category=ScenarioCategory.ADVERSARIAL,
                name=f"Adversarial Price {i+1}",
                description=f"{attack_type} with {perturbation_magnitude*100:.1f}% perturbation",
                parameters={
                    "attack_type": attack_type,
                    "perturbation_magnitude": perturbation_magnitude,
                    "targeted": np.random.choice([True, False]),
                    "imperceptible": True
                },
                expected_behavior="System should detect adversarial inputs",
                severity="critical"
            ))

        # 2. Malformed data (500 scenarios)
        for i in range(500):
            malformation_type = np.random.choice(['null', 'nan', 'inf', 'negative', 'wrong_type'])

            scenarios.append(SimulationScenario(
                scenario_id=f"adv_malformed_{i:04d}",
                category=ScenarioCategory.ADVERSARIAL,
                name=f"Malformed Data {i+1}",
                description=f"Malformed data type: {malformation_type}",
                parameters={
                    "malformation_type": malformation_type,
                    "injection_rate": np.random.uniform(0.01, 0.50)
                },
                expected_behavior="System should reject malformed inputs",
                severity="critical"
            ))

        # 3. Data poisoning (400 scenarios)
        for i in range(400):
            poison_percentage = np.random.uniform(0.01, 0.20)
            poison_type = np.random.choice(['label_flip', 'backdoor', 'clean_label'])

            scenarios.append(SimulationScenario(
                scenario_id=f"adv_poison_{i:04d}",
                category=ScenarioCategory.ADVERSARIAL,
                name=f"Data Poisoning {i+1}",
                description=f"Poisoning: {poison_percentage*100:.1f}% ({poison_type})",
                parameters={
                    "poison_percentage": poison_percentage,
                    "poison_type": poison_type,
                    "stealthy": True
                },
                expected_behavior="System should detect data anomalies",
                severity="critical"
            ))

        # 4. Model evasion (300 scenarios)
        for i in range(300):
            evasion_technique = np.random.choice(['feature_squeezing', 'defensive_distillation', 'ensemble_evasion'])

            scenarios.append(SimulationScenario(
                scenario_id=f"adv_evasion_{i:04d}",
                category=ScenarioCategory.ADVERSARIAL,
                name=f"Model Evasion {i+1}",
                description=f"Evasion using {evasion_technique}",
                parameters={
                    "evasion_technique": evasion_technique,
                    "success_rate": np.random.uniform(0.10, 0.80)
                },
                expected_behavior="Ensemble should detect inconsistencies",
                severity="high"
            ))

        # 5. Timestamp manipulation (300 scenarios)
        for i in range(300):
            time_shift_seconds = np.random.uniform(-3600, 3600)  # ±1 hour

            scenarios.append(SimulationScenario(
                scenario_id=f"adv_timestamp_{i:04d}",
                category=ScenarioCategory.ADVERSARIAL,
                name=f"Timestamp Manipulation {i+1}",
                description=f"Time shift: {time_shift_seconds:.0f} seconds",
                parameters={
                    "time_shift_seconds": time_shift_seconds,
                    "future_data_injection": time_shift_seconds > 0
                },
                expected_behavior="System should detect temporal anomalies",
                severity="critical"
            ))

        logger.info(f"Generated {len(scenarios)} adversarial scenarios")
        return scenarios

    def run_all_scenarios(
        self,
        system_under_test: Callable[[SimulationScenario], SimulationResult],
        parallel: bool = True
    ) -> List[SimulationResult]:
        """
        Run all scenarios against the system

        Args:
            system_under_test: Function that takes scenario and returns result
            parallel: Whether to run in parallel

        Returns:
            List of simulation results
        """
        logger.info(f"Running {len(self.scenarios)} scenarios...")

        if parallel:
            results = self._run_parallel(system_under_test)
        else:
            results = self._run_sequential(system_under_test)

        self.results = results
        self.total_scenarios_run = len(results)
        self.scenarios_passed = sum(1 for r in results if r.passed)
        self.scenarios_failed = sum(1 for r in results if not r.passed)

        logger.info(f"Completed: {self.scenarios_passed} passed, {self.scenarios_failed} failed")

        return results

    def _run_parallel(
        self,
        system_under_test: Callable[[SimulationScenario], SimulationResult]
    ) -> List[SimulationResult]:
        """Run scenarios in parallel"""
        results = []

        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(system_under_test, scenario): scenario
                for scenario in self.scenarios
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)

                    if len(results) % 100 == 0:
                        logger.info(f"Progress: {len(results)}/{len(self.scenarios)} scenarios completed")

                except Exception as e:
                    scenario = futures[future]
                    logger.error(f"Scenario {scenario.scenario_id} failed with exception: {e}")
                    results.append(SimulationResult(
                        scenario_id=scenario.scenario_id,
                        scenario_name=scenario.name,
                        passed=False,
                        execution_time_ms=0,
                        metrics={},
                        errors=[str(e)],
                        warnings=[],
                        hallucinations_detected=0,
                        decisions_rejected=0,
                        validation_failures=0,
                        system_stable=False,
                        data_corruption=False,
                        details={}
                    ))

        return results

    def _run_sequential(
        self,
        system_under_test: Callable[[SimulationScenario], SimulationResult]
    ) -> List[SimulationResult]:
        """Run scenarios sequentially"""
        results = []

        for i, scenario in enumerate(self.scenarios):
            try:
                result = system_under_test(scenario)
                results.append(result)

                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i+1}/{len(self.scenarios)} scenarios completed")

            except Exception as e:
                logger.error(f"Scenario {scenario.scenario_id} failed with exception: {e}")
                results.append(SimulationResult(
                    scenario_id=scenario.scenario_id,
                    scenario_name=scenario.name,
                    passed=False,
                    execution_time_ms=0,
                    metrics={},
                    errors=[str(e)],
                    warnings=[],
                    hallucinations_detected=0,
                    decisions_rejected=0,
                    validation_failures=0,
                    system_stable=False,
                    data_corruption=False,
                    details={}
                ))

        return results

    @lru_cache(maxsize=128)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of simulation results"""
        if not self.results:
            return {"error": "No results available"}

        pass_rate = self.scenarios_passed / self.total_scenarios_run if self.total_scenarios_run > 0 else 0

        # Group by category
        results_by_category = {}
        for result in self.results:
            # Find original scenario
            scenario = next((s for s in self.scenarios if s.scenario_id == result.scenario_id), None)
            if scenario:
                cat = scenario.category.value
                if cat not in results_by_category:
                    results_by_category[cat] = {"passed": 0, "failed": 0}

                if result.passed:
                    results_by_category[cat]["passed"] += 1
                else:
                    results_by_category[cat]["failed"] += 1

        return {
            "total_scenarios": self.total_scenarios_run,
            "passed": self.scenarios_passed,
            "failed": self.scenarios_failed,
            "pass_rate": pass_rate,
            "results_by_category": results_by_category,
            "total_hallucinations": sum(r.hallucinations_detected for r in self.results),
            "total_decisions_rejected": sum(r.decisions_rejected for r in self.results),
            "total_validation_failures": sum(r.validation_failures for r in self.results),
            "unstable_systems": sum(1 for r in self.results if not r.system_stable)
        }
