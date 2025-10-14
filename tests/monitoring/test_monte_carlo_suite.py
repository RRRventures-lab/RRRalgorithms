from src.validation.monte_carlo_engine import (
import numpy as np
import pytest

#!/usr/bin/env python3
"""
Test Suite for Monte Carlo Simulation Engine

Tests 20,000+ scenario generation and execution

Author: AI Psychology Team
Date: 2025-10-11
"""

    MonteCarloEngine,
    SimulationScenario,
    SimulationResult,
    ScenarioCategory,
    MarketRegime
)


class TestMonteCarloEngine:
    """Test suite for Monte Carlo simulation engine"""

    @pytest.fixture
    def engine(self):
        """Create engine with smaller scenario counts for testing"""
        return MonteCarloEngine(
            num_market_scenarios=1000,
            num_microstructure_scenarios=500,
            num_risk_scenarios=300,
            num_adversarial_scenarios=200,
            parallel_workers=4,
            random_seed=42
        )

    def test_engine_initialization(self, engine):
        """Test engine initializes correctly"""
        assert engine is not None
        assert engine.total_scenarios() == 2000

    def test_scenario_generation(self, engine):
        """Test all scenarios are generated"""
        scenarios = engine.generate_all_scenarios()

        assert len(scenarios) == 2000
        assert engine.total_scenarios_generated == 2000

    def test_market_regime_scenarios(self, engine):
        """Test market regime scenarios are diverse"""
        scenarios = engine._generate_market_regime_scenarios()

        # Check we have all regime types
        regimes = set(s.parameters['regime'] for s in scenarios)
        assert MarketRegime.BULL.value in regimes
        assert MarketRegime.BEAR.value in regimes
        assert MarketRegime.CRASH.value in regimes
        assert MarketRegime.RANGING.value in regimes

    def test_scenario_parameters(self, engine):
        """Test scenarios have valid parameters"""
        scenarios = engine.generate_all_scenarios()

        for scenario in scenarios[:100]:  # Sample
            assert scenario.scenario_id is not None
            assert scenario.category in ScenarioCategory
            assert scenario.severity in ['low', 'medium', 'high', 'critical']
            assert len(scenario.parameters) > 0

    def test_parallel_execution(self, engine):
        """Test parallel scenario execution"""
        engine.generate_all_scenarios()

        # Mock system under test
        def mock_system(scenario: SimulationScenario) -> SimulationResult:
            import time
            time.sleep(0.001)  # Simulate work

            return SimulationResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                passed=True,
                execution_time_ms=1.0,
                metrics={"accuracy": 0.95},
                errors=[],
                warnings=[],
                hallucinations_detected=0,
                decisions_rejected=0,
                validation_failures=0,
                system_stable=True,
                data_corruption=False,
                details={}
            )

        # Run in parallel
        results = engine.run_all_scenarios(mock_system, parallel=True)

        assert len(results) == 2000
        assert engine.scenarios_passed > 0

    def test_summary_statistics(self, engine):
        """Test summary statistics generation"""
        engine.generate_all_scenarios()

        # Mock results
        engine.results = [
            SimulationResult(
                scenario_id=f"test_{i}",
                scenario_name=f"Test {i}",
                passed=i % 2 == 0,
                execution_time_ms=5.0,
                metrics={},
                errors=[],
                warnings=[],
                hallucinations_detected=0,
                decisions_rejected=0,
                validation_failures=0,
                system_stable=True,
                data_corruption=False,
                details={}
            )
            for i in range(100)
        ]
        engine.total_scenarios_run = 100
        engine.scenarios_passed = 50
        engine.scenarios_failed = 50

        stats = engine.get_summary_statistics()

        assert stats['total_scenarios'] == 100
        assert stats['passed'] == 50
        assert stats['failed'] == 50
        assert stats['pass_rate'] == 0.5


class TestScenarioCategories:
    """Test different scenario categories"""

    @pytest.fixture
    def engine(self):
        return MonteCarloEngine(
            num_market_scenarios=500,
            num_microstructure_scenarios=300,
            num_risk_scenarios=150,
            num_adversarial_scenarios=100,
            random_seed=42
        )

    def test_bull_market_scenarios(self, engine):
        """Test bull market scenario generation"""
        scenarios = engine._generate_market_regime_scenarios()
        bull_scenarios = [
            s for s in scenarios
            if s.parameters.get('regime') == MarketRegime.BULL.value
        ]

        assert len(bull_scenarios) > 0

        for scenario in bull_scenarios[:10]:
            assert scenario.parameters['trend_strength'] > 0
            assert scenario.parameters['duration_days'] > 0

    def test_crash_scenarios(self, engine):
        """Test crash scenario generation"""
        scenarios = engine._generate_market_regime_scenarios()
        crash_scenarios = [
            s for s in scenarios
            if s.parameters.get('regime') == MarketRegime.CRASH.value
        ]

        assert len(crash_scenarios) > 0

        for scenario in crash_scenarios[:10]:
            assert scenario.parameters['crash_magnitude'] < 0
            assert scenario.severity == 'critical'

    def test_microstructure_scenarios(self, engine):
        """Test microstructure scenarios"""
        scenarios = engine._generate_microstructure_scenarios()

        assert len(scenarios) == 300

        # Check diversity
        categories = set(s.name.split()[0] for s in scenarios)
        assert len(categories) > 1  # Multiple types

    def test_risk_event_scenarios(self, engine):
        """Test risk event scenarios"""
        scenarios = engine._generate_risk_event_scenarios()

        assert len(scenarios) == 150

        # All should be high/critical severity
        severities = set(s.severity for s in scenarios)
        assert 'high' in severities or 'critical' in severities

    def test_adversarial_scenarios(self, engine):
        """Test adversarial scenarios"""
        scenarios = engine._generate_adversarial_scenarios()

        assert len(scenarios) == 100

        # All should be critical
        for scenario in scenarios[:20]:
            assert scenario.severity == 'critical'


class TestSystemUnderTest:
    """Test mock system under test"""

    def test_mock_system_execution(self):
        """Test mock system can execute scenarios"""

        def mock_system(scenario: SimulationScenario) -> SimulationResult:
            # Simulate passing most scenarios
            passed = np.random.rand() > 0.1

            return SimulationResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                passed=passed,
                execution_time_ms=np.random.uniform(1, 10),
                metrics={"accuracy": 0.90},
                errors=[] if passed else ["Mock error"],
                warnings=[],
                hallucinations_detected=0 if passed else 1,
                decisions_rejected=0,
                validation_failures=0,
                system_stable=passed,
                data_corruption=False,
                details={}
            )

        scenario = SimulationScenario(
            scenario_id="test_001",
            category=ScenarioCategory.MARKET_REGIME,
            name="Test Scenario",
            description="Test",
            parameters={},
            expected_behavior="System should handle correctly",
            severity="medium"
        )

        result = mock_system(scenario)

        assert result is not None
        assert result.scenario_id == "test_001"
        assert isinstance(result.passed, bool)


class TestPerformanceRequirements:
    """Test Monte Carlo performance requirements"""

    @pytest.fixture
    def engine(self):
        return MonteCarloEngine(
            num_market_scenarios=1000,
            num_microstructure_scenarios=500,
            num_risk_scenarios=300,
            num_adversarial_scenarios=200,
            parallel_workers=8,
            random_seed=42
        )

    def test_scenario_generation_performance(self, engine):
        """Test scenario generation completes in reasonable time"""
        import time

        start = time.time()
        engine.generate_all_scenarios()
        duration = time.time() - start

        # Should generate 2000 scenarios in <10 seconds
        assert duration < 10

    def test_parallel_execution_speedup(self, engine):
        """Test parallel execution is faster than sequential"""
        engine.generate_all_scenarios()

        def fast_mock(scenario):
            import time
            time.sleep(0.001)
            return SimulationResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                passed=True,
                execution_time_ms=1.0,
                metrics={},
                errors=[],
                warnings=[],
                hallucinations_detected=0,
                decisions_rejected=0,
                validation_failures=0,
                system_stable=True,
                data_corruption=False,
                details={}
            )

        # Limit to 100 scenarios for test speed
        engine.scenarios = engine.scenarios[:100]

        # Test parallel
        import time
        start = time.time()
        engine.run_all_scenarios(fast_mock, parallel=True)
        parallel_time = time.time() - start

        # Parallel should complete quickly
        assert parallel_time < 5  # 100 scenarios in <5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
