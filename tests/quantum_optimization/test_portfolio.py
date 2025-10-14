from pathlib import Path
from quantum.portfolio import (
import numpy as np
import pytest
import sys

"""
Tests for portfolio optimization module

Author: Quantum Optimization Team
Date: 2025-10-11
"""


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    QAOAPortfolioOptimizer,
    ClassicalMarkowitzOptimizer,
    PortfolioConstraints,
    compare_optimizers
)


@pytest.fixture
def sample_portfolio_data():
    """Generate sample portfolio data for testing"""
    np.random.seed(42)
    n_assets = 5
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)

    # Generate positive definite covariance matrix
    random_matrix = np.random.randn(n_assets, n_assets)
    covariance = np.dot(random_matrix, random_matrix.T) * 0.01

    return expected_returns, covariance


@pytest.fixture
def default_constraints():
    """Default portfolio constraints"""
    return PortfolioConstraints(
        min_weight=0.0,
        max_weight=1.0,
        sum_to_one=True,
        long_only=True,
        risk_tolerance=1.0
    )


class TestQAOAPortfolioOptimizer:
    """Test QAOA portfolio optimizer"""

    def test_initialization(self):
        """Test optimizer initialization"""
        optimizer = QAOAPortfolioOptimizer(n_layers=3, n_iterations=50)
        assert optimizer.n_layers == 3
        assert optimizer.n_iterations == 50

    def test_optimize_basic(self, sample_portfolio_data, default_constraints):
        """Test basic optimization"""
        returns, cov = sample_portfolio_data
        optimizer = QAOAPortfolioOptimizer(n_layers=2, n_iterations=10)

        result = optimizer.optimize(returns, cov, default_constraints)

        # Check result properties
        assert result.weights is not None
        assert len(result.weights) == len(returns)
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-4)
        assert np.all(result.weights >= 0)
        assert result.execution_time > 0

    def test_weights_sum_to_one(self, sample_portfolio_data, default_constraints):
        """Test that weights sum to 1"""
        returns, cov = sample_portfolio_data
        optimizer = QAOAPortfolioOptimizer(n_layers=2, n_iterations=10)

        result = optimizer.optimize(returns, cov, default_constraints)

        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-4)

    def test_weights_respect_bounds(self, sample_portfolio_data):
        """Test that weights respect min/max bounds"""
        returns, cov = sample_portfolio_data
        constraints = PortfolioConstraints(
            min_weight=0.1,
            max_weight=0.4,
            long_only=True
        )

        optimizer = QAOAPortfolioOptimizer(n_layers=2, n_iterations=10)
        result = optimizer.optimize(returns, cov, constraints)

        assert np.all(result.weights >= 0.1 - 1e-4)
        assert np.all(result.weights <= 0.4 + 1e-4)

    def test_metrics_calculation(self, sample_portfolio_data, default_constraints):
        """Test portfolio metrics calculation"""
        returns, cov = sample_portfolio_data
        optimizer = QAOAPortfolioOptimizer(n_layers=2, n_iterations=10)

        result = optimizer.optimize(returns, cov, default_constraints)

        # Manually calculate and verify
        expected_return = np.dot(result.weights, returns)
        volatility = np.sqrt(np.dot(result.weights, np.dot(cov, result.weights)))

        assert np.allclose(result.expected_return, expected_return)
        assert np.allclose(result.volatility, volatility)
        assert result.sharpe_ratio > 0


class TestClassicalMarkowitzOptimizer:
    """Test classical Markowitz optimizer"""

    def test_initialization(self):
        """Test optimizer initialization"""
        optimizer = ClassicalMarkowitzOptimizer()
        assert optimizer is not None

    def test_optimize_basic(self, sample_portfolio_data, default_constraints):
        """Test basic optimization"""
        returns, cov = sample_portfolio_data
        optimizer = ClassicalMarkowitzOptimizer()

        result = optimizer.optimize(returns, cov, default_constraints)

        assert result.weights is not None
        assert len(result.weights) == len(returns)
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-4)
        assert result.converged

    def test_comparison_with_qaoa(self, sample_portfolio_data, default_constraints):
        """Test that classical and QAOA give reasonable results"""
        returns, cov = sample_portfolio_data

        qaoa = QAOAPortfolioOptimizer(n_layers=2, n_iterations=20)
        classical = ClassicalMarkowitzOptimizer()

        qaoa_result = qaoa.optimize(returns, cov, default_constraints)
        classical_result = classical.optimize(returns, cov, default_constraints)

        # Both should produce valid portfolios
        assert qaoa_result.sharpe_ratio > 0
        assert classical_result.sharpe_ratio > 0

        # Sharpe ratios should be in similar range
        ratio = qaoa_result.sharpe_ratio / classical_result.sharpe_ratio
        assert 0.5 < ratio < 2.0  # Within 2x of each other


class TestCompareOptimizers:
    """Test optimizer comparison function"""

    def test_compare_optimizers(self, sample_portfolio_data, default_constraints):
        """Test comparison function"""
        returns, cov = sample_portfolio_data

        comparison = compare_optimizers(returns, cov, default_constraints)

        assert 'qaoa' in comparison
        assert 'classical' in comparison
        assert 'performance_ratio' in comparison

        # Check structure
        assert 'sharpe_ratio' in comparison['qaoa']
        assert 'execution_time' in comparison['qaoa']
        assert 'weights' in comparison['qaoa']


class TestPortfolioConstraints:
    """Test portfolio constraints"""

    def test_default_constraints(self):
        """Test default constraint creation"""
        constraints = PortfolioConstraints()

        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 1.0
        assert constraints.sum_to_one is True
        assert constraints.long_only is True

    def test_custom_constraints(self):
        """Test custom constraints"""
        constraints = PortfolioConstraints(
            min_weight=0.1,
            max_weight=0.5,
            risk_tolerance=2.0
        )

        assert constraints.min_weight == 0.1
        assert constraints.max_weight == 0.5
        assert constraints.risk_tolerance == 2.0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_asset(self, default_constraints):
        """Test with single asset"""
        returns = np.array([0.1])
        cov = np.array([[0.01]])

        optimizer = QAOAPortfolioOptimizer(n_layers=1, n_iterations=5)
        result = optimizer.optimize(returns, cov, default_constraints)

        assert np.allclose(result.weights, [1.0])

    def test_identical_assets(self, default_constraints):
        """Test with identical assets"""
        returns = np.array([0.1, 0.1, 0.1])
        cov = np.array([[0.01, 0, 0],
                       [0, 0.01, 0],
                       [0, 0, 0.01]])

        optimizer = QAOAPortfolioOptimizer(n_layers=2, n_iterations=10)
        result = optimizer.optimize(returns, cov, default_constraints)

        # Should distribute evenly
        assert np.allclose(result.weights, [1/3, 1/3, 1/3], atol=0.2)

    def test_large_portfolio(self, default_constraints):
        """Test with larger portfolio"""
        np.random.seed(42)
        n_assets = 20
        returns = np.random.uniform(0.05, 0.15, n_assets)
        random_matrix = np.random.randn(n_assets, n_assets)
        cov = np.dot(random_matrix, random_matrix.T) * 0.01

        optimizer = QAOAPortfolioOptimizer(n_layers=2, n_iterations=10)
        result = optimizer.optimize(returns, cov, default_constraints)

        assert len(result.weights) == n_assets
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
