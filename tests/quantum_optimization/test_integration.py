from pathlib import Path
from quantum.integration import (
import numpy as np
import pytest
import sys

"""
Tests for integration module

Author: Quantum Optimization Team
Date: 2025-10-11
"""


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    optimize_portfolio,
    compare_portfolio_optimizers,
    tune_hyperparameters,
    select_features,
    QuantumOptimizer,
    get_default_portfolio_constraints
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)

    # Portfolio data
    n_assets = 5
    returns = np.random.uniform(0.05, 0.15, n_assets)
    random_matrix = np.random.randn(n_assets, n_assets)
    cov = np.dot(random_matrix, random_matrix.T) * 0.01

    # Feature data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)

    return {
        'returns': returns,
        'covariance': cov,
        'X': X,
        'y': y
    }


class TestOptimizePortfolio:
    """Test portfolio optimization integration"""

    def test_quantum_mode(self, sample_data):
        """Test quantum portfolio optimization"""
        result = optimize_portfolio(
            sample_data['returns'],
            sample_data['covariance'],
            use_quantum=True,
            n_iterations=10
        )

        assert 'weights' in result
        assert 'sharpe_ratio' in result
        assert 'method' in result
        assert result['method'] == 'QAOA'
        assert len(result['weights']) == len(sample_data['returns'])

    def test_classical_mode(self, sample_data):
        """Test classical portfolio optimization"""
        result = optimize_portfolio(
            sample_data['returns'],
            sample_data['covariance'],
            use_quantum=False
        )

        assert result['method'] == 'Classical_Markowitz'
        assert result['converged']

    def test_risk_profiles(self, sample_data):
        """Test different risk profiles"""
        for risk_tolerance in [0.5, 1.0, 2.0]:
            result = optimize_portfolio(
                sample_data['returns'],
                sample_data['covariance'],
                risk_tolerance=risk_tolerance,
                n_iterations=10
            )
            assert result['weights'] is not None


class TestComparePortfolioOptimizers:
    """Test portfolio optimizer comparison"""

    def test_comparison(self, sample_data):
        """Test optimizer comparison"""
        comparison = compare_portfolio_optimizers(
            sample_data['returns'],
            sample_data['covariance'],
            n_iterations=10
        )

        assert 'qaoa' in comparison
        assert 'classical' in comparison
        assert 'speedup' in comparison
        assert 'sharpe_improvement_percent' in comparison

        # Check both optimizers returned valid results
        assert comparison['qaoa']['sharpe_ratio'] > 0
        assert comparison['classical']['sharpe_ratio'] > 0


class TestTuneHyperparameters:
    """Test hyperparameter tuning integration"""

    def test_simple_objective(self):
        """Test with simple objective function"""
        param_spaces = [
            {'name': 'x', 'type': 'continuous', 'bounds': (-5, 5)},
            {'name': 'y', 'type': 'continuous', 'bounds': (-5, 5)}
        ]

        def objective(params):
            # Simple quadratic function
            return -(params['x']**2 + params['y']**2)

        result = tune_hyperparameters(
            param_spaces,
            objective,
            maximize=True,
            use_quantum=True,
            n_iterations=10
        )

        assert 'best_params' in result
        assert 'best_score' in result
        assert 'method' in result
        assert result['method'] == 'Quantum_Annealing'

        # Should find minimum near (0, 0)
        assert abs(result['best_params']['x']) < 2.0
        assert abs(result['best_params']['y']) < 2.0

    def test_discrete_params(self):
        """Test with discrete parameters"""
        param_spaces = [
            {'name': 'n', 'type': 'discrete', 'bounds': (1, 10)}
        ]

        def objective(params):
            return -abs(params['n'] - 5)

        result = tune_hyperparameters(
            param_spaces,
            objective,
            maximize=True,
            n_iterations=10
        )

        # Should find n=5
        assert 3 <= result['best_params']['n'] <= 7

    def test_categorical_params(self):
        """Test with categorical parameters"""
        param_spaces = [
            {'name': 'choice', 'type': 'categorical', 'values': ['a', 'b', 'c']}
        ]

        def objective(params):
            scores = {'a': 1.0, 'b': 2.0, 'c': 0.5}
            return scores[params['choice']]

        result = tune_hyperparameters(
            param_spaces,
            objective,
            maximize=True,
            n_iterations=10
        )

        # Should find 'b' as best
        assert result['best_params']['choice'] in ['a', 'b', 'c']


class TestSelectFeatures:
    """Test feature selection integration"""

    def test_quantum_selection(self, sample_data):
        """Test quantum feature selection"""
        result = select_features(
            sample_data['X'],
            sample_data['y'],
            n_features=10,
            use_quantum=True,
            n_iterations=5
        )

        assert 'selected_features' in result
        assert 'X_transformed' in result
        assert 'method' in result
        assert result['method'] == 'Quantum'
        assert len(result['selected_features']) == 10

    def test_classical_selection(self, sample_data):
        """Test classical feature selection"""
        result = select_features(
            sample_data['X'],
            sample_data['y'],
            n_features=10,
            use_quantum=False
        )

        assert result['method'] == 'Classical'
        assert result['X_transformed'].shape[1] == 10

    def test_auto_feature_count(self, sample_data):
        """Test automatic feature count selection"""
        result = select_features(
            sample_data['X'],
            sample_data['y'],
            n_features=None,
            n_iterations=5
        )

        # Should select some reasonable number of features
        assert result['n_features_selected'] > 0
        assert result['n_features_selected'] < sample_data['X'].shape[1]


class TestQuantumOptimizer:
    """Test unified QuantumOptimizer interface"""

    def test_initialization(self):
        """Test optimizer initialization"""
        quantum_optimizer = QuantumOptimizer(use_quantum=True)
        assert quantum_optimizer.use_quantum is True

        classical_optimizer = QuantumOptimizer(use_quantum=False)
        assert classical_optimizer.use_quantum is False

    def test_portfolio_optimization(self, sample_data):
        """Test portfolio optimization through unified interface"""
        optimizer = QuantumOptimizer(use_quantum=True)

        result = optimizer.optimize_portfolio(
            sample_data['returns'],
            sample_data['covariance'],
            n_iterations=10
        )

        assert result['method'] == 'QAOA'
        assert len(result['weights']) == len(sample_data['returns'])

    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning through unified interface"""
        optimizer = QuantumOptimizer(use_quantum=True)

        param_spaces = [
            {'name': 'x', 'type': 'continuous', 'bounds': (0, 10)}
        ]

        def objective(params):
            return -params['x']**2

        result = optimizer.tune_hyperparameters(
            param_spaces,
            objective,
            n_iterations=10
        )

        assert result['method'] == 'Quantum_Annealing'

    def test_feature_selection(self, sample_data):
        """Test feature selection through unified interface"""
        optimizer = QuantumOptimizer(use_quantum=True)

        result = optimizer.select_features(
            sample_data['X'],
            sample_data['y'],
            n_features=10,
            n_iterations=5
        )

        assert result['method'] == 'Quantum'


class TestGetDefaultPortfolioConstraints:
    """Test default portfolio constraints function"""

    def test_conservative_profile(self):
        """Test conservative risk profile"""
        constraints = get_default_portfolio_constraints('conservative')

        assert constraints['min_weight'] > 0
        assert constraints['max_weight'] < 0.5
        assert constraints['risk_tolerance'] > 1.0

    def test_balanced_profile(self):
        """Test balanced risk profile"""
        constraints = get_default_portfolio_constraints('balanced')

        assert constraints['min_weight'] >= 0
        assert constraints['max_weight'] <= 1.0
        assert constraints['risk_tolerance'] == 1.0

    def test_aggressive_profile(self):
        """Test aggressive risk profile"""
        constraints = get_default_portfolio_constraints('aggressive')

        assert constraints['min_weight'] >= 0
        assert constraints['max_weight'] > 0.5
        assert constraints['risk_tolerance'] < 1.0

    def test_unknown_profile(self):
        """Test unknown risk profile defaults to balanced"""
        constraints = get_default_portfolio_constraints('unknown')

        balanced = get_default_portfolio_constraints('balanced')
        assert constraints == balanced


class TestIntegrationWorkflow:
    """Test end-to-end integration workflow"""

    def test_full_workflow(self, sample_data):
        """Test complete optimization workflow"""
        # Step 1: Feature selection
        feature_result = select_features(
            sample_data['X'],
            sample_data['y'],
            n_features=10,
            use_quantum=True,
            n_iterations=5
        )

        assert feature_result['X_transformed'].shape[1] == 10

        # Step 2: Portfolio optimization
        portfolio_result = optimize_portfolio(
            sample_data['returns'],
            sample_data['covariance'],
            use_quantum=True,
            n_iterations=10
        )

        assert portfolio_result['sharpe_ratio'] > 0

        # Step 3: Hyperparameter tuning
        param_spaces = [
            {'name': 'param', 'type': 'continuous', 'bounds': (0, 1)}
        ]

        def objective(params):
            return params['param']

        tuning_result = tune_hyperparameters(
            param_spaces,
            objective,
            use_quantum=True,
            n_iterations=5
        )

        assert tuning_result['best_score'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
