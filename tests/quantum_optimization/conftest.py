import numpy as np
import pytest

"""
Pytest configuration and shared fixtures

Author: Quantum Optimization Team
Date: 2025-10-11
"""



@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility"""
    np.random.seed(42)
    return 42


@pytest.fixture
def small_portfolio():
    """Small portfolio for quick tests"""
    n_assets = 3
    returns = np.array([0.08, 0.10, 0.12])
    cov = np.array([
        [0.010, 0.002, 0.001],
        [0.002, 0.015, 0.003],
        [0.001, 0.003, 0.020]
    ])
    return returns, cov


@pytest.fixture
def medium_portfolio():
    """Medium portfolio for standard tests"""
    np.random.seed(42)
    n_assets = 10
    returns = np.random.uniform(0.05, 0.15, n_assets)
    random_matrix = np.random.randn(n_assets, n_assets)
    cov = np.dot(random_matrix, random_matrix.T) * 0.01
    return returns, cov


@pytest.fixture
def sample_features():
    """Sample feature dataset"""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    return X, y
