"""Unit tests for ``src.core.config`` utilities."""

from pathlib import Path
import os

import pytest

from src.core.config import (
    get_project_root,
    get_env_file,
    get_config_value,
    is_development,
    is_production,
    is_paper_trading,
    is_live_trading,
)


@pytest.mark.unit
def test_get_project_root():
    """Test getting project root."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.exists()
    assert (root / "CLAUDE.md").exists()


@pytest.mark.unit
def test_get_env_file():
    """Test getting env file path."""
    env_file = get_env_file()
    assert isinstance(env_file, Path)
    assert "config/api-keys/.env" in str(env_file)


@pytest.mark.unit
def test_get_config_value(test_env):
    """Test getting config values."""
    # Test existing value
    value = get_config_value("ENVIRONMENT")
    assert value == "test"

    # Test default value
    value = get_config_value("NONEXISTENT_KEY", default="default_value")
    assert value == "default_value"

    # Test required missing key
    with pytest.raises(ValueError):
        get_config_value("NONEXISTENT_KEY", required=True)


@pytest.mark.unit
def test_is_development(test_env):
    """Test development mode check."""
    os.environ["ENVIRONMENT"] = "development"
    assert is_development() is True

    os.environ["ENVIRONMENT"] = "production"
    assert is_development() is False


@pytest.mark.unit
def test_is_production(test_env):
    """Test production mode check."""
    os.environ["ENVIRONMENT"] = "production"
    assert is_production() is True

    os.environ["ENVIRONMENT"] = "development"
    assert is_production() is False


@pytest.mark.unit
def test_is_paper_trading(test_env):
    """Test paper trading check."""
    os.environ["PAPER_TRADING"] = "true"
    assert is_paper_trading() is True

    os.environ["PAPER_TRADING"] = "false"
    assert is_paper_trading() is False


@pytest.mark.unit
def test_is_live_trading(test_env):
    """Test live trading check."""
    os.environ["LIVE_TRADING"] = "true"
    assert is_live_trading() is True

    os.environ["LIVE_TRADING"] = "false"
    assert is_live_trading() is False
