from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import os

"""
Centralized Settings Management
Uses Pydantic for configuration validation and environment variable loading
"""



class Settings(BaseSettings):
    """
    Application Settings
    Loads configuration from environment variables and validates them
    """

    # ==========================================================================
    # Environment
    # ==========================================================================
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # ==========================================================================
    # Database Configuration
    # ==========================================================================
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    supabase_url: Optional[str] = Field(default=None, env="DATABASE_PATH")
    supabase_service_key: Optional[str] = Field(default=None, env="DATABASE_TYPE")
    supabase_key: Optional[str] = Field(default=None, env="SUPABASE_KEY")

    # Connection pool settings
    database_pool_min: int = Field(default=5, env="DATABASE_POOL_MIN")
    database_pool_max: int = Field(default=20, env="DATABASE_POOL_MAX")
    database_timeout: int = Field(default=30, env="DATABASE_TIMEOUT")

    # ==========================================================================
    # Redis Configuration
    # ==========================================================================
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")

    # ==========================================================================
    # API Keys
    # ==========================================================================
    polygon_api_key: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    perplexity_api_key: Optional[str] = Field(default=None, env="PERPLEXITY_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    coinbase_api_key: Optional[str] = Field(default=None, env="COINBASE_API_KEY")
    coinbase_api_secret: Optional[str] = Field(default=None, env="COINBASE_API_SECRET")
    tradingview_webhook_secret: Optional[str] = Field(
        default=None, env="TRADINGVIEW_WEBHOOK_SECRET"
    )

    # ==========================================================================
    # Trading Configuration
    # ==========================================================================
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    live_trading: bool = Field(default=False, env="LIVE_TRADING")
    initial_capital: float = Field(default=100000.0, env="INITIAL_CAPITAL")

    # ==========================================================================
    # Risk Management
    # ==========================================================================
    max_position_size_pct: float = Field(default=0.20, env="MAX_POSITION_SIZE_PCT")
    max_daily_loss_pct: float = Field(default=0.05, env="MAX_DAILY_LOSS_PCT")
    max_portfolio_volatility: float = Field(default=0.25, env="MAX_PORTFOLIO_VOLATILITY")
    max_drawdown_pct: float = Field(default=0.10, env="MAX_DRAWDOWN_PCT")

    # ==========================================================================
    # ML Model Configuration
    # ==========================================================================
    model_cache_size: int = Field(default=10, env="MODEL_CACHE_SIZE")
    model_inference_batch_size: int = Field(default=32, env="MODEL_INFERENCE_BATCH_SIZE")
    model_cache_ttl: int = Field(default=3600, env="MODEL_CACHE_TTL")  # seconds
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")

    # ==========================================================================
    # Monitoring & Observability
    # ==========================================================================
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    grafana_password: str = Field(default="admin", env="GRAFANA_PASSWORD")

    # ==========================================================================
    # Service Ports
    # ==========================================================================
    neural_network_port: int = Field(default=8000, env="NEURAL_NETWORK_PORT")
    data_pipeline_port: int = Field(default=8001, env="DATA_PIPELINE_PORT")
    trading_engine_port: int = Field(default=8002, env="TRADING_ENGINE_PORT")
    risk_management_port: int = Field(default=8003, env="RISK_MANAGEMENT_PORT")
    backtesting_port: int = Field(default=8004, env="BACKTESTING_PORT")
    api_integration_port: int = Field(default=8005, env="API_INTEGRATION_PORT")
    quantum_optimization_port: int = Field(default=8006, env="QUANTUM_OPTIMIZATION_PORT")
    monitoring_port: int = Field(default=8501, env="MONITORING_PORT")

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    enable_ai_validation: bool = Field(default=True, env="ENABLE_AI_VALIDATION")
    enable_quantum_optimization: bool = Field(default=False, env="ENABLE_QUANTUM_OPTIMIZATION")
    enable_sentiment_analysis: bool = Field(default=True, env="ENABLE_SENTIMENT_ANALYSIS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance

    Returns:
        Settings instance (cached)
    """
    return Settings()


def reload_settings():
    """Force reload settings by clearing cache"""
    get_settings.cache_clear()


def is_production() -> bool:
    """Check if running in production environment"""
    return get_settings().environment.lower() == "production"


def is_development() -> bool:
    """Check if running in development environment"""
    return get_settings().environment.lower() == "development"


def is_testing() -> bool:
    """Check if running in test environment"""
    return get_settings().environment.lower() in ("test", "testing")

