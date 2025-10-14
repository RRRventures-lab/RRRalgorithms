from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional
import os


"""
Configuration loader for research testing framework.

Loads API keys and database credentials from config/api-keys/.env
"""



class Config:
    """Configuration manager for hypothesis testing framework."""

    def __init__(self, env_file: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            env_file: Path to .env file (default: config/api-keys/.env)
        """
        if env_file is None:
            # Default to project root config
            project_root = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms")
            env_file = project_root / "config" / "api-keys" / ".env"

        # Load environment variables
        if env_file.exists():
            load_dotenv(env_file)
            print(f"[Config] Loaded configuration from {env_file}")
        else:
            print(f"[Config] Warning: .env file not found at {env_file}")

        # Market Data APIs
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        self.polygon_rate_limit = int(os.getenv("POLYGON_RATE_LIMIT", "100"))

        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.cache_ttl = int(os.getenv("CACHE_TTL", "300"))

        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # Database Configuration
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
        self.supabase_db_url = os.getenv("SUPABASE_DB_URL")
        self.database_url = os.getenv("DATABASE_URL", self.supabase_db_url)

        # Redis
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # Exchange APIs
        self.coinbase_api_key = os.getenv("COINBASE_API_KEY")
        self.coinbase_api_secret = os.getenv("COINBASE_API_SECRET")
        self.binance_api_key = os.getenv("BINANCE_API_KEY")
        self.binance_api_secret = os.getenv("BINANCE_API_SECRET")

        # Application Configuration
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.project_root = Path(os.getenv("PROJECT_ROOT", "/Volumes/Lexar/RRRVentures/RRRalgorithms"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"

        # Risk Management
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "0.20"))
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "0.05"))
        self.max_portfolio_volatility = float(os.getenv("MAX_PORTFOLIO_VOLATILITY", "0.25"))

        # Feature Flags
        self.paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.live_trading = os.getenv("LIVE_TRADING", "false").lower() == "true"
        self.enable_sentiment = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"
        self.enable_quantum = os.getenv("ENABLE_QUANTUM_OPTIMIZATION", "false").lower() == "true"
        self.enable_multi_agent = os.getenv("ENABLE_MULTI_AGENT", "true").lower() == "true"

    def validate(self) -> Dict[str, bool]:
        """
        Validate configuration.

        Returns:
            Dict of component: availability
        """
        validation = {
            "polygon": bool(self.polygon_api_key),
            "perplexity": bool(self.perplexity_api_key),
            "database": bool(self.database_url),
            "redis": bool(self.redis_url),
            "coinbase": bool(self.coinbase_api_key and self.coinbase_api_secret),
        }

        return validation

    @lru_cache(maxsize=128)

    def get_local_db_path(self) -> Path:
        """Get path for local SQLite database."""
        return self.project_root / "research" / "data" / "hypothesis_testing.db"

    def __repr__(self) -> str:
        """String representation."""
        validation = self.validate()
        available = [k for k, v in validation.items() if v]
        return f"Config(environment={self.environment}, available={available})"


# Global config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    print("=" * 70)
    print("Configuration Status")
    print("=" * 70)

    print(f"\nEnvironment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"Project Root: {config.project_root}")

    print("\nAPI Availability:")
    validation = config.validate()
    for component, available in validation.items():
        status = "✅ Available" if available else "❌ Not configured"
        print(f"  {component}: {status}")

    print("\nDatabase:")
    print(f"  Supabase URL: {config.supabase_url}")
    print(f"  Local DB: {config.get_local_db_path()}")

    print("\nRisk Parameters:")
    print(f"  Max Position Size: {config.max_position_size:.0%}")
    print(f"  Max Daily Loss: {config.max_daily_loss:.0%}")
    print(f"  Max Portfolio Volatility: {config.max_portfolio_volatility:.0%}")

    print("\nFeature Flags:")
    print(f"  Paper Trading: {config.paper_trading}")
    print(f"  Live Trading: {config.live_trading}")
    print(f"  Sentiment Analysis: {config.enable_sentiment}")
    print(f"  Multi-Agent: {config.enable_multi_agent}")

    print("=" * 70)
