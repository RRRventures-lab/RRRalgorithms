from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml


"""
Configuration loader for RRRalgorithms.
Automatically loads the correct configuration based on environment.
Local-first: defaults to local.yml unless ENVIRONMENT=production.
"""



class ConfigLoader:
    """Smart configuration loader that defaults to local development."""
    
    def __init__(self, config_dir: Optional[Path] = None, env_file: Optional[Path] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Path to config directory (defaults to project root/config)
            env_file: Path to .env file (defaults to .env.local in project root)
        """
        # Determine project root
        self.project_root = Path(__file__).parent.parent.parent.parent
        
        # Set config directory
        self.config_dir = config_dir or self.project_root / "config"
        
        # Load environment variables
        env_file = env_file or self.project_root / ".env.local"
        if env_file.exists():
            load_dotenv(env_file)
        
        # Determine environment (defaults to local)
        self.environment = os.getenv("ENVIRONMENT", "local").lower()
        
        # Load appropriate config
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file based on environment."""
        config_file = self.config_dir / f"{self.environment}.yml"
        
        if not config_file.exists():
            # Fallback to local if specified config doesn't exist
            config_file = self.config_dir / "local.yml"
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {config_file}. "
                    "Please ensure config/local.yml exists."
                )
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expand environment variables in config
        config = self._expand_env_vars(config)
        
        return config
    
    def _expand_env_vars(self, config: Any) -> Any:
        """Recursively expand ${VAR} style environment variables in config."""
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        else:
            return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'database.type')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config = ConfigLoader()
            >>> db_type = config.get('database.type', 'sqlite')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @lru_cache(maxsize=128)
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.get('database', {})
    
    @lru_cache(maxsize=128)
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.get('trading', {})
    
    @lru_cache(maxsize=128)
    
    def get_neural_network_config(self) -> Dict[str, Any]:
        """Get neural network configuration."""
        return self.get('neural_network', {})
    
    @lru_cache(maxsize=128)
    
    def get_data_pipeline_config(self) -> Dict[str, Any]:
        """Get data pipeline configuration."""
        return self.get('data_pipeline', {})
    
    @lru_cache(maxsize=128)
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.get('monitoring', {})
    
    def is_local(self) -> bool:
        """Check if running in local development mode."""
        return self.environment == "local"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get('debug', False)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __repr__(self) -> str:
        return f"ConfigLoader(environment={self.environment})"


# Global config instance (singleton pattern)
_global_config: Optional[ConfigLoader] = None


@lru_cache(maxsize=128)


def get_config(reload: bool = False) -> ConfigLoader:
    """
    Get the global configuration instance.
    
    Args:
        reload: Force reload of configuration
    
    Returns:
        ConfigLoader instance
    """
    global _global_config
    
    if _global_config is None or reload:
        _global_config = ConfigLoader()
    
    return _global_config


# Convenience function for quick access
def config_get(key_path: str, default: Any = None) -> Any:
    """
    Quick access to configuration values.
    
    Example:
        >>> from src.core.config.loader import config_get
        >>> db_type = config_get('database.type', 'sqlite')
    """
    return get_config().get(key_path, default)


if __name__ == "__main__":
    # Test the config loader
    config = ConfigLoader()
    print(f"Environment: {config.environment}")
    print(f"Is Local: {config.is_local()}")
    print(f"Database Type: {config.get('database.type')}")
    print(f"Trading Mode: {config.get('trading.mode')}")
    print(f"Neural Network Mode: {config.get('neural_network.mode')}")

