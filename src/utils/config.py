"""Configuration loader and management utilities."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration class for loading and accessing YAML config."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'dataset.root_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Access config sections directly."""
        return self._config[key]
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


# Global config instance
_config = None


def get_config(config_path: str = "configs/config.yaml") -> Config:
    """
    Get global configuration instance (singleton pattern).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: str = "configs/config.yaml") -> Config:
    """
    Force reload configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        New Config instance
    """
    global _config
    _config = Config(config_path)
    return _config

