"""
Configuration loader for the ensemble experiment.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration files."""

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expand environment variables
        return self._expand_env_vars(config)

    def _expand_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively expand environment variables in config."""
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str):
            if config.startswith('${') and config.endswith('}'):
                env_var = config[2:-1]
                return os.getenv(env_var, config)
            return config
        else:
            return config

    def get_api_keys(self) -> Dict[str, Any]:
        """Load API keys configuration."""
        return self.load_yaml('api_keys.yaml')

    def get_experiment_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        return self.load_yaml('experiment_config.yaml')
