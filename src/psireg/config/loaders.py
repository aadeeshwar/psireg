"""Configuration loaders for PSIREG renewable energy grid system.

This module provides configuration loading functionality using YAML files
with Pydantic validation for runtime settings.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import PSIREGConfig


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders."""

    @abstractmethod
    def load_config(self, config_path: str | Path) -> PSIREGConfig:
        """Load configuration from a file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Loaded and validated configuration

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValidationError: If the config validation fails
            Exception: For other loading errors
        """
        pass

    @abstractmethod
    def load_config_from_dict(self, config_dict: dict[str, Any]) -> PSIREGConfig:
        """Load configuration from a dictionary.

        Args:
            config_dict: Configuration data as dictionary

        Returns:
            Loaded and validated configuration

        Raises:
            ValidationError: If the config validation fails
        """
        pass


class YamlConfigLoader(ConfigLoader):
    """YAML configuration loader implementation."""

    def __init__(self, safe_load: bool = True):
        """Initialize YAML configuration loader.

        Args:
            safe_load: Whether to use safe YAML loading (default: True)
        """
        self.safe_load = safe_load

    def load_config(self, config_path: str | Path) -> PSIREGConfig:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Loaded and validated configuration

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If the config validation fails
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if not config_path.is_file():
            raise ValueError(f"Configuration path is not a file: {config_path}")

        try:
            with open(config_path, encoding="utf-8") as f:
                if self.safe_load:
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = yaml.load(f, Loader=yaml.FullLoader)

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {config_path}: {e}") from e
        except Exception as e:
            raise Exception(f"Failed to read configuration file {config_path}: {e}") from e

        if config_dict is None:
            raise ValueError(f"Configuration file {config_path} is empty")

        if not isinstance(config_dict, dict):
            raise ValueError(f"Configuration file {config_path} must contain a YAML mapping")

        return self.load_config_from_dict(config_dict)

    def load_config_from_dict(self, config_dict: dict[str, Any]) -> PSIREGConfig:
        """Load configuration from a dictionary.

        Args:
            config_dict: Configuration data as dictionary

        Returns:
            Loaded and validated configuration

        Raises:
            ValidationError: If the config validation fails
        """
        if not isinstance(config_dict, dict):
            raise ValueError("Configuration data must be a dictionary")

        try:
            return PSIREGConfig(**config_dict)
        except ValidationError as e:
            # Re-raise the original ValidationError with additional context
            raise e
        except Exception as e:
            raise Exception(f"Failed to create configuration: {e}") from e

    def validate_config_structure(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize configuration structure.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Validated and normalized configuration dictionary

        Raises:
            ValueError: If configuration structure is invalid
        """
        if not isinstance(config_dict, dict):
            raise ValueError("Configuration must be a dictionary")

        # Define required top-level sections
        required_sections = {"simulation", "rl", "swarm", "grid", "logging", "database"}

        # Ensure all required sections exist (even if empty)
        for section in required_sections:
            if section not in config_dict:
                config_dict[section] = {}

        # Validate that nested sections are dictionaries
        for section in required_sections:
            if not isinstance(config_dict[section], dict):
                raise ValueError(f"Configuration section '{section}' must be a dictionary")

        return config_dict

    def merge_with_defaults(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Merge configuration with default values.

        Args:
            config_dict: User-provided configuration dictionary

        Returns:
            Configuration dictionary merged with defaults
        """
        # Create a default configuration instance to get default values
        default_config = PSIREGConfig()
        default_dict = default_config.dict()

        # Deep merge user config with defaults
        merged_dict = self._deep_merge(default_dict, config_dict)

        return merged_dict

    def _deep_merge(self, base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            update: Dictionary to merge into base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def load_config_with_env_override(self, config_path: str | Path, env_prefix: str = "PSIREG_") -> PSIREGConfig:
        """Load configuration with environment variable overrides.

        Args:
            config_path: Path to the YAML configuration file
            env_prefix: Environment variable prefix (default: "PSIREG_")

        Returns:
            Loaded configuration with environment overrides

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If the config validation fails
        """
        import os

        # Load base configuration
        config = self.load_config(config_path)
        config_dict = config.dict()

        # Apply environment variable overrides
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix) :].lower()
                # Simple override for top-level keys
                if hasattr(config, config_key):
                    # Try to convert to appropriate type
                    try:
                        if isinstance(getattr(config, config_key), bool):
                            config_dict[config_key] = value.lower() in ("true", "1", "yes", "on")
                        elif isinstance(getattr(config, config_key), int):
                            config_dict[config_key] = int(value)
                        elif isinstance(getattr(config, config_key), float):
                            config_dict[config_key] = float(value)
                        else:
                            config_dict[config_key] = value
                    except (ValueError, TypeError):
                        # If conversion fails, keep as string
                        config_dict[config_key] = value

        return PSIREGConfig(**config_dict)

    def save_config(self, config: PSIREGConfig, config_path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            config: Configuration to save
            config_path: Path to save the configuration file

        Raises:
            Exception: If saving fails
        """
        config_path = Path(config_path)

        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config.dict(), f, default_flow_style=False, sort_keys=False, indent=2)
        except Exception as e:
            raise Exception(f"Failed to save configuration to {config_path}: {e}") from e

    def generate_example_config(self, config_path: str | Path) -> None:
        """Generate an example configuration file.

        Args:
            config_path: Path to save the example configuration file
        """
        example_config = PSIREGConfig()
        self.save_config(example_config, config_path)


def create_yaml_loader(safe_load: bool = True) -> YamlConfigLoader:
    """Factory function to create a YAML configuration loader.

    Args:
        safe_load: Whether to use safe YAML loading (default: True)

    Returns:
        YamlConfigLoader instance
    """
    return YamlConfigLoader(safe_load=safe_load)


def load_config_from_yaml(config_path: str | Path) -> PSIREGConfig:
    """Convenience function to load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Loaded and validated configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValidationError: If the config validation fails
    """
    loader = YamlConfigLoader()
    return loader.load_config(config_path)


def load_config_from_dict(config_dict: dict[str, Any]) -> PSIREGConfig:
    """Convenience function to load configuration from dictionary.

    Args:
        config_dict: Configuration data as dictionary

    Returns:
        Loaded and validated configuration

    Raises:
        ValidationError: If the config validation fails
    """
    loader = YamlConfigLoader()
    return loader.load_config_from_dict(config_dict)
