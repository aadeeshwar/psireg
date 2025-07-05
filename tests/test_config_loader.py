"""Tests for config loader functionality.

This module tests the configuration loading system that uses Pydantic models
to read YAML configuration files for PSIREG runtime settings.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from psireg.config.loaders import ConfigLoader, YamlConfigLoader
from psireg.config.schema import (
    DatabaseConfig,
    GridConfig,
    LoggingConfig,
    PSIREGConfig,
    RLConfig,
    SimulationConfig,
    SwarmConfig,
)
from pydantic import ValidationError


class TestConfigSchema:
    """Test Pydantic configuration schema models."""

    def test_simulation_config_creation(self):
        """Test creation of simulation configuration."""
        config = SimulationConfig(
            timestep_minutes=15,
            horizon_hours=24,
            mode="REAL_TIME",
            max_assets=100,
            enable_weather=True,
            weather_update_interval=60,
        )
        assert config.timestep_minutes == 15
        assert config.horizon_hours == 24
        assert config.mode == "REAL_TIME"
        assert config.max_assets == 100
        assert config.enable_weather is True
        assert config.weather_update_interval == 60

    def test_simulation_config_validation(self):
        """Test validation of simulation configuration."""
        # Test negative timestep
        with pytest.raises(ValidationError):
            SimulationConfig(timestep_minutes=-1)

        # Test invalid mode
        with pytest.raises(ValidationError):
            SimulationConfig(mode="INVALID_MODE")

        # Test zero horizon
        with pytest.raises(ValidationError):
            SimulationConfig(horizon_hours=0)

    def test_rl_config_creation(self):
        """Test creation of RL configuration."""
        config = RLConfig(
            learning_rate=0.001,
            gamma=0.95,
            epsilon=0.1,
            batch_size=32,
            memory_size=10000,
            model_path="models/rl_model.pkl",
            training_episodes=1000,
            prediction_horizon=24,
        )
        assert config.learning_rate == 0.001
        assert config.gamma == 0.95
        assert config.epsilon == 0.1
        assert config.batch_size == 32
        assert config.memory_size == 10000
        assert config.model_path == "models/rl_model.pkl"
        assert config.training_episodes == 1000
        assert config.prediction_horizon == 24

    def test_rl_config_validation(self):
        """Test validation of RL configuration."""
        # Test negative learning rate
        with pytest.raises(ValidationError):
            RLConfig(learning_rate=-0.001)

        # Test gamma out of range
        with pytest.raises(ValidationError):
            RLConfig(gamma=1.5)

        # Test negative batch size
        with pytest.raises(ValidationError):
            RLConfig(batch_size=-1)

    def test_swarm_config_creation(self):
        """Test creation of swarm configuration."""
        config = SwarmConfig(
            num_agents=10,
            pheromone_decay=0.95,
            response_time_seconds=1.0,
            communication_range=5.0,
            max_iterations=100,
            convergence_threshold=0.01,
        )
        assert config.num_agents == 10
        assert config.pheromone_decay == 0.95
        assert config.response_time_seconds == 1.0
        assert config.communication_range == 5.0
        assert config.max_iterations == 100
        assert config.convergence_threshold == 0.01

    def test_swarm_config_validation(self):
        """Test validation of swarm configuration."""
        # Test negative num_agents
        with pytest.raises(ValidationError):
            SwarmConfig(num_agents=-1)

        # Test pheromone_decay out of range
        with pytest.raises(ValidationError):
            SwarmConfig(pheromone_decay=1.5)

        # Test negative response_time
        with pytest.raises(ValidationError):
            SwarmConfig(response_time_seconds=-1.0)

    def test_grid_config_creation(self):
        """Test creation of grid configuration."""
        config = GridConfig(
            frequency_hz=60.0,
            voltage_kv=230.0,
            stability_threshold=0.1,
            max_power_mw=1000.0,
            enable_monitoring=True,
            alert_threshold=0.8,
        )
        assert config.frequency_hz == 60.0
        assert config.voltage_kv == 230.0
        assert config.stability_threshold == 0.1
        assert config.max_power_mw == 1000.0
        assert config.enable_monitoring is True
        assert config.alert_threshold == 0.8

    def test_grid_config_validation(self):
        """Test validation of grid configuration."""
        # Test negative frequency
        with pytest.raises(ValidationError):
            GridConfig(frequency_hz=-60.0)

        # Test negative voltage
        with pytest.raises(ValidationError):
            GridConfig(voltage_kv=-230.0)

        # Test negative max_power
        with pytest.raises(ValidationError):
            GridConfig(max_power_mw=-1000.0)

    def test_logging_config_creation(self):
        """Test creation of logging configuration."""
        config = LoggingConfig(
            level="INFO",
            file_path="logs/psireg.log",
            max_file_size_mb=10,
            backup_count=5,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            enable_console=True,
        )
        assert config.level == "INFO"
        assert config.file_path == "logs/psireg.log"
        assert config.max_file_size_mb == 10
        assert config.backup_count == 5
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.enable_console is True

    def test_logging_config_validation(self):
        """Test validation of logging configuration."""
        # Test invalid level
        with pytest.raises(ValidationError):
            LoggingConfig(level="INVALID")

        # Test negative file size
        with pytest.raises(ValidationError):
            LoggingConfig(max_file_size_mb=-1)

        # Test negative backup count
        with pytest.raises(ValidationError):
            LoggingConfig(backup_count=-1)

    def test_database_config_creation(self):
        """Test creation of database configuration."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="psireg",
            username="user",
            password="password",
            pool_size=10,
            max_overflow=20,
            echo=False,
        )
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "psireg"
        assert config.username == "user"
        assert config.password == "password"
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.echo is False

    def test_database_config_validation(self):
        """Test validation of database configuration."""
        # Test negative port
        with pytest.raises(ValidationError):
            DatabaseConfig(port=-1)

        # Test invalid port range
        with pytest.raises(ValidationError):
            DatabaseConfig(port=70000)

        # Test negative pool size
        with pytest.raises(ValidationError):
            DatabaseConfig(pool_size=-1)

    def test_psireg_config_creation(self):
        """Test creation of main PSIREG configuration."""
        config = PSIREGConfig(
            version="0.1.0",
            environment="development",
            simulation=SimulationConfig(),
            rl=RLConfig(),
            swarm=SwarmConfig(),
            grid=GridConfig(),
            logging=LoggingConfig(),
            database=DatabaseConfig(),
        )
        assert config.version == "0.1.0"
        assert config.environment == "development"
        assert isinstance(config.simulation, SimulationConfig)
        assert isinstance(config.rl, RLConfig)
        assert isinstance(config.swarm, SwarmConfig)
        assert isinstance(config.grid, GridConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.database, DatabaseConfig)

    def test_psireg_config_validation(self):
        """Test validation of main PSIREG configuration."""
        # Test invalid environment
        with pytest.raises(ValidationError):
            PSIREGConfig(environment="invalid")


class TestYamlConfigLoader:
    """Test YAML configuration loader functionality."""

    def test_yaml_loader_creation(self):
        """Test creation of YAML configuration loader."""
        loader = YamlConfigLoader()
        assert loader is not None
        assert hasattr(loader, "load_config")
        assert hasattr(loader, "load_config_from_dict")

    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "version": "0.1.0",
            "environment": "testing",
            "simulation": {
                "timestep_minutes": 15,
                "horizon_hours": 24,
                "mode": "BATCH",
                "max_assets": 50,
                "enable_weather": True,
                "weather_update_interval": 60,
            },
            "rl": {
                "learning_rate": 0.001,
                "gamma": 0.95,
                "epsilon": 0.1,
                "batch_size": 32,
                "memory_size": 10000,
                "model_path": "models/test_model.pkl",
                "training_episodes": 100,
                "prediction_horizon": 24,
            },
            "swarm": {
                "num_agents": 5,
                "pheromone_decay": 0.9,
                "response_time_seconds": 0.5,
                "communication_range": 3.0,
                "max_iterations": 50,
                "convergence_threshold": 0.01,
            },
            "grid": {
                "frequency_hz": 60.0,
                "voltage_kv": 230.0,
                "stability_threshold": 0.1,
                "max_power_mw": 500.0,
                "enable_monitoring": True,
                "alert_threshold": 0.8,
            },
            "logging": {
                "level": "DEBUG",
                "file_path": "logs/test.log",
                "max_file_size_mb": 5,
                "backup_count": 3,
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "enable_console": True,
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test_psireg",
                "username": "test_user",
                "password": "test_password",
                "pool_size": 5,
                "max_overflow": 10,
                "echo": True,
            },
        }

        loader = YamlConfigLoader()
        config = loader.load_config_from_dict(config_dict)

        assert config.version == "0.1.0"
        assert config.environment == "testing"
        assert config.simulation.timestep_minutes == 15
        assert config.rl.learning_rate == 0.001
        assert config.swarm.num_agents == 5
        assert config.grid.frequency_hz == 60.0
        assert config.logging.level == "DEBUG"
        assert config.database.host == "localhost"

    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "version": "0.1.0",
            "environment": "production",
            "simulation": {
                "timestep_minutes": 30,
                "horizon_hours": 48,
                "mode": "REAL_TIME",
                "max_assets": 200,
                "enable_weather": True,
                "weather_update_interval": 120,
            },
            "rl": {
                "learning_rate": 0.0005,
                "gamma": 0.99,
                "epsilon": 0.05,
                "batch_size": 64,
                "memory_size": 50000,
                "model_path": "models/production_model.pkl",
                "training_episodes": 5000,
                "prediction_horizon": 48,
            },
            "swarm": {
                "num_agents": 20,
                "pheromone_decay": 0.95,
                "response_time_seconds": 1.0,
                "communication_range": 10.0,
                "max_iterations": 200,
                "convergence_threshold": 0.005,
            },
            "grid": {
                "frequency_hz": 60.0,
                "voltage_kv": 230.0,
                "stability_threshold": 0.05,
                "max_power_mw": 2000.0,
                "enable_monitoring": True,
                "alert_threshold": 0.9,
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/production.log",
                "max_file_size_mb": 50,
                "backup_count": 10,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "enable_console": False,
            },
            "database": {
                "host": "prod-db.example.com",
                "port": 5432,
                "database": "psireg_prod",
                "username": "psireg_user",
                "password": "secure_password",
                "pool_size": 20,
                "max_overflow": 40,
                "echo": False,
            },
        }

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            loader = YamlConfigLoader()
            config = loader.load_config(temp_file)

            assert config.version == "0.1.0"
            assert config.environment == "production"
            assert config.simulation.timestep_minutes == 30
            assert config.rl.learning_rate == 0.0005
            assert config.swarm.num_agents == 20
            assert config.grid.frequency_hz == 60.0
            assert config.logging.level == "INFO"
            assert config.database.host == "prod-db.example.com"
        finally:
            Path(temp_file).unlink()

    def test_load_config_file_not_found(self):
        """Test loading configuration from non-existent file."""
        loader = YamlConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_config("non_existent_file.yaml")

    def test_load_config_invalid_yaml(self):
        """Test loading configuration from invalid YAML file."""
        # Create temporary invalid YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = f.name

        try:
            loader = YamlConfigLoader()
            with pytest.raises(yaml.YAMLError):
                loader.load_config(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_load_config_validation_error(self):
        """Test loading configuration with validation errors."""
        config_data = {
            "version": "0.1.0",
            "environment": "testing",
            "simulation": {
                "timestep_minutes": -15,  # Invalid: negative value
                "horizon_hours": 24,
                "mode": "BATCH",
                "max_assets": 50,
                "enable_weather": True,
                "weather_update_interval": 60,
            },
            "rl": {
                "learning_rate": 0.001,
                "gamma": 0.95,
                "epsilon": 0.1,
                "batch_size": 32,
                "memory_size": 10000,
                "model_path": "models/test_model.pkl",
                "training_episodes": 100,
                "prediction_horizon": 24,
            },
            "swarm": {
                "num_agents": 5,
                "pheromone_decay": 0.9,
                "response_time_seconds": 0.5,
                "communication_range": 3.0,
                "max_iterations": 50,
                "convergence_threshold": 0.01,
            },
            "grid": {
                "frequency_hz": 60.0,
                "voltage_kv": 230.0,
                "stability_threshold": 0.1,
                "max_power_mw": 500.0,
                "enable_monitoring": True,
                "alert_threshold": 0.8,
            },
            "logging": {
                "level": "DEBUG",
                "file_path": "logs/test.log",
                "max_file_size_mb": 5,
                "backup_count": 3,
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "enable_console": True,
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test_psireg",
                "username": "test_user",
                "password": "test_password",
                "pool_size": 5,
                "max_overflow": 10,
                "echo": True,
            },
        }

        loader = YamlConfigLoader()
        with pytest.raises(ValidationError):
            loader.load_config_from_dict(config_data)


class TestConfigLoader:
    """Test generic configuration loader functionality."""

    def test_config_loader_interface(self):
        """Test that ConfigLoader defines proper interface."""
        # Test that it's an abstract base class
        with pytest.raises(TypeError):
            ConfigLoader()

    def test_yaml_loader_implements_interface(self):
        """Test that YamlConfigLoader properly implements ConfigLoader interface."""
        loader = YamlConfigLoader()
        assert isinstance(loader, ConfigLoader)
        assert hasattr(loader, "load_config")
        assert hasattr(loader, "load_config_from_dict")


class TestConfigIntegration:
    """Test integration scenarios for configuration loading."""

    def test_minimal_valid_config(self):
        """Test loading minimal valid configuration."""
        minimal_config = {
            "version": "0.1.0",
            "environment": "development",
            "simulation": {},
            "rl": {},
            "swarm": {},
            "grid": {},
            "logging": {},
            "database": {},
        }

        loader = YamlConfigLoader()
        config = loader.load_config_from_dict(minimal_config)

        # Should use default values
        assert config.version == "0.1.0"
        assert config.environment == "development"
        assert config.simulation.timestep_minutes == 15  # Default value
        assert config.rl.learning_rate == 0.001  # Default value
        assert config.swarm.num_agents == 10  # Default value

    def test_partial_config_override(self):
        """Test loading configuration with partial overrides."""
        partial_config = {
            "version": "0.1.0",
            "environment": "staging",
            "simulation": {
                "timestep_minutes": 5,  # Override default
                "mode": "HISTORICAL",
            },
            "rl": {
                "learning_rate": 0.01,  # Override default
            },
            "swarm": {},  # Use all defaults
            "grid": {},
            "logging": {},
            "database": {},
        }

        loader = YamlConfigLoader()
        config = loader.load_config_from_dict(partial_config)

        # Check overrides
        assert config.simulation.timestep_minutes == 5
        assert config.simulation.mode == "HISTORICAL"
        assert config.rl.learning_rate == 0.01

        # Check defaults are preserved
        assert config.simulation.horizon_hours == 24  # Default
        assert config.rl.gamma == 0.95  # Default
        assert config.swarm.num_agents == 10  # Default

    def test_environment_specific_config(self):
        """Test loading configuration for different environments."""
        environments = ["development", "testing", "staging", "production"]

        for env in environments:
            config_dict = {
                "version": "0.1.0",
                "environment": env,
                "simulation": {},
                "rl": {},
                "swarm": {},
                "grid": {},
                "logging": {},
                "database": {},
            }

            loader = YamlConfigLoader()
            config = loader.load_config_from_dict(config_dict)
            assert config.environment == env

    @pytest.mark.slow
    def test_large_config_loading(self):
        """Test loading large configuration files."""
        large_config = {
            "version": "0.1.0",
            "environment": "production",
            "simulation": {
                "timestep_minutes": 1,
                "horizon_hours": 168,  # 1 week
                "mode": "REAL_TIME",
                "max_assets": 10000,
                "enable_weather": True,
                "weather_update_interval": 300,
            },
            "rl": {
                "learning_rate": 0.0001,
                "gamma": 0.999,
                "epsilon": 0.01,
                "batch_size": 1024,
                "memory_size": 1000000,
                "model_path": "models/large_production_model.pkl",
                "training_episodes": 100000,
                "prediction_horizon": 168,
            },
            "swarm": {
                "num_agents": 1000,
                "pheromone_decay": 0.99,
                "response_time_seconds": 0.1,
                "communication_range": 50.0,
                "max_iterations": 10000,
                "convergence_threshold": 0.0001,
            },
            "grid": {
                "frequency_hz": 60.0,
                "voltage_kv": 500.0,
                "stability_threshold": 0.01,
                "max_power_mw": 10000.0,
                "enable_monitoring": True,
                "alert_threshold": 0.95,
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/large_production.log",
                "max_file_size_mb": 1000,
                "backup_count": 100,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "enable_console": False,
            },
            "database": {
                "host": "cluster.production.example.com",
                "port": 5432,
                "database": "psireg_large_prod",
                "username": "psireg_prod_user",
                "password": "very_secure_password_123",
                "pool_size": 100,
                "max_overflow": 200,
                "echo": False,
            },
        }

        loader = YamlConfigLoader()
        config = loader.load_config_from_dict(large_config)

        assert config.simulation.max_assets == 10000
        assert config.rl.memory_size == 1000000
        assert config.swarm.num_agents == 1000
        assert config.database.pool_size == 100
