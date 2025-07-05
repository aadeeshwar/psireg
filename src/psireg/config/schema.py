"""Configuration schema models for PSIREG renewable energy grid system.

This module defines Pydantic models for configuration validation and runtime settings
for the Predictive Swarm Intelligence for Renewable Energy Grids project.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Environment(str, Enum):
    """Valid deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SimulationMode(str, Enum):
    """Valid simulation execution modes."""

    REAL_TIME = "REAL_TIME"
    HISTORICAL = "HISTORICAL"
    FORECAST = "FORECAST"
    BATCH = "BATCH"
    INTERACTIVE = "INTERACTIVE"


class LogLevel(str, Enum):
    """Valid logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SimulationConfig(BaseModel):
    """Configuration for simulation parameters."""

    timestep_minutes: int = Field(default=15, ge=1, le=60, description="Simulation timestep in minutes")
    horizon_hours: int = Field(default=24, ge=1, le=8760, description="Simulation horizon in hours")  # 1 year maximum
    mode: SimulationMode = Field(default=SimulationMode.REAL_TIME, description="Simulation execution mode")
    max_assets: int = Field(default=100, ge=1, le=100000, description="Maximum number of grid assets")
    enable_weather: bool = Field(default=True, description="Enable weather data integration")
    weather_update_interval: int = Field(
        default=60, ge=1, le=3600, description="Weather data update interval in seconds"
    )


class RLConfig(BaseModel):
    """Configuration for reinforcement learning parameters."""

    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0, description="Learning rate for RL algorithm")
    gamma: float = Field(default=0.95, ge=0.0, le=1.0, description="Discount factor for future rewards")
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0, description="Exploration rate for epsilon-greedy policy")
    batch_size: int = Field(default=32, ge=1, le=10000, description="Batch size for neural network training")
    memory_size: int = Field(default=10000, ge=1000, le=10000000, description="Experience replay buffer size")
    model_path: str = Field(default="models/rl_model.pkl", description="Path to save/load RL model")
    training_episodes: int = Field(default=1000, ge=1, le=1000000, description="Number of training episodes")
    prediction_horizon: int = Field(
        default=24, ge=1, le=168, description="Prediction horizon in hours"  # 1 week maximum
    )


class SwarmConfig(BaseModel):
    """Configuration for swarm intelligence parameters."""

    num_agents: int = Field(default=10, ge=1, le=10000, description="Number of swarm agents")
    pheromone_decay: float = Field(default=0.95, ge=0.0, le=1.0, description="Pheromone decay rate per timestep")
    response_time_seconds: float = Field(
        default=1.0, gt=0.0, le=3600.0, description="Local swarm response time in seconds"
    )
    communication_range: float = Field(default=5.0, gt=0.0, le=1000.0, description="Agent communication range")
    max_iterations: int = Field(default=100, ge=1, le=100000, description="Maximum optimization iterations")
    convergence_threshold: float = Field(
        default=0.01, gt=0.0, le=1.0, description="Convergence threshold for optimization"
    )


class GridConfig(BaseModel):
    """Configuration for electrical grid parameters."""

    frequency_hz: float = Field(default=60.0, gt=0.0, le=400.0, description="Grid frequency in Hz")
    voltage_kv: float = Field(default=230.0, gt=0.0, le=1000.0, description="Grid voltage in kV")
    stability_threshold: float = Field(default=0.1, gt=0.0, le=10.0, description="Grid stability threshold in Hz")
    max_power_mw: float = Field(default=1000.0, gt=0.0, le=100000.0, description="Maximum grid power in MW")
    enable_monitoring: bool = Field(default=True, description="Enable real-time grid monitoring")
    alert_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Alert threshold as fraction of maximum capacity"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging parameters."""

    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    file_path: str = Field(default="logs/psireg.log", description="Log file path")
    max_file_size_mb: int = Field(default=10, ge=1, le=1000, description="Maximum log file size in MB")
    backup_count: int = Field(default=5, ge=0, le=100, description="Number of backup log files to keep")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log message format"
    )
    enable_console: bool = Field(default=True, description="Enable console output")


class DatabaseConfig(BaseModel):
    """Configuration for database connection parameters."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(default="psireg", description="Database name")
    username: str = Field(default="psireg_user", description="Database username")
    password: str = Field(default="psireg_password", description="Database password")
    pool_size: int = Field(default=10, ge=1, le=1000, description="Database connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=1000, description="Maximum overflow connections")
    echo: bool = Field(default=False, description="Enable SQL query logging")


class PSIREGConfig(BaseModel):
    """Main configuration model for PSIREG system."""

    version: str = Field(default="0.1.0", description="PSIREG version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Deployment environment")
    simulation: SimulationConfig = Field(default_factory=SimulationConfig, description="Simulation configuration")
    rl: RLConfig = Field(default_factory=RLConfig, description="Reinforcement learning configuration")
    swarm: SwarmConfig = Field(default_factory=SwarmConfig, description="Swarm intelligence configuration")
    grid: GridConfig = Field(default_factory=GridConfig, description="Grid system configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format."""
        if not v or not isinstance(v, str):
            raise ValueError("Version must be a non-empty string")
        # Basic semantic version validation
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError("Version must be in format X.Y.Z")
        try:
            for part in parts:
                int(part)
        except ValueError:
            raise ValueError("Version parts must be integers") from None
        return v

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid",  # Don't allow extra fields
        json_schema_extra={
            "example": {
                "version": "0.1.0",
                "environment": "development",
                "simulation": {
                    "timestep_minutes": 15,
                    "horizon_hours": 24,
                    "mode": "REAL_TIME",
                    "max_assets": 100,
                    "enable_weather": True,
                    "weather_update_interval": 60,
                },
                "rl": {
                    "learning_rate": 0.001,
                    "gamma": 0.95,
                    "epsilon": 0.1,
                    "batch_size": 32,
                    "memory_size": 10000,
                    "model_path": "models/rl_model.pkl",
                    "training_episodes": 1000,
                    "prediction_horizon": 24,
                },
                "swarm": {
                    "num_agents": 10,
                    "pheromone_decay": 0.95,
                    "response_time_seconds": 1.0,
                    "communication_range": 5.0,
                    "max_iterations": 100,
                    "convergence_threshold": 0.01,
                },
                "grid": {
                    "frequency_hz": 60.0,
                    "voltage_kv": 230.0,
                    "stability_threshold": 0.1,
                    "max_power_mw": 1000.0,
                    "enable_monitoring": True,
                    "alert_threshold": 0.8,
                },
                "logging": {
                    "level": "INFO",
                    "file_path": "logs/psireg.log",
                    "max_file_size_mb": 10,
                    "backup_count": 5,
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "enable_console": True,
                },
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "psireg",
                    "username": "psireg_user",
                    "password": "psireg_password",
                    "pool_size": 10,
                    "max_overflow": 20,
                    "echo": False,
                },
            }
        },
    )
