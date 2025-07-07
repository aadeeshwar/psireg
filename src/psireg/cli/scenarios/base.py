"""Base scenario class for PSIREG renewable energy grid scenarios.

This module provides the base scenario class that defines the common interface
and basic functionality for all renewable energy grid simulation scenarios.
"""

from abc import ABC, abstractmethod
from typing import Any

from psireg.utils.enums import WeatherCondition


class BaseScenario(ABC):
    """Base class for renewable energy grid scenarios.

    This abstract base class defines the interface that all scenarios must
    implement for consistent scenario orchestration and execution.
    """

    def __init__(self, name: str, description: str, duration_hours: int = 24):
        """Initialize base scenario.

        Args:
            name: Scenario name
            description: Scenario description
            duration_hours: Scenario duration in hours
        """
        self.name = name
        self.description = description
        self.duration_hours = duration_hours
        self.features = []

    @abstractmethod
    def get_weather_conditions(self) -> dict[str, Any]:
        """Get weather conditions for the scenario.

        Returns:
            Weather conditions dictionary
        """
        pass

    @abstractmethod
    def get_grid_events(self) -> list[dict[str, Any]]:
        """Get grid events for the scenario.

        Returns:
            List of grid event dictionaries
        """
        pass

    @abstractmethod
    def get_asset_configuration(self) -> dict[str, Any]:
        """Get asset configuration for the scenario.

        Returns:
            Asset configuration dictionary
        """
        pass

    def get_weather_timeline(self) -> list[dict[str, Any]]:
        """Get weather progression timeline for the scenario.

        Returns:
            List of hourly weather conditions
        """
        # Default implementation returns constant weather
        base_weather = self.get_weather_conditions()
        timeline = []

        for hour in range(self.duration_hours):
            weather = {"hour": hour, **base_weather}
            timeline.append(weather)

        return timeline

    def get_emergency_response_config(self) -> dict[str, Any]:
        """Get emergency response configuration.

        Returns:
            Emergency response configuration dictionary
        """
        return {
            "enable_emergency_mode": False,
            "response_time_seconds": 60.0,
            "activation_triggers": {"frequency_deviation": 0.5, "voltage_deviation": 0.1, "transmission_outage": True},
            "swarm_coordination": {"enhanced_communication": False, "response_time_seconds": 10.0},
            "priority_assets": {"critical_loads": [], "backup_generation": [], "energy_storage": []},
            "recovery_procedures": {"automatic_restart": True, "restart_delay_minutes": 5},
        }

    def get_metrics_configuration(self) -> dict[str, Any]:
        """Get metrics collection configuration.

        Returns:
            Metrics configuration dictionary
        """
        return {
            "enable_emergency_metrics": False,
            "frequency_monitoring": {"enable": True, "threshold_hz": 0.2},
            "voltage_monitoring": {"enable": True, "threshold_kv": 10.0},
            "resilience_metrics": {
                "system_recovery_time": True,
                "load_served_percentage": True,
                "asset_availability": True,
            },
            "emergency_performance": {
                "response_time_metrics": False,
                "coordination_effectiveness": False,
                "frequency_stability": True,
            },
            "weather_impact": {"renewable_curtailment": True, "weather_correlation": True, "forecast_accuracy": False},
            "economic_impact": {
                "lost_load_cost": False,
                "emergency_generation_cost": False,
                "infrastructure_damage_cost": False,
            },
        }

    @abstractmethod
    def apply_weather_conditions(self, weather_config: dict[str, Any]) -> None:
        """Apply weather conditions to the scenario.

        Args:
            weather_config: Weather configuration to apply
        """
        pass

    @abstractmethod
    def activate_emergency_response(self) -> None:
        """Activate emergency response procedures."""
        pass

    @abstractmethod
    def setup_metrics_collection(self) -> None:
        """Setup metrics collection for the scenario."""
        pass


class DefaultScenario(BaseScenario):
    """Default scenario implementation for normal grid operations."""

    def __init__(self):
        """Initialize default scenario."""
        super().__init__(name="default", description="Default scenario with normal grid operations", duration_hours=24)
        self.features = ["normal_operations", "basic_metrics"]

    def get_weather_conditions(self) -> dict[str, Any]:
        """Get default weather conditions."""
        return {
            "condition": WeatherCondition.CLEAR,
            "wind_speed_ms": 8.0,
            "temperature_c": 22.0,
            "irradiance_w_m2": 700.0,
            "humidity_percent": 60.0,
            "pressure_hpa": 1013.0,
            "visibility_km": 15.0,
            "air_density_kg_m3": 1.225,
        }

    def get_grid_events(self) -> list[dict[str, Any]]:
        """Get default grid events (minimal)."""
        return [
            {"type": "load_variation", "hour": 8, "magnitude": 1.2, "description": "Morning peak load"},
            {"type": "load_variation", "hour": 18, "magnitude": 1.3, "description": "Evening peak load"},
        ]

    def get_asset_configuration(self) -> dict[str, Any]:
        """Get default asset configuration."""
        return {
            "renewable": {
                "wind_farms": {"count": 2, "capacity_mw": 50.0, "rotor_diameter_m": 120.0, "hub_height_m": 100.0},
                "solar_farms": {
                    "count": 3,
                    "capacity_mw": 30.0,
                    "efficiency": 0.20,
                    "area_m2": 150000.0,
                    "reduced_output": False,
                },
            },
            "storage": {
                "batteries": {
                    "count": 1,
                    "capacity_mw": 50.0,
                    "energy_capacity_mwh": 200.0,
                    "initial_soc": 50.0,
                    "emergency_reserve": 0.1,
                    "priority_discharge": False,
                }
            },
            "thermal": {
                "natural_gas": {
                    "count": 1,
                    "capacity_mw": 150.0,
                    "min_output_mw": 30.0,
                    "ramp_rate": 8.0,
                    "emergency_ramping": False,
                    "max_ramp_rate": 8.0,
                },
                "coal": {"count": 0},
            },
            "load": {
                "demand_response": {
                    "count": 3,
                    "capacity_mw": 100.0,
                    "baseline_mw": 80.0,
                    "dr_capability_mw": 20.0,
                    "emergency_load_shedding": False,
                    "max_shed_percentage": 0.0,
                }
            },
        }

    def apply_weather_conditions(self, weather_config: dict[str, Any]) -> None:
        """Apply weather conditions to the scenario.

        Args:
            weather_config: Weather configuration to apply
        """
        # Default implementation - no special weather handling
        pass

    def activate_emergency_response(self) -> None:
        """Activate emergency response procedures.

        Default implementation - no emergency response
        """
        pass

    def setup_metrics_collection(self) -> None:
        """Setup metrics collection for the scenario.

        Default implementation - standard metrics only
        """
        pass
