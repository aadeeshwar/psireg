"""Storm Day scenario for PSIREG renewable energy grid system.

This module implements the storm day scenario with comprehensive weather
progression, emergency response procedures, and enhanced grid resilience testing.
"""

import math
from typing import Any

from psireg.utils.enums import WeatherCondition

from .base import BaseScenario


class StormDayScenario(BaseScenario):
    """Comprehensive storm day scenario with weather progression and emergency response.

    This scenario simulates a severe weather day with:
    - Progressive storm development and intensification
    - Emergency response activation and coordination
    - Asset behavior under extreme weather conditions
    - Grid resilience testing and recovery procedures
    """

    def __init__(self):
        """Initialize storm day scenario."""
        super().__init__(
            name="storm_day",
            description=(
                "Severe weather scenario with storm conditions, emergency response "
                "activation, and grid resilience testing"
            ),
            duration_hours=16,
        )

        self.features = [
            "severe_weather",
            "emergency_response",
            "asset_coordination",
            "grid_resilience",
            "swarm_intelligence",
            "weather_progression",
            "load_shedding",
            "renewable_curtailment",
        ]

        # Storm progression configuration
        self.storm_config = {
            "start_hour": 4,
            "peak_hour": 8,
            "end_hour": 12,
            "max_wind_speed": 35.0,  # m/s
            "min_visibility": 0.5,  # km
            "max_precipitation": 50.0,  # mm/h
            "pressure_drop": 30.0,  # hPa from baseline
            "temperature_drop": 8.0,  # °C from baseline
        }

        # Emergency thresholds
        self.emergency_thresholds = {
            "wind_turbine_shutdown": 25.0,  # m/s
            "frequency_deviation": 0.3,  # Hz
            "voltage_deviation": 0.15,  # p.u.
            "transmission_capacity": 0.7,  # Reduce to 70% during storm
        }

    def get_weather_conditions(self) -> dict[str, Any]:
        """Get base weather conditions (pre-storm)."""
        return {
            "condition": WeatherCondition.CLOUDY,
            "wind_speed_ms": 12.0,
            "temperature_c": 18.0,
            "irradiance_w_m2": 400.0,
            "humidity_percent": 85.0,
            "pressure_hpa": 1005.0,
            "visibility_km": 8.0,
            "precipitation_mm_h": 2.0,
            "air_density_kg_m3": 1.240,
        }

    def get_weather_timeline(self) -> list[dict[str, Any]]:
        """Get detailed weather progression timeline for storm day.

        Returns:
            List of hourly weather conditions with storm progression
        """
        timeline = []
        baseline_conditions = self.get_weather_conditions()

        for hour in range(self.duration_hours):
            # Calculate storm intensity (0.0 to 1.0)
            storm_intensity = self._calculate_storm_intensity(hour)

            # Apply storm effects to weather conditions
            weather = self._apply_storm_effects(baseline_conditions, storm_intensity, hour)
            weather["hour"] = hour
            weather["storm_intensity"] = storm_intensity

            timeline.append(weather)

        return timeline

    def _calculate_storm_intensity(self, hour: int) -> float:
        """Calculate storm intensity for given hour.

        Args:
            hour: Current hour

        Returns:
            Storm intensity (0.0 to 1.0)
        """
        start_hour = self.storm_config["start_hour"]
        peak_hour = self.storm_config["peak_hour"]
        end_hour = self.storm_config["end_hour"]

        if hour < start_hour:
            # Pre-storm conditions
            return 0.1 + 0.1 * (hour / start_hour)
        elif hour < peak_hour:
            # Storm building
            progress = (hour - start_hour) / (peak_hour - start_hour)
            return 0.2 + 0.8 * math.sin(progress * math.pi / 2)
        elif hour < end_hour:
            # Storm weakening
            progress = (hour - peak_hour) / (end_hour - peak_hour)
            return 1.0 - 0.7 * progress
        else:
            # Post-storm recovery
            progress = min(1.0, (hour - end_hour) / 4.0)
            return 0.3 - 0.2 * progress

    def _apply_storm_effects(self, baseline: dict[str, Any], intensity: float, hour: int) -> dict[str, Any]:
        """Apply storm effects to baseline weather conditions.

        Args:
            baseline: Baseline weather conditions
            intensity: Storm intensity (0.0 to 1.0)
            hour: Current hour

        Returns:
            Modified weather conditions
        """
        weather = baseline.copy()

        # Determine weather condition based on intensity
        if intensity < 0.3:
            weather["condition"] = WeatherCondition.CLOUDY
        elif intensity < 0.6:
            weather["condition"] = WeatherCondition.RAINY
        elif intensity < 0.9:
            weather["condition"] = WeatherCondition.STORMY
        else:
            weather["condition"] = WeatherCondition.STORMY  # Severe storm

        # Apply wind effects
        base_wind = baseline["wind_speed_ms"]
        max_wind = self.storm_config["max_wind_speed"]
        weather["wind_speed_ms"] = base_wind + intensity * (max_wind - base_wind)

        # Add wind gusts during peak storm
        if intensity > 0.7:
            gust_factor = 1.0 + 0.3 * intensity * math.sin(hour * math.pi / 2)
            weather["wind_speed_ms"] *= gust_factor

        # Apply temperature effects
        temp_drop = self.storm_config["temperature_drop"] * intensity
        weather["temperature_c"] = baseline["temperature_c"] - temp_drop

        # Apply irradiance effects (solar reduction)
        base_irradiance = baseline["irradiance_w_m2"]
        weather["irradiance_w_m2"] = base_irradiance * (1.0 - 0.8 * intensity)

        # Apply pressure effects
        pressure_drop = self.storm_config["pressure_drop"] * intensity
        weather["pressure_hpa"] = baseline["pressure_hpa"] - pressure_drop

        # Apply visibility effects
        min_vis = self.storm_config["min_visibility"]
        base_vis = baseline["visibility_km"]
        weather["visibility_km"] = max(min_vis, base_vis * (1.0 - 0.9 * intensity))

        # Apply precipitation effects
        max_precip = self.storm_config["max_precipitation"]
        weather["precipitation_mm_h"] = baseline["precipitation_mm_h"] + intensity * max_precip

        # Apply humidity effects
        weather["humidity_percent"] = min(100.0, baseline["humidity_percent"] + 10 * intensity)

        # Apply air density effects (temperature and pressure dependent)
        weather["air_density_kg_m3"] = baseline["air_density_kg_m3"] * (
            (baseline["pressure_hpa"] / weather["pressure_hpa"])
            * (weather["temperature_c"] + 273.15)
            / (baseline["temperature_c"] + 273.15)
        )

        return weather

    def get_grid_events(self) -> list[dict[str, Any]]:
        """Get comprehensive grid events for storm day scenario.

        Returns:
            List of grid events with timing and parameters
        """
        events = []

        # Transmission line outages during storm peak
        events.extend(
            [
                {
                    "type": "transmission_line_outage",
                    "hour": 6,
                    "line_id": "line_2",
                    "duration_hours": 4,
                    "capacity_reduction": 1.0,
                    "description": "Storm-induced transmission outage on main line",
                },
                {
                    "type": "transmission_line_outage",
                    "hour": 9,
                    "line_id": "line_3",
                    "duration_hours": 2,
                    "capacity_reduction": 0.6,
                    "description": "Partial outage on interconnect line",
                },
            ]
        )

        # Generation trips due to extreme weather
        events.extend(
            [
                {
                    "type": "generation_trip",
                    "hour": 5,
                    "asset_id": "wind_001",
                    "duration_hours": 8,
                    "trigger": "high_wind_speed",
                    "description": "Wind turbine emergency shutdown at 25 m/s",
                },
                {
                    "type": "generation_trip",
                    "hour": 7,
                    "asset_id": "wind_002",
                    "duration_hours": 6,
                    "trigger": "high_wind_speed",
                    "description": "Wind turbine emergency shutdown",
                },
                {
                    "type": "generation_trip",
                    "hour": 8,
                    "asset_id": "solar_002",
                    "duration_hours": 5,
                    "trigger": "storm_conditions",
                    "description": "Solar panel system emergency disconnect",
                },
            ]
        )

        # Emergency response activation
        events.extend(
            [
                {
                    "type": "emergency_response",
                    "hour": 6,
                    "trigger": "transmission_outage",
                    "mode": "grid_stabilization",
                    "priority": "high",
                    "description": "Emergency response activated due to transmission outage",
                },
                {
                    "type": "emergency_response",
                    "hour": 8,
                    "trigger": "frequency_deviation",
                    "mode": "swarm_coordination",
                    "priority": "critical",
                    "description": "Enhanced swarm coordination for frequency control",
                },
            ]
        )

        # Load shedding events
        events.extend(
            [
                {
                    "type": "load_shedding",
                    "hour": 7,
                    "amount_mw": 50.0,
                    "duration_hours": 3,
                    "priority": "low",
                    "description": "Controlled load shedding for grid stability",
                },
                {
                    "type": "load_shedding",
                    "hour": 9,
                    "amount_mw": 80.0,
                    "duration_hours": 2,
                    "priority": "medium",
                    "description": "Additional load shedding during storm peak",
                },
            ]
        )

        # Thermal plant emergency ramping
        events.extend(
            [
                {
                    "type": "thermal_emergency_ramp",
                    "hour": 6,
                    "asset_id": "gas_001",
                    "ramp_rate_multiplier": 2.0,
                    "target_output_mw": 180.0,
                    "description": "Emergency ramping of gas plant",
                }
            ]
        )

        # Battery emergency discharge
        events.extend(
            [
                {
                    "type": "battery_emergency_discharge",
                    "hour": 7,
                    "asset_id": "battery_001",
                    "discharge_rate_mw": 80.0,
                    "duration_hours": 2,
                    "reserve_threshold": 20.0,
                    "description": "Emergency battery discharge for grid support",
                }
            ]
        )

        # Grid frequency events
        events.extend(
            [
                {
                    "type": "frequency_event",
                    "hour": 8,
                    "deviation_hz": -0.4,
                    "duration_minutes": 15,
                    "trigger": "generation_loss",
                    "description": "Grid frequency drop due to generation loss",
                },
                {
                    "type": "frequency_event",
                    "hour": 10,
                    "deviation_hz": 0.25,
                    "duration_minutes": 10,
                    "trigger": "load_recovery",
                    "description": "Frequency overshoot during load recovery",
                },
            ]
        )

        # Recovery and normalization
        events.extend(
            [
                {
                    "type": "system_recovery",
                    "hour": 12,
                    "mode": "automatic",
                    "assets": ["wind_001", "wind_002", "solar_002"],
                    "description": "Automatic system recovery post-storm",
                },
                {
                    "type": "load_restoration",
                    "hour": 13,
                    "amount_mw": 130.0,
                    "rate_mw_per_hour": 40.0,
                    "description": "Gradual load restoration",
                },
            ]
        )

        return events

    def get_asset_configuration(self) -> dict[str, Any]:
        """Get asset configuration optimized for storm day scenario.

        Returns:
            Enhanced asset configuration with storm-resistant settings
        """
        return {
            "renewable": {
                "wind_farms": {
                    "count": 4,
                    "capacity_mw": 75.0,
                    "rotor_diameter_m": 120.0,
                    "hub_height_m": 100.0,
                    "emergency_shutdown_wind_speed": 25.0,
                    "cut_in_wind_speed": 3.0,
                    "rated_wind_speed": 12.0,
                    "storm_protection": True,
                },
                "solar_farms": {
                    "count": 5,
                    "capacity_mw": 40.0,
                    "efficiency": 0.22,
                    "area_m2": 180000.0,
                    "reduced_output": True,
                    "storm_disconnection": True,
                    "tracking_system": False,  # Disable tracking during storm
                },
            },
            "storage": {
                "batteries": {
                    "count": 3,
                    "capacity_mw": 120.0,
                    "energy_capacity_mwh": 480.0,
                    "initial_soc": 80.0,  # Start with higher charge
                    "emergency_reserve": 0.2,  # 20% emergency reserve
                    "priority_discharge": True,
                    "fast_response_capability": True,
                }
            },
            "thermal": {
                "natural_gas": {
                    "count": 2,
                    "capacity_mw": 200.0,
                    "min_output_mw": 40.0,
                    "ramp_rate": 12.0,
                    "emergency_ramping": True,
                    "max_ramp_rate": 25.0,  # Enhanced emergency ramp rate
                    "fuel_availability": "assured",
                },
                "coal": {
                    "count": 1,
                    "capacity_mw": 250.0,
                    "min_output_mw": 100.0,
                    "ramp_rate": 6.0,
                    "emergency_mode": True,
                },
            },
            "load": {
                "demand_response": {
                    "count": 6,
                    "capacity_mw": 120.0,
                    "baseline_mw": 90.0,
                    "dr_capability_mw": 40.0,
                    "emergency_load_shedding": True,
                    "max_shed_percentage": 0.3,  # Up to 30% shedding
                    "priority_classification": True,
                }
            },
        }

    def get_emergency_response_config(self) -> dict[str, Any]:
        """Get enhanced emergency response configuration for storm day.

        Returns:
            Emergency response configuration with storm-specific settings
        """
        return {
            "enable_emergency_mode": True,
            "response_time_seconds": 15.0,  # Faster response during storms
            "activation_triggers": {
                "frequency_deviation": 0.2,  # More sensitive
                "voltage_deviation": 0.1,
                "transmission_outage": True,
                "weather_severity": True,
                "renewable_curtailment": 0.4,  # Trigger at 40% curtailment
            },
            "swarm_coordination": {
                "enhanced_communication": True,
                "response_time_seconds": 5.0,  # Ultra-fast swarm response
                "emergency_pheromones": True,
                "coordination_radius": 8.0,
            },
            "priority_assets": {
                "critical_loads": ["load_001", "load_002"],  # Hospital, emergency services
                "backup_generation": ["gas_001", "gas_002", "battery_001"],
                "energy_storage": ["battery_001", "battery_002", "battery_003"],
            },
            "recovery_procedures": {
                "automatic_restart": True,
                "restart_delay_minutes": 2,  # Faster restart
                "staged_recovery": True,
                "wind_speed_threshold": 20.0,  # Safe wind speed for recovery
            },
            "storm_specific": {
                "wind_turbine_protection": True,
                "transmission_line_monitoring": True,
                "enhanced_weather_monitoring": True,
                "asset_protection_mode": True,
            },
        }

    def get_metrics_configuration(self) -> dict[str, Any]:
        """Get comprehensive metrics configuration for storm day scenario.

        Returns:
            Enhanced metrics configuration for storm resilience testing
        """
        return {
            "enable_emergency_metrics": True,
            "frequency_monitoring": {
                "enable": True,
                "threshold_hz": 0.1,  # More sensitive monitoring
                "nadir_tracking": True,
                "rate_of_change": True,
            },
            "voltage_monitoring": {
                "enable": True,
                "threshold_kv": 5.0,  # More sensitive
                "voltage_stability": True,
                "reactive_power_tracking": True,
            },
            "resilience_metrics": {
                "system_recovery_time": True,
                "load_served_percentage": True,
                "asset_availability": True,
                "storm_impact_assessment": True,
                "grid_stability_index": True,
            },
            "emergency_performance": {
                "response_time_metrics": True,
                "coordination_effectiveness": True,
                "frequency_stability": True,
                "swarm_coordination_efficiency": True,
                "asset_protection_effectiveness": True,
            },
            "weather_impact": {
                "renewable_curtailment": True,
                "weather_correlation": True,
                "forecast_accuracy": True,
                "wind_speed_impact": True,
                "visibility_correlation": True,
            },
            "economic_impact": {
                "lost_load_cost": True,
                "emergency_generation_cost": True,
                "infrastructure_damage_cost": True,
                "recovery_cost": True,
                "resilience_value": True,
            },
            "storm_specific_metrics": {
                "wind_turbine_shutdowns": True,
                "transmission_outage_duration": True,
                "load_shedding_amount": True,
                "battery_emergency_usage": True,
                "thermal_emergency_ramping": True,
                "weather_severity_correlation": True,
            },
        }

    def get_swarm_coordination_config(self) -> dict[str, Any]:
        """Get swarm coordination configuration for storm day.

        Returns:
            Swarm coordination configuration optimized for emergency response
        """
        return {
            "enhanced_mode": True,
            "emergency_pheromones": True,
            "communication_range": 8.0,  # Extended range
            "response_threshold": 0.1,  # More sensitive
            "coordination_frequency": 5.0,  # Every 5 seconds
            "priority_asset_weighting": 2.0,
            "weather_adaptation": True,
            "storm_response_protocols": {
                "wind_turbine_coordination": True,
                "battery_coordination": True,
                "thermal_plant_coordination": True,
                "load_shedding_coordination": True,
            },
        }

    def apply_weather_conditions(self, weather_config: dict[str, Any]) -> None:
        """Apply storm-specific weather conditions and asset responses.

        Args:
            weather_config: Current weather configuration
        """
        wind_speed = weather_config.get("wind_speed_ms", 0.0)
        condition = weather_config.get("condition", WeatherCondition.CLEAR)

        # Apply emergency wind turbine shutdowns
        if wind_speed >= self.emergency_thresholds["wind_turbine_shutdown"]:
            # This would be implemented in the actual orchestrator
            pass

        # Apply solar panel output reduction
        if condition in [WeatherCondition.STORMY, WeatherCondition.RAINY]:
            # This would be implemented in the actual orchestrator
            pass

    def activate_emergency_response(self) -> None:
        """Activate comprehensive emergency response for storm conditions."""
        # This would be implemented in the actual orchestrator
        # - Enhanced swarm coordination
        # - Priority asset management
        # - Emergency generation dispatch
        # - Load shedding coordination
        pass

    def setup_metrics_collection(self) -> None:
        """Setup enhanced metrics collection for storm day scenario."""
        # This would be implemented in the actual orchestrator
        # - Storm-specific metrics
        # - Enhanced frequency monitoring
        # - Asset performance tracking
        # - Resilience metrics
        pass
