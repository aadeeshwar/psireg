"""Wind turbine asset implementation for PSIREG simulation system.

This module provides the WindTurbine class that models wind turbines with
power curves, air density effects, and curtailment logic.
"""

import math
from typing import Any

from pydantic import Field, field_validator

from psireg.sim.assets.base import Asset
from psireg.utils.enums import AssetType, WeatherCondition
from psireg.utils.types import MW


class WindTurbine(Asset):
    """Wind turbine asset.

    Models wind turbine power generation based on wind speed, air density,
    weather conditions, and curtailment signals using a realistic power curve.
    """

    # Wind turbine specific fields
    rotor_diameter_m: float = Field(default=150.0, gt=0.0, description="Rotor diameter in meters")
    hub_height_m: float = Field(default=120.0, gt=0.0, description="Hub height in meters")
    cut_in_speed_ms: float = Field(default=3.0, ge=0.0, le=10.0, description="Cut-in wind speed in m/s")
    cut_out_speed_ms: float = Field(default=25.0, ge=15.0, le=35.0, description="Cut-out wind speed in m/s")
    rated_speed_ms: float = Field(default=12.0, ge=8.0, le=20.0, description="Rated wind speed in m/s")
    power_coefficient: float = Field(
        default=0.45, ge=0.1, le=0.59, description="Power coefficient (Cp) - theoretical max is 0.593 (Betz limit)"
    )

    # Environmental conditions
    current_wind_speed_ms: float = Field(default=0.0, ge=0.0, le=100.0, description="Current wind speed in m/s")
    current_air_density_kg_m3: float = Field(default=1.225, ge=0.5, le=2.0, description="Current air density in kg/m³")
    current_weather_condition: WeatherCondition = Field(
        default=WeatherCondition.CLEAR, description="Current weather condition"
    )

    # Curtailment and control
    curtailment_factor: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Curtailment factor (0.0=full curtailment, 1.0=no curtailment)"
    )

    # Constants
    STANDARD_AIR_DENSITY_KG_M3: float = Field(default=1.225, description="Standard air density at sea level in kg/m³")

    def __init__(self, **data: Any) -> None:
        """Initialize wind turbine with WIND asset type."""
        data["asset_type"] = AssetType.WIND
        super().__init__(**data)

    @field_validator("cut_in_speed_ms")
    @classmethod
    def validate_cut_in_speed(cls, v: float) -> float:
        """Validate cut-in speed is reasonable."""
        if v < 0.0 or v > 10.0:
            raise ValueError("Cut-in speed must be between 0 and 10 m/s")
        return v

    @field_validator("cut_out_speed_ms")
    @classmethod
    def validate_cut_out_speed(cls, v: float) -> float:
        """Validate cut-out speed is reasonable."""
        if v < 15.0 or v > 35.0:
            raise ValueError("Cut-out speed must be between 15 and 35 m/s")
        return v

    @field_validator("rated_speed_ms")
    @classmethod
    def validate_rated_speed(cls, v: float) -> float:
        """Validate rated speed is reasonable."""
        if v < 8.0 or v > 20.0:
            raise ValueError("Rated speed must be between 8 and 20 m/s")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate wind speed relationships."""
        if self.cut_in_speed_ms >= self.rated_speed_ms:
            raise ValueError("Cut-in speed must be less than rated speed")
        if self.rated_speed_ms >= self.cut_out_speed_ms:
            raise ValueError("Rated speed must be less than cut-out speed")

    def set_wind_speed(self, wind_speed_ms: float) -> None:
        """Set current wind speed.

        Args:
            wind_speed_ms: Wind speed in m/s
        """
        self.current_wind_speed_ms = max(0.0, min(wind_speed_ms, 100.0))

    def set_air_density(self, air_density_kg_m3: float) -> None:
        """Set current air density.

        Args:
            air_density_kg_m3: Air density in kg/m³
        """
        self.current_air_density_kg_m3 = max(0.5, min(air_density_kg_m3, 2.0))

    def set_weather_condition(self, condition: WeatherCondition) -> None:
        """Set current weather condition.

        Args:
            condition: Weather condition affecting wind generation
        """
        self.current_weather_condition = condition

    def set_curtailment_factor(self, factor: float) -> None:
        """Set curtailment factor.

        Args:
            factor: Curtailment factor (0.0=full curtailment, 1.0=no curtailment)
        """
        self.curtailment_factor = max(0.0, min(factor, 1.0))

    def calculate_power_output(self) -> MW:
        """Calculate current power output based on conditions.

        Returns:
            Power output in MW
        """
        # Return 0 if offline
        if not self.is_online():
            return 0.0

        # Check weather conditions first
        if self._is_weather_shutdown():
            return 0.0

        # Check wind speed ranges
        if self.current_wind_speed_ms < self.cut_in_speed_ms or self.current_wind_speed_ms >= self.cut_out_speed_ms:
            return 0.0

        # Calculate power using simplified power curve
        power_mw = self._calculate_power_curve()

        # Apply air density correction
        density_factor = self.current_air_density_kg_m3 / self.STANDARD_AIR_DENSITY_KG_M3
        power_mw *= density_factor

        # Apply weather condition factor
        weather_factor = self._get_weather_factor()
        power_mw *= weather_factor

        # Apply curtailment
        power_mw *= self.curtailment_factor

        # Ensure power doesn't exceed capacity
        power_mw = min(power_mw, self.capacity_mw)

        return max(0.0, power_mw)

    def _calculate_power_curve(self) -> MW:
        """Calculate power using simplified power curve model.

        Returns:
            Power in MW based on wind speed
        """
        wind_speed = self.current_wind_speed_ms

        if wind_speed < self.cut_in_speed_ms:
            return 0.0
        elif wind_speed >= self.cut_out_speed_ms:
            return 0.0
        elif wind_speed <= self.rated_speed_ms:
            # Power curve region (cubic relationship)
            # P = 0.5 * ρ * A * Cp * v³
            # Simplified as linear interpolation for practical purposes
            rotor_area = math.pi * (self.rotor_diameter_m / 2.0) ** 2

            # Use theoretical wind power calculation
            theoretical_power_w = (
                0.5 * self.STANDARD_AIR_DENSITY_KG_M3 * rotor_area * self.power_coefficient * (wind_speed**3)
            )

            theoretical_power_mw = theoretical_power_w / 1_000_000.0

            # Scale to match capacity at rated speed
            scale_factor = self.capacity_mw / self._get_rated_theoretical_power()
            return theoretical_power_mw * scale_factor
        else:
            # Rated power region (between rated and cut-out)
            return self.capacity_mw

    def _get_rated_theoretical_power(self) -> MW:
        """Calculate theoretical power at rated wind speed.

        Returns:
            Theoretical power in MW at rated wind speed
        """
        rotor_area = math.pi * (self.rotor_diameter_m / 2.0) ** 2
        theoretical_power_w = (
            0.5 * self.STANDARD_AIR_DENSITY_KG_M3 * rotor_area * self.power_coefficient * (self.rated_speed_ms**3)
        )
        return theoretical_power_w / 1_000_000.0

    def _is_weather_shutdown(self) -> bool:
        """Check if weather conditions require shutdown.

        Returns:
            True if turbine should be shut down due to weather
        """
        shutdown_conditions = {
            WeatherCondition.STORMY,
            # Add other severe weather conditions as needed
        }
        return self.current_weather_condition in shutdown_conditions

    def _get_weather_factor(self) -> float:
        """Get weather-based power adjustment factor.

        Returns:
            Weather adjustment factor (0.0 to 1.0)
        """
        weather_factors = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.PARTLY_CLOUDY: 1.0,
            WeatherCondition.CLOUDY: 1.0,
            WeatherCondition.RAINY: 0.95,  # Slight reduction due to blade contamination
            WeatherCondition.SNOWY: 0.85,  # Ice buildup can reduce efficiency
            WeatherCondition.FOGGY: 0.98,
            WeatherCondition.STORMY: 0.0,  # Shutdown condition
            WeatherCondition.WINDY: 1.05,  # Optimal wind conditions
        }
        return weather_factors.get(self.current_weather_condition, 1.0)

    def tick(self, dt_seconds: float) -> float:
        """Update wind turbine state for one simulation timestep.

        Args:
            dt_seconds: Time delta in seconds since last tick

        Returns:
            Power output change in MW
        """
        if not self.is_online():
            self.current_output_mw = 0.0
            return 0.0

        # Calculate new power output
        new_power = self.calculate_power_output()
        power_change = new_power - self.current_output_mw
        self.current_output_mw = new_power

        return power_change

    def get_state(self) -> dict[str, Any]:
        """Get comprehensive state information for the wind turbine.

        Returns:
            Dictionary containing complete wind turbine state
        """
        base_state = super().get_state()

        # Add wind-specific state
        wind_state = {
            "rotor_diameter_m": self.rotor_diameter_m,
            "hub_height_m": self.hub_height_m,
            "cut_in_speed_ms": self.cut_in_speed_ms,
            "cut_out_speed_ms": self.cut_out_speed_ms,
            "rated_speed_ms": self.rated_speed_ms,
            "power_coefficient": self.power_coefficient,
            "current_wind_speed_ms": self.current_wind_speed_ms,
            "current_air_density_kg_m3": self.current_air_density_kg_m3,
            "current_weather_condition": self.current_weather_condition,
            "curtailment_factor": self.curtailment_factor,
            "capacity_factor": self.get_capacity_factor(),
            "theoretical_max_power_mw": self.get_theoretical_max_power(),
            "rotor_area_m2": self.get_rotor_area(),
        }

        base_state.update(wind_state)
        return base_state

    def get_capacity_factor(self) -> float:
        """Calculate current capacity factor.

        Returns:
            Capacity factor (0.0 to 1.0)
        """
        if self.capacity_mw == 0:
            return 0.0
        return self.current_output_mw / self.capacity_mw

    def get_theoretical_max_power(self) -> MW:
        """Calculate theoretical maximum power under current wind conditions.

        Returns:
            Theoretical maximum power in MW
        """
        if self.current_wind_speed_ms < self.cut_in_speed_ms:
            return 0.0
        elif self.current_wind_speed_ms >= self.cut_out_speed_ms:
            return 0.0
        else:
            rotor_area = self.get_rotor_area()
            theoretical_power_w = (
                0.5
                * self.current_air_density_kg_m3
                * rotor_area
                * self.power_coefficient
                * (self.current_wind_speed_ms**3)
            )
            return theoretical_power_w / 1_000_000.0

    def get_rotor_area(self) -> float:
        """Calculate rotor swept area.

        Returns:
            Rotor swept area in m²
        """
        return math.pi * (self.rotor_diameter_m / 2.0) ** 2

    def get_efficiency(self) -> float:
        """Get current efficiency considering wind conditions.

        Returns:
            Current efficiency factor (0.0 to 1.0)
        """
        if self.current_wind_speed_ms < self.cut_in_speed_ms:
            return 0.0

        # Calculate efficiency based on actual vs theoretical output
        theoretical_max = self.get_theoretical_max_power()
        if theoretical_max == 0:
            return 0.0

        actual_power = self.calculate_power_output()
        return min(1.0, actual_power / theoretical_max)

    def __str__(self) -> str:
        """String representation of the wind turbine."""
        return (
            f"{self.name} (Wind Turbine, {self.capacity_mw:.1f} MW, "
            f"{self.rotor_diameter_m:.0f}m rotor, {self.hub_height_m:.0f}m hub)"
        )
