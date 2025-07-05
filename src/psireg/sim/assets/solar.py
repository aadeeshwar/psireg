"""Solar panel asset implementation for PSIREG simulation system.

This module provides the SolarPanel class that models photovoltaic solar panels
with irradiance response curves, temperature derating, and curtailment logic.
"""

from typing import Any

from pydantic import Field, field_validator

from psireg.sim.assets.base import Asset
from psireg.utils.enums import AssetType, WeatherCondition
from psireg.utils.types import MW


class SolarPanel(Asset):
    """Solar photovoltaic panel asset.

    Models solar panel power generation based on irradiance, temperature,
    weather conditions, and curtailment signals.
    """

    # Solar panel specific fields
    panel_efficiency: float = Field(default=0.20, ge=0.01, le=1.0, description="Solar panel efficiency (0.01 to 1.0)")
    panel_area_m2: float = Field(default=10000.0, gt=0.0, description="Solar panel area in square meters")
    tilt_degrees: float = Field(
        default=25.0, ge=0.0, le=90.0, description="Panel tilt angle in degrees (0=horizontal, 90=vertical)"
    )
    azimuth_degrees: float = Field(
        default=180.0,
        ge=0.0,
        le=360.0,
        description="Panel azimuth angle in degrees (0=north, 90=east, 180=south, 270=west)",
    )

    # Environmental conditions
    current_irradiance_w_m2: float = Field(
        default=0.0, ge=0.0, le=1500.0, description="Current solar irradiance in W/m²"
    )
    current_temperature_c: float = Field(
        default=25.0, ge=-40.0, le=80.0, description="Current ambient temperature in Celsius"
    )
    current_weather_condition: WeatherCondition = Field(
        default=WeatherCondition.CLEAR, description="Current weather condition"
    )

    # Curtailment and control
    curtailment_factor: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Curtailment factor (0.0=full curtailment, 1.0=no curtailment)"
    )

    # Constants for temperature derating
    TEMP_COEFF_POWER_PER_C: float = Field(default=-0.004, description="Temperature coefficient of power per degree C")
    STC_TEMPERATURE_C: float = Field(default=25.0, description="Standard Test Conditions temperature in Celsius")

    def __init__(self, **data: Any) -> None:
        """Initialize solar panel with SOLAR asset type."""
        data["asset_type"] = AssetType.SOLAR
        super().__init__(**data)

    @field_validator("panel_efficiency")
    @classmethod
    def validate_efficiency(cls, v: float) -> float:
        """Validate panel efficiency is reasonable."""
        if v <= 0.0 or v > 1.0:
            raise ValueError("Panel efficiency must be between 0.01 and 1.0")
        return v

    @field_validator("panel_area_m2")
    @classmethod
    def validate_area(cls, v: float) -> float:
        """Validate panel area is positive."""
        if v <= 0.0:
            raise ValueError("Panel area must be positive")
        return v

    @field_validator("tilt_degrees")
    @classmethod
    def validate_tilt(cls, v: float) -> float:
        """Validate tilt angle is reasonable."""
        if v < 0.0 or v > 90.0:
            raise ValueError("Tilt angle must be between 0 and 90 degrees")
        return v

    def set_irradiance(self, irradiance_w_m2: float) -> None:
        """Set current solar irradiance.

        Args:
            irradiance_w_m2: Solar irradiance in W/m²
        """
        self.current_irradiance_w_m2 = max(0.0, min(irradiance_w_m2, 1500.0))

    def set_temperature(self, temperature_c: float) -> None:
        """Set current ambient temperature.

        Args:
            temperature_c: Ambient temperature in Celsius
        """
        self.current_temperature_c = max(-40.0, min(temperature_c, 80.0))

    def set_weather_condition(self, condition: WeatherCondition) -> None:
        """Set current weather condition.

        Args:
            condition: Weather condition affecting solar generation
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

        # Start with base power calculation
        # P = Irradiance × Area × Efficiency
        base_power_w = self.current_irradiance_w_m2 * self.panel_area_m2 * self.panel_efficiency

        # Convert to MW
        base_power_mw = base_power_w / 1_000_000.0

        # Apply temperature derating
        temp_factor = 1.0 + self.TEMP_COEFF_POWER_PER_C * (self.current_temperature_c - self.STC_TEMPERATURE_C)
        power_mw = base_power_mw * temp_factor

        # Apply weather condition factors
        weather_factor = self._get_weather_factor()
        power_mw *= weather_factor

        # Apply curtailment
        power_mw *= self.curtailment_factor

        # Ensure power doesn't exceed capacity
        power_mw = min(power_mw, self.capacity_mw)

        return max(0.0, power_mw)

    def _get_weather_factor(self) -> float:
        """Get weather-based power adjustment factor.

        Returns:
            Weather adjustment factor (0.0 to 1.0)
        """
        weather_factors = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.PARTLY_CLOUDY: 0.8,
            WeatherCondition.CLOUDY: 0.3,
            WeatherCondition.RAINY: 0.2,
            WeatherCondition.SNOWY: 0.1,
            WeatherCondition.FOGGY: 0.4,
            WeatherCondition.STORMY: 0.1,
            WeatherCondition.WINDY: 0.95,  # Slight cooling benefit
        }
        return weather_factors.get(self.current_weather_condition, 1.0)

    def tick(self, dt_seconds: float) -> float:
        """Update solar panel state for one simulation timestep.

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
        """Get comprehensive state information for the solar panel.

        Returns:
            Dictionary containing complete solar panel state
        """
        base_state = super().get_state()

        # Add solar-specific state
        solar_state = {
            "panel_efficiency": self.panel_efficiency,
            "panel_area_m2": self.panel_area_m2,
            "tilt_degrees": self.tilt_degrees,
            "azimuth_degrees": self.azimuth_degrees,
            "current_irradiance_w_m2": self.current_irradiance_w_m2,
            "current_temperature_c": self.current_temperature_c,
            "current_weather_condition": self.current_weather_condition,
            "curtailment_factor": self.curtailment_factor,
            "capacity_factor": self.get_capacity_factor(),
            "theoretical_max_power_mw": self.get_theoretical_max_power(),
        }

        base_state.update(solar_state)
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
        """Calculate theoretical maximum power under current irradiance.

        Returns:
            Theoretical maximum power in MW
        """
        theoretical_w = self.current_irradiance_w_m2 * self.panel_area_m2 * self.panel_efficiency
        return theoretical_w / 1_000_000.0

    def get_efficiency(self) -> float:
        """Get current efficiency considering temperature and weather.

        Returns:
            Current efficiency factor (0.0 to 1.0)
        """
        if self.current_irradiance_w_m2 == 0:
            return 0.0

        # Calculate efficiency based on actual vs theoretical output
        theoretical_max = self.get_theoretical_max_power()
        if theoretical_max == 0:
            return 0.0

        actual_power = self.calculate_power_output()
        return min(1.0, actual_power / theoretical_max)

    def __str__(self) -> str:
        """String representation of the solar panel."""
        return (
            f"{self.name} (Solar Panel, {self.capacity_mw:.1f} MW, "
            f"{self.panel_efficiency:.1%} efficiency, {self.panel_area_m2:.0f} m²)"
        )
