"""Battery storage asset implementation for PSIREG simulation system.

This module provides the Battery class that models battery energy storage systems
with State of Charge (SoC) management, charge/discharge efficiency, voltage sensing,
thermal effects, and degradation modeling.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field, field_validator

from psireg.sim.assets.base import Asset
from psireg.utils.enums import AssetType
from psireg.utils.types import MW, MWh

logger = logging.getLogger(__name__)


class Battery(Asset):
    """Battery energy storage system asset.

    Models battery storage with comprehensive functionality including:
    - State of Charge (SoC) management
    - Charge/discharge efficiency curves
    - Voltage sensing with SoC dependency
    - Thermal modeling and temperature effects
    - Battery degradation and health tracking
    - Power limits based on SoC and temperature
    """

    # Battery capacity parameters
    energy_capacity_mwh: MWh = Field(..., gt=0.0, description="Total energy capacity in MWh")
    initial_soc_percent: float = Field(default=50.0, ge=0.0, le=100.0, description="Initial State of Charge percentage")
    current_soc_percent: float = Field(default=50.0, ge=0.0, le=100.0, description="Current State of Charge percentage")
    min_soc_percent: float = Field(default=5.0, ge=0.0, le=100.0, description="Minimum allowable SoC percentage")
    max_soc_percent: float = Field(default=95.0, ge=0.0, le=100.0, description="Maximum allowable SoC percentage")

    # Efficiency parameters
    charge_efficiency: float = Field(default=0.95, ge=0.01, le=1.0, description="Charge efficiency (0.01 to 1.0)")
    discharge_efficiency: float = Field(default=0.92, ge=0.01, le=1.0, description="Discharge efficiency (0.01 to 1.0)")
    round_trip_efficiency: float = Field(
        default=0.87, ge=0.01, le=1.0, description="Round-trip efficiency (charge * discharge)"
    )

    # Voltage and electrical parameters
    nominal_voltage_v: float = Field(default=800.0, gt=0.0, description="Nominal voltage in volts")
    min_voltage_v: float = Field(default=600.0, gt=0.0, description="Minimum voltage in volts")
    max_voltage_v: float = Field(default=900.0, gt=0.0, description="Maximum voltage in volts")
    internal_resistance_ohm: float = Field(default=0.01, ge=0.0, description="Internal resistance in ohms")

    # Thermal parameters
    current_temperature_c: float = Field(
        default=25.0, ge=-40.0, le=80.0, description="Current battery temperature in Celsius"
    )
    ambient_temperature_c: float = Field(default=25.0, ge=-40.0, le=80.0, description="Ambient temperature in Celsius")
    thermal_time_constant_s: float = Field(default=3600.0, gt=0.0, description="Thermal time constant in seconds")
    thermal_resistance_c_per_kw: float = Field(default=0.1, ge=0.0, description="Thermal resistance in °C/kW")

    # Degradation and health parameters
    initial_health_percent: float = Field(
        default=100.0, ge=0.0, le=100.0, description="Initial battery health percentage"
    )
    current_health_percent: float = Field(
        default=100.0, ge=0.0, le=100.0, description="Current battery health percentage"
    )
    cycle_count: float = Field(default=0.0, ge=0.0, description="Total equivalent full cycles")
    degradation_rate_per_cycle: float = Field(
        default=0.0001, ge=0.0, le=1.0, description="Health degradation rate per cycle"
    )

    # Control parameters
    power_setpoint_mw: MW = Field(
        default=0.0, description="Power setpoint in MW (positive=charging, negative=discharging)"
    )
    max_c_rate: float = Field(default=1.0, gt=0.0, description="Maximum C-rate (charge/discharge rate)")

    # Temperature coefficients
    TEMP_COEFF_EFFICIENCY_PER_C: float = Field(
        default=-0.005, description="Temperature coefficient of efficiency per degree C"
    )
    TEMP_COEFF_VOLTAGE_PER_C: float = Field(
        default=-0.002, description="Temperature coefficient of voltage per degree C"
    )
    TEMP_COEFF_POWER_PER_C: float = Field(default=-0.01, description="Temperature coefficient of power per degree C")
    OPTIMAL_TEMPERATURE_C: float = Field(default=25.0, description="Optimal operating temperature in Celsius")

    def __init__(self, **data: Any) -> None:
        """Initialize battery with BATTERY asset type."""
        data["asset_type"] = AssetType.BATTERY
        super().__init__(**data)

        # Set current SoC to initial SoC
        self.current_soc_percent = self.initial_soc_percent
        self.current_health_percent = self.initial_health_percent

        # Calculate round-trip efficiency
        self.round_trip_efficiency = self.charge_efficiency * self.discharge_efficiency

    @field_validator("energy_capacity_mwh")
    @classmethod
    def validate_energy_capacity(cls, v: float) -> float:
        """Validate energy capacity is positive."""
        if v <= 0.0:
            raise ValueError("Energy capacity must be positive")
        return v

    @field_validator("initial_soc_percent", "current_soc_percent", "min_soc_percent", "max_soc_percent")
    @classmethod
    def validate_soc_percent(cls, v: float) -> float:
        """Validate SoC percentage is between 0 and 100."""
        if v < 0.0 or v > 100.0:
            raise ValueError("SoC percentage must be between 0 and 100")
        return v

    @field_validator("charge_efficiency", "discharge_efficiency", "round_trip_efficiency")
    @classmethod
    def validate_efficiency(cls, v: float) -> float:
        """Validate efficiency is between 0.01 and 1.0."""
        if v < 0.01 or v > 1.0:
            raise ValueError("Efficiency must be between 0.01 and 1.0")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate battery configuration after initialization."""
        if self.min_soc_percent >= self.max_soc_percent:
            raise ValueError("Minimum SoC must be less than maximum SoC")

        if self.min_voltage_v >= self.max_voltage_v:
            raise ValueError("Minimum voltage must be less than maximum voltage")

        if self.nominal_voltage_v < self.min_voltage_v or self.nominal_voltage_v > self.max_voltage_v:
            raise ValueError("Nominal voltage must be between minimum and maximum voltage")

    def set_power_setpoint(self, power_mw: MW) -> None:
        """Set power setpoint for charging/discharging.

        Args:
            power_mw: Power setpoint in MW (positive=charging, negative=discharging)
        """
        # Clamp power setpoint to available limits
        max_charge = self.get_max_charge_power()
        max_discharge = self.get_max_discharge_power()

        if power_mw > 0:
            # Charging
            self.power_setpoint_mw = min(power_mw, max_charge)
        elif power_mw < 0:
            # Discharging
            self.power_setpoint_mw = max(power_mw, -max_discharge)
        else:
            # Idle
            self.power_setpoint_mw = 0.0

    def get_max_charge_power(self) -> MW:
        """Get maximum charge power based on current conditions.

        Returns:
            Maximum charge power in MW
        """
        if not self.is_online():
            return 0.0

        # Base power limit from capacity
        base_power = self.capacity_mw

        # SoC-based limitation
        if self.current_soc_percent >= self.max_soc_percent:
            return 0.0

        # Reduce power as SoC approaches maximum
        soc_factor = 1.0 - max(0.0, (self.current_soc_percent - 80.0) / 20.0)

        # C-rate limitation
        c_rate_limit = self.energy_capacity_mwh * self.max_c_rate

        # Temperature-based limitation
        temp_factor = self._get_temperature_power_factor()

        # Health-based limitation
        health_factor = self.current_health_percent / 100.0

        return min(base_power, c_rate_limit) * soc_factor * temp_factor * health_factor

    def get_max_discharge_power(self) -> MW:
        """Get maximum discharge power based on current conditions.

        Returns:
            Maximum discharge power in MW
        """
        if not self.is_online():
            return 0.0

        # Base power limit from capacity
        base_power = self.capacity_mw

        # SoC-based limitation
        if self.current_soc_percent <= self.min_soc_percent:
            return 0.0

        # Reduce power as SoC approaches minimum
        soc_factor = 1.0 - max(0.0, (20.0 - self.current_soc_percent) / 20.0)

        # C-rate limitation
        c_rate_limit = self.energy_capacity_mwh * self.max_c_rate

        # Temperature-based limitation
        temp_factor = self._get_temperature_power_factor()

        # Health-based limitation
        health_factor = self.current_health_percent / 100.0

        return min(base_power, c_rate_limit) * soc_factor * temp_factor * health_factor

    def get_stored_energy_mwh(self) -> MWh:
        """Get currently stored energy.

        Returns:
            Stored energy in MWh
        """
        effective_capacity = self.get_effective_capacity_mwh()
        return effective_capacity * (self.current_soc_percent / 100.0)

    def get_effective_capacity_mwh(self) -> MWh:
        """Get effective capacity considering degradation.

        Returns:
            Effective capacity in MWh
        """
        return self.energy_capacity_mwh * (self.current_health_percent / 100.0)

    def get_terminal_voltage(self) -> float:
        """Get terminal voltage based on SoC, load, and temperature.

        Returns:
            Terminal voltage in volts
        """
        # Base voltage from SoC curve
        base_voltage = self._get_soc_voltage()

        # Load-based voltage drop/rise
        load_voltage = self._get_load_voltage_effect()

        # Temperature effect
        temp_voltage = self._get_temperature_voltage_effect()

        # Combine effects
        terminal_voltage = base_voltage + load_voltage + temp_voltage

        # Clamp to min/max limits
        return max(self.min_voltage_v, min(terminal_voltage, self.max_voltage_v))

    def _get_soc_voltage(self) -> float:
        """Get voltage based on SoC curve.

        Returns:
            Voltage in volts
        """
        # Simplified Li-ion voltage curve
        soc_fraction = self.current_soc_percent / 100.0

        if soc_fraction <= 0.1:
            # Low SoC region - steep voltage drop
            voltage_factor = 0.75 + 0.1 * (soc_fraction / 0.1)
        elif soc_fraction <= 0.9:
            # Mid SoC region - relatively flat
            voltage_factor = 0.85 + 0.1 * ((soc_fraction - 0.1) / 0.8)
        else:
            # High SoC region - voltage rise
            voltage_factor = 0.95 + 0.05 * ((soc_fraction - 0.9) / 0.1)

        return self.nominal_voltage_v * voltage_factor

    def _get_load_voltage_effect(self) -> float:
        """Get voltage effect from current load.

        Returns:
            Voltage change in volts
        """
        # Voltage drop/rise due to internal resistance
        # Use nominal voltage to avoid recursion
        if self.nominal_voltage_v > 0 and abs(self.power_setpoint_mw) > 0.001:
            # For discharge: voltage drops (negative effect)
            # For charge: voltage rises slightly (positive effect)
            if self.power_setpoint_mw < 0:
                # Discharging - voltage drop
                current_magnitude = abs(self.power_setpoint_mw) * 1000.0 / self.nominal_voltage_v
                return -current_magnitude * self.internal_resistance_ohm
            else:
                # Charging - slight voltage rise
                current_magnitude = self.power_setpoint_mw * 1000.0 / self.nominal_voltage_v
                return current_magnitude * self.internal_resistance_ohm * 0.5  # Smaller effect for charging
        return 0.0

    def _get_temperature_voltage_effect(self) -> float:
        """Get voltage effect from temperature.

        Returns:
            Voltage change in volts
        """
        temp_deviation = self.current_temperature_c - self.OPTIMAL_TEMPERATURE_C
        return self.nominal_voltage_v * self.TEMP_COEFF_VOLTAGE_PER_C * temp_deviation

    def get_current_charge_efficiency(self) -> float:
        """Get current charge efficiency considering temperature and SoC.

        Returns:
            Current charge efficiency (0.0 to 1.0)
        """
        base_efficiency = self.charge_efficiency

        # Temperature effect
        temp_factor = self._get_temperature_efficiency_factor()

        # SoC effect (efficiency decreases at high SoC)
        soc_factor = 1.0 - max(0.0, (self.current_soc_percent - 80.0) / 40.0) * 0.1

        return base_efficiency * temp_factor * soc_factor

    def get_current_discharge_efficiency(self) -> float:
        """Get current discharge efficiency considering temperature and SoC.

        Returns:
            Current discharge efficiency (0.0 to 1.0)
        """
        base_efficiency = self.discharge_efficiency

        # Temperature effect
        temp_factor = self._get_temperature_efficiency_factor()

        # SoC effect (efficiency decreases at low SoC)
        soc_factor = 1.0 - max(0.0, (20.0 - self.current_soc_percent) / 20.0) * 0.1

        return base_efficiency * temp_factor * soc_factor

    def _get_temperature_efficiency_factor(self) -> float:
        """Get temperature-based efficiency factor.

        Returns:
            Temperature efficiency factor (0.0 to 1.0)
        """
        temp_deviation = abs(self.current_temperature_c - self.OPTIMAL_TEMPERATURE_C)
        return max(0.5, 1.0 + self.TEMP_COEFF_EFFICIENCY_PER_C * temp_deviation)

    def _get_temperature_power_factor(self) -> float:
        """Get temperature-based power factor.

        Returns:
            Temperature power factor (0.0 to 1.0)
        """
        temp_deviation = abs(self.current_temperature_c - self.OPTIMAL_TEMPERATURE_C)
        return max(0.1, 1.0 - abs(self.TEMP_COEFF_POWER_PER_C) * temp_deviation)

    def set_temperature(self, temperature_c: float) -> None:
        """Set current battery temperature.

        Args:
            temperature_c: Temperature in Celsius
        """
        self.current_temperature_c = max(-40.0, min(temperature_c, 80.0))

    def get_cycle_count(self) -> float:
        """Get total equivalent full cycle count.

        Returns:
            Total cycle count
        """
        return self.cycle_count

    def calculate_power_output(self) -> MW:
        """Calculate current power output based on setpoint and limits.

        Returns:
            Power output in MW (positive=charging, negative=discharging)
        """
        if not self.is_online():
            return 0.0

        # Return the setpoint (already limited by set_power_setpoint)
        return self.power_setpoint_mw

    def tick(self, dt_seconds: float) -> float:
        """Update battery state for one simulation timestep.

        Args:
            dt_seconds: Time delta in seconds since last tick

        Returns:
            Power output change in MW
        """
        if not self.is_online():
            self.current_output_mw = 0.0
            return 0.0

        # Calculate power output
        new_power = self.calculate_power_output()
        power_change = new_power - self.current_output_mw
        self.current_output_mw = new_power

        # Update SoC based on power flow
        self._update_soc(dt_seconds)

        # Update temperature
        self._update_temperature(dt_seconds)

        # Update degradation
        self._update_degradation(dt_seconds)

        return power_change

    def _update_soc(self, dt_seconds: float) -> None:
        """Update State of Charge based on power flow.

        Args:
            dt_seconds: Time delta in seconds
        """
        dt_hours = dt_seconds / 3600.0
        energy_flow_mwh = self.current_output_mw * dt_hours

        if energy_flow_mwh > 0:
            # Charging
            efficiency = self.get_current_charge_efficiency()
            energy_stored = energy_flow_mwh * efficiency
        elif energy_flow_mwh < 0:
            # Discharging
            efficiency = self.get_current_discharge_efficiency()
            energy_stored = energy_flow_mwh / efficiency
        else:
            # Idle
            energy_stored = 0.0

        # Update SoC
        effective_capacity = self.get_effective_capacity_mwh()
        if effective_capacity > 0:
            soc_change = (energy_stored / effective_capacity) * 100.0
            self.current_soc_percent = max(
                self.min_soc_percent, min(self.max_soc_percent, self.current_soc_percent + soc_change)
            )

        # Update cycle count
        if abs(energy_flow_mwh) > 0:
            cycle_increment = abs(energy_flow_mwh) / self.energy_capacity_mwh
            self.cycle_count += cycle_increment

    def _update_temperature(self, dt_seconds: float) -> None:
        """Update battery temperature based on power losses and cooling.

        Args:
            dt_seconds: Time delta in seconds
        """
        # Calculate power losses (heating)
        power_losses_kw = self._calculate_power_losses()

        # Temperature rise from power losses
        temp_rise_rate = power_losses_kw * self.thermal_resistance_c_per_kw

        # Cooling toward ambient temperature
        temp_diff = self.current_temperature_c - self.ambient_temperature_c
        cooling_rate = temp_diff / self.thermal_time_constant_s

        # Update temperature
        dt_hours = dt_seconds / 3600.0
        temp_change = (temp_rise_rate - cooling_rate) * dt_hours
        self.current_temperature_c += temp_change

        # Clamp temperature to reasonable bounds
        self.current_temperature_c = max(-40.0, min(80.0, self.current_temperature_c))

    def _calculate_power_losses(self) -> float:
        """Calculate power losses for thermal modeling.

        Returns:
            Power losses in kW
        """
        if abs(self.current_output_mw) < 0.001:
            return 0.0

        if self.current_output_mw > 0:
            # Charging losses
            efficiency = self.get_current_charge_efficiency()
            losses = self.current_output_mw * (1.0 - efficiency)
        else:
            # Discharging losses
            efficiency = self.get_current_discharge_efficiency()
            losses = abs(self.current_output_mw) * (1.0 - efficiency)

        return losses * 1000.0  # Convert MW to kW

    def _update_degradation(self, dt_seconds: float) -> None:
        """Update battery degradation based on usage.

        Args:
            dt_seconds: Time delta in seconds
        """
        # Calculate degradation based on cycle count
        if self.cycle_count > 0:
            degradation = self.cycle_count * self.degradation_rate_per_cycle
            self.current_health_percent = max(0.0, self.initial_health_percent - degradation * 100.0)

    def get_state(self) -> dict[str, Any]:
        """Get comprehensive state information for the battery.

        Returns:
            Dictionary containing complete battery state
        """
        base_state = super().get_state()

        # Add battery-specific state
        battery_state = {
            "energy_capacity_mwh": self.energy_capacity_mwh,
            "current_soc_percent": self.current_soc_percent,
            "stored_energy_mwh": self.get_stored_energy_mwh(),
            "effective_capacity_mwh": self.get_effective_capacity_mwh(),
            "min_soc_percent": self.min_soc_percent,
            "max_soc_percent": self.max_soc_percent,
            "charge_efficiency": self.get_current_charge_efficiency(),
            "discharge_efficiency": self.get_current_discharge_efficiency(),
            "round_trip_efficiency": self.round_trip_efficiency,
            "terminal_voltage_v": self.get_terminal_voltage(),
            "nominal_voltage_v": self.nominal_voltage_v,
            "current_temperature_c": self.current_temperature_c,
            "ambient_temperature_c": self.ambient_temperature_c,
            "current_health_percent": self.current_health_percent,
            "cycle_count": self.cycle_count,
            "power_setpoint_mw": self.power_setpoint_mw,
            "max_charge_power_mw": self.get_max_charge_power(),
            "max_discharge_power_mw": self.get_max_discharge_power(),
            "is_charging": self.current_output_mw > 0,
            "is_discharging": self.current_output_mw < 0,
            "is_idle": abs(self.current_output_mw) < 0.001,
        }

        base_state.update(battery_state)
        return base_state

    def reset(self) -> None:
        """Reset battery to initial state."""
        super().reset()
        self.current_soc_percent = self.initial_soc_percent
        self.current_health_percent = self.initial_health_percent
        self.current_temperature_c = self.ambient_temperature_c
        self.power_setpoint_mw = 0.0
        self.cycle_count = 0.0

    def __str__(self) -> str:
        """String representation of the battery."""
        return (
            f"{self.name} (Battery, {self.capacity_mw:.1f} MW, "
            f"{self.energy_capacity_mwh:.1f} MWh, {self.current_soc_percent:.1f}% SoC)"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the battery."""
        return (
            f"Battery(id={self.asset_id}, power={self.capacity_mw:.1f} MW, "
            f"energy={self.energy_capacity_mwh:.1f} MWh, soc={self.current_soc_percent:.1f}%, "
            f"health={self.current_health_percent:.1f}%, temp={self.current_temperature_c:.1f}°C)"
        )
