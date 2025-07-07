"""Thermal power plant asset implementations for PSIREG simulation system.

This module provides thermal power plant classes including coal, natural gas, and nuclear
plants with realistic operational characteristics like minimum load, ramp rates,
efficiency, fuel costs, and emissions.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field, field_validator

from psireg.sim.assets.base import Asset
from psireg.utils.enums import AssetType
from psireg.utils.types import MW

logger = logging.getLogger(__name__)


class ThermalPlant(Asset):
    """Base class for thermal power plants with common characteristics."""

    # Operational parameters
    efficiency: float = Field(default=0.35, ge=0.1, le=0.6, description="Plant efficiency (0.1-0.6)")
    min_load_mw: MW = Field(default=0.0, ge=0.0, description="Minimum stable load in MW")
    max_load_mw: MW = Field(default=0.0, ge=0.0, description="Maximum load in MW")

    # Ramping characteristics
    ramp_rate_mw_per_min: float = Field(default=5.0, ge=0.1, le=50.0, description="Ramp rate in MW/min")
    startup_time_minutes: int = Field(default=60, ge=5, le=4320, description="Startup time in minutes (up to 72 hours)")
    shutdown_time_minutes: int = Field(
        default=30, ge=5, le=1440, description="Shutdown time in minutes (up to 24 hours)"
    )

    # Fuel parameters
    heat_rate_btu_per_kwh: float = Field(default=9500.0, ge=6000.0, le=15000.0, description="Heat rate in BTU/kWh")
    fuel_cost_per_mmbtu: float = Field(default=3.0, ge=0.0, le=50.0, description="Fuel cost in $/MMBTU")

    # Emissions
    co2_emissions_lb_per_mmbtu: float = Field(default=205.0, ge=0.0, le=300.0, description="CO2 emissions in lb/MMBTU")

    # Control parameters
    load_setpoint_mw: MW = Field(default=0.0, ge=0.0, description="Load setpoint in MW")

    def __init__(self, **data: Any) -> None:
        """Initialize thermal plant."""
        super().__init__(**data)

        # Set default max_load_mw to capacity if not specified
        if self.max_load_mw == 0.0:
            self.max_load_mw = self.capacity_mw

    @field_validator("efficiency")
    @classmethod
    def validate_efficiency(cls, v: float) -> float:
        """Validate efficiency is reasonable for thermal plants."""
        if v < 0.1 or v > 0.6:
            raise ValueError("Thermal plant efficiency must be between 0.1 and 0.6")
        return v

    @field_validator("min_load_mw")
    @classmethod
    def validate_min_load(cls, v: float) -> float:
        """Validate minimum load is non-negative."""
        if v < 0.0:
            raise ValueError("Minimum load must be non-negative")
        return v

    def set_load_setpoint(self, load_mw: MW) -> None:
        """Set load setpoint for the plant.

        Args:
            load_mw: Target load in MW
        """
        self.load_setpoint_mw = max(0.0, min(load_mw, self.max_load_mw))

    def calculate_power_output(self) -> MW:
        """Calculate current power output based on setpoint and constraints.

        Returns:
            Power output in MW
        """
        if not self.is_online():
            return 0.0

        # Apply minimum load constraint
        if self.load_setpoint_mw > 0 and self.load_setpoint_mw < self.min_load_mw:
            return self.min_load_mw

        return min(self.load_setpoint_mw, self.max_load_mw)

    def tick(self, dt_seconds: float) -> float:
        """Update thermal plant state for one simulation timestep.

        Args:
            dt_seconds: Time delta in seconds since last tick

        Returns:
            Power output change in MW
        """
        if not self.is_online():
            old_output = self.current_output_mw
            self.current_output_mw = 0.0
            return self.current_output_mw - old_output

        # Calculate target power output
        target_power = self.calculate_power_output()
        old_output = self.current_output_mw

        # Apply ramping constraints
        max_ramp_mw = self.ramp_rate_mw_per_min * (dt_seconds / 60.0)

        if target_power > self.current_output_mw:
            # Ramping up
            self.current_output_mw = min(target_power, self.current_output_mw + max_ramp_mw)
        elif target_power < self.current_output_mw:
            # Ramping down
            self.current_output_mw = max(target_power, self.current_output_mw - max_ramp_mw)
        else:
            # Target equals current, no change needed
            self.current_output_mw = target_power

        return self.current_output_mw - old_output

    def calculate_fuel_cost_per_hour(self) -> float:
        """Calculate fuel cost per hour at current output.

        Returns:
            Fuel cost in $/hour
        """
        if self.current_output_mw == 0.0:
            return 0.0

        # Calculate fuel consumption:
        # Power (MW) * 1000 (kW/MW) * Heat Rate (BTU/kWh) * Fuel Cost ($/MMBTU) / 1,000,000 (BTU/MMBTU)
        fuel_cost_per_hour = (
            self.current_output_mw * 1000.0 * self.heat_rate_btu_per_kwh * self.fuel_cost_per_mmbtu / 1_000_000.0
        )

        return fuel_cost_per_hour

    def calculate_co2_emissions_per_hour(self) -> float:
        """Calculate CO2 emissions per hour at current output.

        Returns:
            CO2 emissions in lb/hour
        """
        if self.current_output_mw == 0.0:
            return 0.0

        # Calculate emissions:
        # Power (MW) * 1000 (kW/MW) * Heat Rate (BTU/kWh) * Emissions Factor (lb/MMBTU) / 1,000,000 (BTU/MMBTU)
        emissions_per_hour = (
            self.current_output_mw * 1000.0 * self.heat_rate_btu_per_kwh * self.co2_emissions_lb_per_mmbtu / 1_000_000.0
        )

        return emissions_per_hour

    def is_fossil_fuel(self) -> bool:
        """Check if plant uses fossil fuels.

        Returns:
            True if plant uses fossil fuels
        """
        return self.asset_type in [AssetType.COAL, AssetType.GAS]

    def is_renewable(self) -> bool:
        """Check if plant is renewable.

        Returns:
            False for all thermal plants
        """
        return False

    def get_state(self) -> dict[str, Any]:
        """Get comprehensive state information for the thermal plant.

        Returns:
            Dictionary containing complete thermal plant state
        """
        base_state = super().get_state()

        # Add thermal-specific state
        thermal_state = {
            "efficiency": self.efficiency,
            "min_load_mw": self.min_load_mw,
            "max_load_mw": self.max_load_mw,
            "ramp_rate_mw_per_min": self.ramp_rate_mw_per_min,
            "startup_time_minutes": self.startup_time_minutes,
            "shutdown_time_minutes": self.shutdown_time_minutes,
            "heat_rate_btu_per_kwh": self.heat_rate_btu_per_kwh,
            "fuel_cost_per_mmbtu": self.fuel_cost_per_mmbtu,
            "co2_emissions_lb_per_mmbtu": self.co2_emissions_lb_per_mmbtu,
            "load_setpoint_mw": self.load_setpoint_mw,
            "fuel_cost_per_hour": self.calculate_fuel_cost_per_hour(),
            "co2_emissions_per_hour": self.calculate_co2_emissions_per_hour(),
            "is_fossil_fuel": self.is_fossil_fuel(),
        }

        base_state.update(thermal_state)
        return base_state


class CoalPlant(ThermalPlant):
    """Coal-fired power plant asset."""

    def __init__(self, **data: Any) -> None:
        """Initialize coal plant with COAL asset type and typical characteristics."""
        # Set typical coal plant characteristics
        data.setdefault("efficiency", 0.35)
        data.setdefault("min_load_mw", data.get("capacity_mw", 100.0) * 0.3)  # 30% of capacity
        data.setdefault("ramp_rate_mw_per_min", 3.0)  # Slower ramping
        data.setdefault("startup_time_minutes", 480)  # 8 hours
        data.setdefault("shutdown_time_minutes", 120)  # 2 hours
        data.setdefault("heat_rate_btu_per_kwh", 9500.0)  # Higher heat rate
        data.setdefault("fuel_cost_per_mmbtu", 2.5)  # Lower fuel cost
        data.setdefault("co2_emissions_lb_per_mmbtu", 205.0)  # High CO2 emissions

        data["asset_type"] = AssetType.COAL
        super().__init__(**data)


class NaturalGasPlant(ThermalPlant):
    """Natural gas-fired power plant asset."""

    def __init__(self, **data: Any) -> None:
        """Initialize natural gas plant with GAS asset type and typical characteristics."""
        # Set typical natural gas plant characteristics
        data.setdefault("efficiency", 0.50)  # Higher efficiency
        data.setdefault("min_load_mw", data.get("capacity_mw", 100.0) * 0.2)  # 20% of capacity
        data.setdefault("ramp_rate_mw_per_min", 10.0)  # Faster ramping
        data.setdefault("startup_time_minutes", 30)  # Quick start
        data.setdefault("shutdown_time_minutes", 15)  # Quick shutdown
        data.setdefault("heat_rate_btu_per_kwh", 7000.0)  # Lower heat rate
        data.setdefault("fuel_cost_per_mmbtu", 4.0)  # Higher fuel cost
        data.setdefault("co2_emissions_lb_per_mmbtu", 117.0)  # Lower CO2 emissions

        data["asset_type"] = AssetType.GAS
        super().__init__(**data)


class NuclearPlant(ThermalPlant):
    """Nuclear power plant asset."""

    # Nuclear-specific parameters
    fuel_cost_per_mwh: float = Field(default=5.0, ge=0.1, le=50.0, description="Fuel cost in $/MWh")

    def __init__(self, **data: Any) -> None:
        """Initialize nuclear plant with NUCLEAR asset type and typical characteristics."""
        # Set typical nuclear plant characteristics
        data.setdefault("efficiency", 0.33)
        data.setdefault("min_load_mw", data.get("capacity_mw", 1000.0) * 0.7)  # 70% of capacity
        data.setdefault("ramp_rate_mw_per_min", 1.0)  # Very slow ramping
        data.setdefault("startup_time_minutes", 2880)  # 48 hours
        data.setdefault("shutdown_time_minutes", 720)  # 12 hours
        data.setdefault("heat_rate_btu_per_kwh", 10300.0)  # Nuclear heat rate
        data.setdefault("fuel_cost_per_mmbtu", 0.0)  # Use fuel_cost_per_mwh instead
        data.setdefault("co2_emissions_lb_per_mmbtu", 0.0)  # No CO2 emissions

        data["asset_type"] = AssetType.NUCLEAR
        super().__init__(**data)

    def calculate_fuel_cost_per_hour(self) -> float:
        """Calculate fuel cost per hour at current output.

        For nuclear plants, use fuel cost per MWh instead of BTU-based calculation.

        Returns:
            Fuel cost in $/hour
        """
        if self.current_output_mw == 0.0:
            return 0.0

        # Nuclear fuel cost is typically expressed per MWh
        fuel_cost_per_hour = self.current_output_mw * self.fuel_cost_per_mwh

        return fuel_cost_per_hour

    def calculate_co2_emissions_per_hour(self) -> float:
        """Calculate CO2 emissions per hour at current output.

        Nuclear plants have no CO2 emissions during operation.

        Returns:
            CO2 emissions in lb/hour (always 0 for nuclear)
        """
        return 0.0

    def is_fossil_fuel(self) -> bool:
        """Check if plant uses fossil fuels.

        Returns:
            False for nuclear plants
        """
        return False

    def get_state(self) -> dict[str, Any]:
        """Get comprehensive state information for the nuclear plant.

        Returns:
            Dictionary containing complete nuclear plant state
        """
        base_state = super().get_state()

        # Add nuclear-specific state
        nuclear_state = {
            "fuel_cost_per_mwh": self.fuel_cost_per_mwh,
        }

        base_state.update(nuclear_state)
        return base_state


# Add thermal plants to asset types if not already defined
try:
    from psireg.utils.enums import AssetType

    # Check if thermal types are already defined
    if not hasattr(AssetType, "COAL"):
        # This would require updating the enum, but since we can't modify enums at runtime,
        # we'll assume the types are already defined in the enum
        pass

except ImportError:
    logger.warning("Could not import AssetType enum - thermal asset types may not be available")
