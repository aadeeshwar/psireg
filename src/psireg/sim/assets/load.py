"""Load/Demand Node asset implementation for PSIREG simulation system.

This module provides the Load class that models electrical load/demand with:
- Stochastic demand profile generation with realistic patterns
- Trace-driven demand profiles from CSV/data files
- Time-of-use patterns with peak/off-peak behavior
- Demand response capabilities with price elasticity
- Seasonal variations and weather effects
- Integration with GridEngine simulation
"""

from __future__ import annotations

import csv
import logging
import math
import random
from datetime import datetime, timedelta
from typing import Any

from pydantic import Field, ValidationInfo, field_validator

from psireg.sim.assets.base import Asset
from psireg.utils.enums import AssetType
from psireg.utils.types import MW

logger = logging.getLogger(__name__)


class Load(Asset):
    """Load/Demand Node asset for electrical demand modeling.

    Models electrical load/demand with comprehensive functionality including:
    - Stochastic demand profile generation
    - Trace-driven demand profiles from data files
    - Time-of-use patterns with peak/off-peak behavior
    - Demand response capabilities with price elasticity
    - Seasonal variations and weather effects
    - Integration with grid simulation engine
    """

    # Basic demand parameters
    baseline_demand_mw: MW = Field(..., ge=0.0, description="Baseline demand in MW")
    current_demand_mw: MW = Field(default=0.0, ge=0.0, description="Current demand in MW")
    min_demand_mw: MW = Field(default=0.0, ge=0.0, description="Minimum demand in MW")
    max_demand_mw: MW = Field(default=0.0, ge=0.0, description="Maximum demand in MW")

    # Time-of-use parameters
    peak_demand_mw: MW = Field(default=0.0, ge=0.0, description="Peak demand in MW")
    off_peak_demand_mw: MW = Field(default=0.0, ge=0.0, description="Off-peak demand in MW")
    peak_hours_start: int = Field(default=8, ge=0, le=23, description="Peak hours start (0-23)")
    peak_hours_end: int = Field(default=20, ge=0, le=23, description="Peak hours end (0-23)")

    # Stochastic profile parameters
    demand_volatility: float = Field(default=0.1, ge=0.0, le=1.0, description="Demand volatility factor (0.0-1.0)")
    profile_type: str = Field(
        default="stochastic", description="Profile type: 'stochastic', 'trace_driven', or 'time_of_use'"
    )

    # Trace-driven profile parameters
    trace_file_path: str | None = Field(default=None, description="Path to CSV trace file")
    trace_data: list[tuple[datetime, float]] = Field(
        default_factory=list, description="Loaded trace data (timestamp, demand_mw)"
    )

    # Demand response parameters
    dr_capability_mw: MW = Field(default=0.0, ge=0.0, description="Demand response capability in MW")
    dr_response_rate: float = Field(default=0.8, ge=0.0, le=1.0, description="DR response rate (0.0-1.0)")
    dr_signal_mw: MW = Field(default=0.0, description="Current DR signal in MW")
    price_elasticity: float = Field(default=0.0, ge=-2.0, le=2.0, description="Price elasticity of demand")
    baseline_price: float = Field(default=50.0, gt=0.0, description="Baseline electricity price $/MWh")
    current_price: float = Field(default=50.0, gt=0.0, description="Current electricity price $/MWh")

    # Seasonal variation parameters
    seasonal_factor_winter: float = Field(default=1.0, ge=0.0, le=2.0, description="Winter demand factor")
    seasonal_factor_spring: float = Field(default=0.9, ge=0.0, le=2.0, description="Spring demand factor")
    seasonal_factor_summer: float = Field(default=1.1, ge=0.0, le=2.0, description="Summer demand factor")
    seasonal_factor_fall: float = Field(default=0.95, ge=0.0, le=2.0, description="Fall demand factor")

    # Current time tracking
    current_time: datetime = Field(default_factory=lambda: datetime.now(), description="Current simulation time")

    def __init__(self, **data: Any) -> None:
        """Initialize load with LOAD asset type."""
        data["asset_type"] = AssetType.LOAD
        super().__init__(**data)

        # Set default max_demand_mw to capacity if not specified
        if self.max_demand_mw == 0.0:
            self.max_demand_mw = self.capacity_mw

        # Set default peak/off-peak demands if not specified
        if self.peak_demand_mw == 0.0:
            self.peak_demand_mw = min(self.baseline_demand_mw * 1.25, self.capacity_mw)
        if self.off_peak_demand_mw == 0.0:
            self.off_peak_demand_mw = self.baseline_demand_mw * 0.75

    @field_validator("baseline_demand_mw")
    @classmethod
    def validate_baseline_demand(cls, v: float) -> float:
        """Validate baseline demand is non-negative."""
        if v < 0:
            raise ValueError("Baseline demand must be non-negative")
        return v

    @field_validator("peak_demand_mw")
    @classmethod
    def validate_peak_demand(cls, v: float, info: ValidationInfo) -> float:
        """Validate peak demand doesn't exceed capacity."""
        if info.data and "capacity_mw" in info.data:
            capacity = info.data["capacity_mw"]
            if v > capacity:
                raise ValueError(f"Peak demand ({v}) cannot exceed capacity ({capacity})")
        return v

    @field_validator("peak_hours_start", "peak_hours_end")
    @classmethod
    def validate_peak_hours(cls, v: int) -> int:
        """Validate peak hours are within 0-23 range."""
        if v < 0 or v > 23:
            raise ValueError("Peak hours must be between 0 and 23")
        return v

    @field_validator("demand_volatility")
    @classmethod
    def validate_demand_volatility(cls, v: float) -> float:
        """Validate demand volatility is within reasonable range."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Demand volatility must be between 0.0 and 1.0")
        return v

    def set_current_time(self, time: datetime) -> None:
        """Set current simulation time.

        Args:
            time: Current simulation time
        """
        self.current_time = time

    def set_demand_response_signal(self, signal_mw: MW) -> None:
        """Set demand response signal.

        Args:
            signal_mw: DR signal in MW (positive=increase, negative=decrease)
        """
        self.dr_signal_mw = signal_mw

    def set_electricity_price(self, price: float) -> None:
        """Set current electricity price.

        Args:
            price: Current electricity price in $/MWh
        """
        self.current_price = max(0.1, price)  # Minimum price of $0.1/MWh

    def is_peak_hour(self, hour: int) -> bool:
        """Check if given hour is within peak hours.

        Args:
            hour: Hour of day (0-23)

        Returns:
            True if hour is within peak hours
        """
        if self.peak_hours_start <= self.peak_hours_end:
            return self.peak_hours_start <= hour < self.peak_hours_end
        else:
            # Handle case where peak hours span midnight
            return hour >= self.peak_hours_start or hour < self.peak_hours_end

    def get_seasonal_factor(self) -> float:
        """Get seasonal demand factor based on current time.

        Returns:
            Seasonal demand factor
        """
        month = self.current_time.month

        if month in [12, 1, 2]:  # Winter
            return self.seasonal_factor_winter
        elif month in [3, 4, 5]:  # Spring
            return self.seasonal_factor_spring
        elif month in [6, 7, 8]:  # Summer
            return self.seasonal_factor_summer
        else:  # Fall (9, 10, 11)
            return self.seasonal_factor_fall

    def calculate_time_of_use_demand(self) -> MW:
        """Calculate demand based on time-of-use patterns.

        Returns:
            Time-of-use demand in MW
        """
        hour = self.current_time.hour

        if self.is_peak_hour(hour):
            base_demand = self.peak_demand_mw
        else:
            base_demand = self.off_peak_demand_mw

        # Apply seasonal factor
        seasonal_factor = self.get_seasonal_factor()
        demand = base_demand * seasonal_factor

        return min(demand, self.max_demand_mw)

    def _generate_demand_noise(self) -> float:
        """Generate demand noise factor.

        Returns:
            Noise factor (typically around 1.0)
        """
        if self.demand_volatility == 0.0:
            return 1.0

        # Generate normally distributed noise
        noise = random.gauss(0, self.demand_volatility)
        # Clamp to reasonable bounds
        noise = max(-0.5, min(0.5, noise))
        return 1.0 + noise

    def generate_stochastic_profile(self, hours: int = 24, timestep_minutes: int = 15) -> list[MW]:
        """Generate stochastic demand profile.

        Args:
            hours: Number of hours to generate
            timestep_minutes: Timestep in minutes

        Returns:
            List of demand values in MW
        """
        profile = []
        steps = (hours * 60) // timestep_minutes

        for i in range(steps):
            # Calculate time for this step
            time_offset = timedelta(minutes=i * timestep_minutes)
            step_time = self.current_time + time_offset

            # Get base demand based on time-of-use
            hour = step_time.hour
            if self.is_peak_hour(hour):
                base_demand = self.peak_demand_mw
            else:
                base_demand = self.off_peak_demand_mw

            # Apply seasonal factor
            seasonal_factor = self.get_seasonal_factor()
            base_demand *= seasonal_factor

            # Add daily profile curve (sinusoidal)
            hour_factor = 0.5 * (1 + math.sin(2 * math.pi * (hour - 6) / 24))
            daily_factor = 0.7 + 0.3 * hour_factor

            # Apply daily factor
            base_demand *= daily_factor

            # Add stochastic noise
            noise_factor = self._generate_demand_noise()
            demand = base_demand * noise_factor

            # Apply limits
            demand = max(self.min_demand_mw, min(demand, self.max_demand_mw))

            profile.append(demand)

        return profile

    def load_trace_data(self) -> None:
        """Load trace data from CSV file.

        Raises:
            FileNotFoundError: If trace file doesn't exist
            ValueError: If trace file format is invalid
        """
        if not self.trace_file_path:
            raise ValueError("Trace file path not specified")

        try:
            with open(self.trace_file_path, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                self.trace_data = []

                for row in reader:
                    timestamp_str = row["timestamp"]
                    demand_mw = float(row["demand_mw"])

                    # Parse timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    self.trace_data.append((timestamp, demand_mw))

                # Sort by timestamp
                self.trace_data.sort(key=lambda x: x[0])
                logger.info(f"Loaded {len(self.trace_data)} trace data points from {self.trace_file_path}")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Trace file not found: {self.trace_file_path}") from e
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid trace file format: {e}") from e

    def calculate_demand_from_trace(self) -> MW:
        """Calculate demand from trace data with interpolation.

        Returns:
            Interpolated demand in MW
        """
        if not self.trace_data:
            return self.baseline_demand_mw

        # Find surrounding data points
        before_point = None
        after_point = None

        for timestamp, demand in self.trace_data:
            if timestamp <= self.current_time:
                before_point = (timestamp, demand)
            elif timestamp > self.current_time:
                after_point = (timestamp, demand)
                break

        # Handle edge cases
        if before_point is None:
            # Current time is before all trace data
            return self.baseline_demand_mw
        elif after_point is None:
            # Current time is after all trace data
            return self.baseline_demand_mw

        # Linear interpolation
        before_time, before_demand = before_point
        after_time, after_demand = after_point

        if before_time == after_time:
            return before_demand

        # Calculate interpolation factor
        total_seconds = (after_time - before_time).total_seconds()
        elapsed_seconds = (self.current_time - before_time).total_seconds()
        factor = elapsed_seconds / total_seconds

        # Interpolate demand
        interpolated_demand = before_demand + factor * (after_demand - before_demand)

        # Apply limits and return as MW
        return max(self.min_demand_mw, min(interpolated_demand, self.max_demand_mw))

    def calculate_demand_at_time(self) -> MW:
        """Calculate demand at current time based on profile type.

        Returns:
            Demand in MW
        """
        if self.profile_type == "trace_driven":
            return self.calculate_demand_from_trace()
        elif self.profile_type == "time_of_use":
            return self.calculate_time_of_use_demand()
        else:  # stochastic
            base_demand = self.calculate_time_of_use_demand()
            noise_factor = self._generate_demand_noise()
            demand = base_demand * noise_factor
            # Clamp to capacity limits
            return max(self.min_demand_mw, min(demand, self.capacity_mw))

    def calculate_price_response_demand(self, price: float) -> MW:
        """Calculate demand response to price changes.

        Args:
            price: Current electricity price in $/MWh

        Returns:
            Price-responsive demand in MW
        """
        if self.price_elasticity == 0.0:
            return self.baseline_demand_mw

        # Calculate price ratio
        price_ratio = price / self.baseline_price

        # Calculate demand response using elasticity
        # Demand change = elasticity * price change
        demand_factor = price_ratio**self.price_elasticity
        responsive_demand = self.baseline_demand_mw * demand_factor

        return max(self.min_demand_mw, min(responsive_demand, self.max_demand_mw))

    def calculate_demand_response(self) -> MW:
        """Calculate demand response to grid signals.

        Returns:
            Demand response in MW
        """
        if self.dr_capability_mw == 0.0:
            return 0.0

        # Apply response rate
        response = self.dr_signal_mw * self.dr_response_rate

        # Limit to capability
        response = max(-self.dr_capability_mw, min(response, self.dr_capability_mw))

        return response

    def calculate_final_demand(self) -> MW:
        """Calculate final demand considering all factors.

        Returns:
            Final demand in MW
        """
        # Base demand from profile
        base_demand = self.calculate_demand_at_time()

        # Apply price response
        if self.price_elasticity != 0.0:
            base_demand = self.calculate_price_response_demand(self.current_price)

        # Apply demand response
        dr_response = self.calculate_demand_response()
        final_demand = base_demand + dr_response

        # Apply limits
        final_demand = max(self.min_demand_mw, min(final_demand, self.max_demand_mw))

        return final_demand

    def calculate_power_output(self) -> MW:
        """Calculate current power output (negative for loads).

        Returns:
            Power output in MW (negative for loads)
        """
        if not self.is_online():
            return 0.0

        # Calculate demand and convert to negative power output
        self.current_demand_mw = self.calculate_final_demand()
        return -self.current_demand_mw

    def tick(self, dt_seconds: float) -> float:
        """Update load state for one simulation timestep.

        Args:
            dt_seconds: Time delta in seconds since last tick

        Returns:
            Power output change in MW
        """
        if not self.is_online():
            self.current_output_mw = 0.0
            self.current_demand_mw = 0.0
            return 0.0

        # Update current time
        self.current_time += timedelta(seconds=dt_seconds)

        # Calculate new power output
        new_power = self.calculate_power_output()
        power_change = new_power - self.current_output_mw
        self.current_output_mw = new_power

        return power_change

    def get_state(self) -> dict[str, Any]:
        """Get comprehensive state information for the load.

        Returns:
            Dictionary containing complete load state
        """
        base_state = super().get_state()

        # Add load-specific state
        load_state = {
            "baseline_demand_mw": self.baseline_demand_mw,
            "current_demand_mw": self.current_demand_mw,
            "peak_demand_mw": self.peak_demand_mw,
            "off_peak_demand_mw": self.off_peak_demand_mw,
            "demand_volatility": self.demand_volatility,
            "profile_type": self.profile_type,
            "dr_capability_mw": self.dr_capability_mw,
            "dr_signal_mw": self.dr_signal_mw,
            "price_elasticity": self.price_elasticity,
            "current_price": self.current_price,
            "seasonal_factor": self.get_seasonal_factor(),
            "is_peak_hour": self.is_peak_hour(self.current_time.hour),
            "demand_response_mw": self.calculate_demand_response(),
        }

        # Merge states
        base_state.update(load_state)
        return base_state

    def __str__(self) -> str:
        """String representation of load asset."""
        return (
            f"Load(id={self.asset_id}, "
            f"baseline={self.baseline_demand_mw:.1f}MW, "
            f"current={self.current_demand_mw:.1f}MW, "
            f"type={self.profile_type})"
        )
