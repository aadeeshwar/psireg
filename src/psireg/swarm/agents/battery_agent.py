"""Battery storage agent for swarm intelligence coordination.

This module provides the BatteryAgent class that implements intelligent coordination
strategies for battery storage systems in the distributed grid using swarm intelligence
principles like pheromone-based communication and local optimization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from psireg.sim.assets.battery import Battery
from psireg.utils.types import MW, MWh

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BatterySwarmState(BaseModel):
    """State information for battery swarm coordination."""

    agent_id: str = Field(..., description="Unique agent identifier")
    soc_percent: float = Field(..., ge=0.0, le=100.0, description="Current SoC percentage")
    available_charge_power: MW = Field(..., ge=0.0, description="Available charging power in MW")
    available_discharge_power: MW = Field(..., ge=0.0, description="Available discharging power in MW")
    energy_capacity: MWh = Field(..., gt=0.0, description="Total energy capacity in MWh")
    health_percent: float = Field(..., ge=0.0, le=100.0, description="Battery health percentage")
    temperature_c: float = Field(..., description="Current temperature in Celsius")
    efficiency_factor: float = Field(..., ge=0.0, le=1.0, description="Current efficiency factor")
    grid_support_priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Priority for grid support")
    coordination_signal: float = Field(default=0.0, description="Swarm coordination signal strength")


class BatteryAgent:
    """Battery storage agent for swarm intelligence coordination.

    This agent implements intelligent control strategies for battery storage systems
    including:
    - Grid frequency regulation
    - Peak shaving and load leveling
    - Renewable energy smoothing
    - Emergency grid support
    - Economic optimization
    - Swarm coordination through pheromone-like signals
    """

    def __init__(
        self,
        battery: Battery,
        agent_id: str | None = None,
        communication_range: float = 5.0,
        response_time_s: float = 1.0,
        coordination_weight: float = 0.3,
    ):
        """Initialize battery storage agent.

        Args:
            battery: Battery asset to control
            agent_id: Unique agent identifier (defaults to battery asset_id)
            communication_range: Communication range for swarm coordination
            response_time_s: Response time for control actions in seconds
            coordination_weight: Weight for coordination vs local optimization (0-1)
        """
        self.battery = battery
        self.agent_id = agent_id or battery.asset_id
        self.communication_range = communication_range
        self.response_time_s = response_time_s
        self.coordination_weight = coordination_weight

        # Control parameters
        self.target_soc_percent: float = 50.0
        self.soc_deadband_percent: float = 5.0
        self.frequency_deadband_hz: float = 0.02
        self.max_frequency_response_rate: float = 0.1  # MW per 0.1 Hz

        # Swarm coordination
        self.pheromone_strength: float = 0.0
        self.local_grid_stress: float = 0.0
        self.coordination_signal: float = 0.0
        self.neighbor_signals: list[float] = []

        # Economic parameters
        self.electricity_price: float = 50.0  # $/MWh
        self.degradation_cost: float = 100.0  # $/MWh equivalent
        self.grid_service_value: float = 150.0  # $/MWh for grid services

    def update_grid_conditions(
        self,
        frequency_hz: float,
        voltage_kv: float,
        local_load_mw: float,
        local_generation_mw: float,
        electricity_price: float | None = None,
    ) -> None:
        """Update current grid conditions for decision making.

        Args:
            frequency_hz: Current grid frequency in Hz
            voltage_kv: Current grid voltage in kV
            local_load_mw: Local load in MW
            local_generation_mw: Local generation in MW
            electricity_price: Current electricity price in $/MWh
        """
        # Calculate frequency deviation
        nominal_frequency = 60.0  # Hz
        frequency_deviation = frequency_hz - nominal_frequency

        # Calculate local power imbalance
        power_imbalance = local_generation_mw - local_load_mw

        # Update local grid stress indicator
        frequency_stress = abs(frequency_deviation) / 0.5  # Normalize to 0.5 Hz range
        power_stress = abs(power_imbalance) / 100.0  # Normalize to 100 MW range
        self.local_grid_stress = min(1.0, max(frequency_stress, power_stress))

        # Update electricity price
        if electricity_price is not None:
            self.electricity_price = electricity_price

    def update_swarm_signals(self, neighbor_signals: list[float]) -> None:
        """Update swarm coordination signals from neighboring agents.

        Args:
            neighbor_signals: List of coordination signals from neighboring agents
        """
        self.neighbor_signals = neighbor_signals

        # Calculate coordination signal based on neighbors
        if neighbor_signals:
            avg_neighbor_signal = sum(neighbor_signals) / len(neighbor_signals)
            # Decay previous signal and incorporate neighbor influence
            self.coordination_signal = 0.7 * self.coordination_signal + 0.3 * avg_neighbor_signal
        else:
            # Decay signal when no neighbors
            self.coordination_signal *= 0.9

    def calculate_optimal_power(
        self,
        forecast_load: list[float],
        forecast_generation: list[float],
        forecast_prices: list[float],
        time_horizon_hours: int = 24,
    ) -> MW:
        """Calculate optimal power setpoint based on forecasts and objectives.

        Args:
            forecast_load: Forecasted load in MW for next hours
            forecast_generation: Forecasted generation in MW for next hours
            forecast_prices: Forecasted electricity prices in $/MWh
            time_horizon_hours: Planning horizon in hours

        Returns:
            Optimal power setpoint in MW (positive=charging, negative=discharging)
        """
        # Get current battery state
        available_charge = self.battery.get_max_charge_power()
        available_discharge = self.battery.get_max_discharge_power()

        # Multi-objective optimization weights
        frequency_weight = 0.4
        economic_weight = 0.3
        coordination_weight = self.coordination_weight
        soc_management_weight = 0.3

        # 1. Frequency regulation objective
        frequency_response = self._calculate_frequency_response()

        # 2. Economic optimization objective
        economic_response = self._calculate_economic_response(forecast_prices)

        # 3. Swarm coordination objective
        coordination_response = self._calculate_coordination_response()

        # 4. SoC management objective
        soc_response = self._calculate_soc_management_response()

        # Combine objectives
        total_response = (
            frequency_weight * frequency_response
            + economic_weight * economic_response
            + coordination_weight * coordination_response
            + soc_management_weight * soc_response
        )

        # Apply power limits
        if total_response > 0:
            # Charging
            optimal_power = min(total_response, available_charge)
        else:
            # Discharging
            optimal_power = max(total_response, -available_discharge)

        return optimal_power

    def _calculate_frequency_response(self) -> MW:
        """Calculate power response for frequency regulation.

        Returns:
            Power adjustment in MW for frequency support
        """
        # This would use actual grid frequency in a real implementation
        # For now, use local grid stress as a proxy
        frequency_response = -self.local_grid_stress * self.battery.capacity_mw * 0.5
        return frequency_response

    def _calculate_economic_response(self, forecast_prices: list[float]) -> MW:
        """Calculate power response for economic optimization.

        Args:
            forecast_prices: Forecasted electricity prices

        Returns:
            Power adjustment in MW for economic optimization
        """
        if not forecast_prices:
            return 0.0

        current_price = forecast_prices[0] if forecast_prices else self.electricity_price
        avg_future_price = sum(forecast_prices) / len(forecast_prices)

        # Simple economic strategy: charge when prices are low, discharge when high
        price_signal = (current_price - avg_future_price) / avg_future_price

        # Scale by battery capacity and economic sensitivity
        economic_response = -price_signal * self.battery.capacity_mw * 0.3

        return economic_response

    def _calculate_coordination_response(self) -> MW:
        """Calculate power response for swarm coordination.

        Returns:
            Power adjustment in MW for swarm coordination
        """
        # Respond to coordination signal from swarm
        coordination_response = self.coordination_signal * self.battery.capacity_mw * 0.2
        return coordination_response

    def _calculate_soc_management_response(self) -> MW:
        """Calculate power response for SoC management.

        Returns:
            Power adjustment in MW for SoC management
        """
        current_soc = self.battery.current_soc_percent
        soc_error = self.target_soc_percent - current_soc

        # Only respond if outside deadband
        if abs(soc_error) > self.soc_deadband_percent:
            # Proportional control for SoC management
            soc_response = soc_error * self.battery.capacity_mw * 0.01  # 1% capacity per 1% SoC error
            return soc_response

        return 0.0

    def execute_control_action(self, optimal_power: MW) -> None:
        """Execute the calculated optimal power setpoint.

        Args:
            optimal_power: Optimal power setpoint in MW
        """
        # Set battery power setpoint
        self.battery.set_power_setpoint(optimal_power)

        # Update pheromone strength based on action
        action_magnitude = abs(optimal_power) / self.battery.capacity_mw
        self.pheromone_strength = 0.8 * self.pheromone_strength + 0.2 * action_magnitude

    def get_agent_state(self) -> BatterySwarmState:
        """Get current agent state for swarm coordination.

        Returns:
            Current agent state information
        """
        return BatterySwarmState(
            agent_id=self.agent_id,
            soc_percent=self.battery.current_soc_percent,
            available_charge_power=self.battery.get_max_charge_power(),
            available_discharge_power=self.battery.get_max_discharge_power(),
            energy_capacity=self.battery.energy_capacity_mwh,
            health_percent=self.battery.current_health_percent,
            temperature_c=self.battery.current_temperature_c,
            efficiency_factor=self.battery.get_current_charge_efficiency(),
            grid_support_priority=self._calculate_grid_support_priority(),
            coordination_signal=self.coordination_signal,
        )

    def _calculate_grid_support_priority(self) -> float:
        """Calculate priority for providing grid support services.

        Returns:
            Priority value between 0.0 and 1.0
        """
        # Higher priority when:
        # - Battery is healthy
        # - SoC is in optimal range
        # - Temperature is acceptable
        # - Local grid stress is high

        health_factor = self.battery.current_health_percent / 100.0

        # SoC factor: highest priority around 50% SoC
        soc_factor = 1.0 - abs(self.battery.current_soc_percent - 50.0) / 50.0

        # Temperature factor: reduce priority at extreme temperatures
        temp_factor = max(0.3, 1.0 - abs(self.battery.current_temperature_c - 25.0) / 50.0)

        # Grid stress factor: higher priority when grid needs support
        stress_factor = self.local_grid_stress

        priority = 0.3 * health_factor + 0.3 * soc_factor + 0.2 * temp_factor + 0.2 * stress_factor

        return min(1.0, max(0.0, priority))

    def update_target_soc(self, target_soc_percent: float) -> None:
        """Update target SoC for the battery.

        Args:
            target_soc_percent: Target SoC percentage (0-100)
        """
        self.target_soc_percent = max(
            self.battery.min_soc_percent, min(target_soc_percent, self.battery.max_soc_percent)
        )

    def get_coordination_signal(self) -> float:
        """Get coordination signal for sharing with other agents.

        Returns:
            Coordination signal strength
        """
        return self.coordination_signal

    def get_pheromone_strength(self) -> float:
        """Get pheromone strength for swarm communication.

        Returns:
            Pheromone strength value
        """
        return self.pheromone_strength

    def reset(self) -> None:
        """Reset agent to initial state."""
        self.target_soc_percent = 50.0
        self.pheromone_strength = 0.0
        self.local_grid_stress = 0.0
        self.coordination_signal = 0.0
        self.neighbor_signals.clear()

    def __str__(self) -> str:
        """String representation of the battery agent."""
        return (
            f"BatteryAgent(id={self.agent_id}, "
            f"soc={self.battery.current_soc_percent:.1f}%, "
            f"power={self.battery.current_output_mw:.1f} MW, "
            f"signal={self.coordination_signal:.3f})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the battery agent."""
        return (
            f"BatteryAgent(id={self.agent_id}, "
            f"battery_capacity={self.battery.capacity_mw:.1f} MW, "
            f"soc={self.battery.current_soc_percent:.1f}%, "
            f"coordination_weight={self.coordination_weight:.2f})"
        )
