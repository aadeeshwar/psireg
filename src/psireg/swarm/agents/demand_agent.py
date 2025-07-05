"""Demand/Load agent for swarm intelligence coordination.

This module provides the DemandAgent class that implements intelligent coordination
strategies for demand/load nodes in the distributed grid using swarm intelligence
principles like pheromone-based communication and demand response optimization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from psireg.sim.assets.load import Load
from psireg.utils.types import MW

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DemandSwarmState(BaseModel):
    """State information for demand swarm coordination."""

    agent_id: str = Field(..., description="Unique agent identifier")
    current_demand_mw: MW = Field(..., ge=0.0, description="Current demand in MW")
    baseline_demand_mw: MW = Field(..., ge=0.0, description="Baseline demand in MW")
    peak_demand_mw: MW = Field(..., ge=0.0, description="Peak demand in MW")
    dr_capability_mw: MW = Field(..., ge=0.0, description="Demand response capability in MW")
    price_elasticity: float = Field(..., ge=-2.0, le=2.0, description="Price elasticity of demand")
    current_price: float = Field(..., gt=0.0, description="Current electricity price $/MWh")
    flexibility_factor: float = Field(..., ge=0.0, le=1.0, description="Demand flexibility factor")
    grid_support_priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Priority for grid support")
    coordination_signal: float = Field(default=0.0, description="Swarm coordination signal strength")


class DemandAgent:
    """Demand/Load agent for swarm intelligence coordination.

    This agent implements intelligent control strategies for demand/load nodes
    including:
    - Demand response optimization
    - Load scheduling and shifting
    - Peak shaving coordination
    - Grid frequency support through demand modulation
    - Economic optimization based on price signals
    - Swarm coordination through pheromone-like signals
    """

    def __init__(
        self,
        load: Load,
        agent_id: str | None = None,
        communication_range: float = 5.0,
        response_time_s: float = 10.0,
        coordination_weight: float = 0.25,
    ):
        """Initialize demand agent.

        Args:
            load: Load asset to control
            agent_id: Unique agent identifier (defaults to load asset_id)
            communication_range: Communication range for swarm coordination
            response_time_s: Response time for control actions in seconds
            coordination_weight: Weight for coordination vs local optimization (0-1)
        """
        self.load = load
        self.agent_id = agent_id or load.asset_id
        self.communication_range = communication_range
        self.response_time_s = response_time_s
        self.coordination_weight = coordination_weight

        # Control parameters
        self.target_demand_factor: float = 1.0  # Target demand as factor of baseline
        self.demand_deadband_percent: float = 2.0  # Deadband for demand response
        self.frequency_deadband_hz: float = 0.05  # Frequency deadband for response
        self.max_frequency_response_rate: float = 0.05  # MW per 0.1 Hz

        # Swarm coordination
        self.pheromone_strength: float = 0.0
        self.local_grid_stress: float = 0.0
        self.coordination_signal: float = 0.0
        self.neighbor_signals: list[float] = []

        # Economic parameters
        self.comfort_cost: float = 200.0  # $/MWh for comfort sacrifice
        self.peak_avoidance_value: float = 100.0  # $/MWh for peak avoidance
        self.grid_service_value: float = 120.0  # $/MWh for grid services

        # Demand scheduling parameters
        self.flexible_load_fraction: float = 0.3  # Fraction of load that can be shifted
        self.max_shift_hours: int = 4  # Maximum hours to shift load
        self.scheduled_adjustments: dict[int, float] = {}  # Hour -> adjustment factor

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

        # Update electricity price in load asset
        if electricity_price is not None:
            self.load.set_electricity_price(electricity_price)

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
            self.coordination_signal = 0.6 * self.coordination_signal + 0.4 * avg_neighbor_signal
        else:
            # Decay signal when no neighbors
            self.coordination_signal *= 0.8

    def calculate_optimal_demand(
        self,
        forecast_prices: list[float],
        forecast_generation: list[float],
        forecast_grid_stress: list[float],
        time_horizon_hours: int = 24,
    ) -> MW:
        """Calculate optimal demand setpoint based on forecasts and objectives.

        Args:
            forecast_prices: Forecasted electricity prices in $/MWh
            forecast_generation: Forecasted generation in MW for next hours
            forecast_grid_stress: Forecasted grid stress levels (0-1)
            time_horizon_hours: Planning horizon in hours

        Returns:
            Optimal demand setpoint in MW
        """
        # Get current load characteristics
        baseline_demand = self.load.baseline_demand_mw
        dr_capability = self.load.dr_capability_mw

        # Multi-objective optimization weights
        frequency_weight = 0.3
        economic_weight = 0.4
        coordination_weight = self.coordination_weight
        comfort_weight = 0.3

        # 1. Frequency regulation objective
        frequency_response = self._calculate_frequency_response()

        # 2. Economic optimization objective
        economic_response = self._calculate_economic_response(forecast_prices)

        # 3. Swarm coordination objective
        coordination_response = self._calculate_coordination_response()

        # 4. Comfort/service quality objective
        comfort_response = self._calculate_comfort_response()

        # Combine objectives
        total_adjustment = (
            frequency_weight * frequency_response
            + economic_weight * economic_response
            + coordination_weight * coordination_response
            + comfort_weight * comfort_response
        )

        # Apply demand response limits
        max_reduction = min(dr_capability, baseline_demand * 0.5)  # Max 50% reduction
        max_increase = min(dr_capability, baseline_demand * 0.3)  # Max 30% increase

        # Clamp adjustment to capability limits
        total_adjustment = max(-max_reduction, min(total_adjustment, max_increase))

        # Calculate final optimal demand
        optimal_demand = baseline_demand + total_adjustment

        return max(0.0, min(optimal_demand, self.load.capacity_mw))

    def _calculate_frequency_response(self) -> MW:
        """Calculate demand adjustment for frequency regulation.

        Returns:
            Demand adjustment in MW for frequency support
        """
        # Use grid stress as frequency deviation proxy
        # Reduce demand when grid is stressed (high frequency deviation)
        frequency_response = -self.local_grid_stress * self.load.dr_capability_mw * 0.6
        return frequency_response

    def _calculate_economic_response(self, forecast_prices: list[float]) -> MW:
        """Calculate demand adjustment for economic optimization.

        Args:
            forecast_prices: Forecasted electricity prices

        Returns:
            Demand adjustment in MW for economic optimization
        """
        if not forecast_prices:
            return 0.0

        current_price = self.load.current_price
        avg_future_price = sum(forecast_prices) / len(forecast_prices)

        # Calculate price signal strength
        price_signal = (current_price - avg_future_price) / avg_future_price

        # Use price elasticity to determine demand response
        # Higher prices -> reduce demand, lower prices -> increase demand
        economic_response = price_signal * self.load.price_elasticity * self.load.baseline_demand_mw

        # Scale by economic sensitivity
        economic_response *= 0.5  # Moderate economic response

        return economic_response

    def _calculate_coordination_response(self) -> MW:
        """Calculate demand adjustment for swarm coordination.

        Returns:
            Demand adjustment in MW for swarm coordination
        """
        # Respond to coordination signal from swarm
        # Positive signal = reduce demand, negative signal = increase demand
        coordination_response = -self.coordination_signal * self.load.dr_capability_mw * 0.4
        return coordination_response

    def _calculate_comfort_response(self) -> MW:
        """Calculate demand adjustment for comfort/service quality.

        Returns:
            Demand adjustment in MW for comfort maintenance
        """
        # Bias toward maintaining baseline demand (comfort)
        current_demand = self.load.current_demand_mw
        baseline_demand = self.load.baseline_demand_mw

        # Calculate comfort deviation
        comfort_deviation = current_demand - baseline_demand

        # Apply restoring force toward baseline
        comfort_response = -comfort_deviation * 0.3

        return comfort_response

    def schedule_demand_shift(
        self,
        shift_mw: MW,
        from_hour: int,
        to_hour: int,
        duration_hours: int = 1,
    ) -> bool:
        """Schedule demand shift from one time period to another.

        Args:
            shift_mw: Amount of demand to shift in MW
            from_hour: Hour to shift demand from (0-23)
            to_hour: Hour to shift demand to (0-23)
            duration_hours: Duration of shift in hours

        Returns:
            True if shift was scheduled successfully
        """
        # Check if shift is within flexibility limits
        max_shift = self.load.baseline_demand_mw * self.flexible_load_fraction

        if abs(shift_mw) > max_shift:
            return False

        # Check if shift is within time limits
        if abs(to_hour - from_hour) > self.max_shift_hours:
            return False

        # Schedule the shift
        for h in range(duration_hours):
            from_h = (from_hour + h) % 24
            to_h = (to_hour + h) % 24

            # Reduce demand at from_hour
            self.scheduled_adjustments[from_h] = self.scheduled_adjustments.get(from_h, 0.0) - shift_mw

            # Increase demand at to_hour
            self.scheduled_adjustments[to_h] = self.scheduled_adjustments.get(to_h, 0.0) + shift_mw

        logger.info(f"Scheduled demand shift: {shift_mw:.1f} MW from hour {from_hour} to hour {to_hour}")
        return True

    def execute_control_action(self, optimal_demand: MW) -> None:
        """Execute control action to achieve optimal demand.

        Args:
            optimal_demand: Target demand setpoint in MW
        """
        # Calculate demand response signal
        baseline_demand = self.load.baseline_demand_mw
        dr_signal = optimal_demand - baseline_demand

        # Apply demand response signal
        self.load.set_demand_response_signal(dr_signal)

        # Update pheromone strength based on control action
        self.pheromone_strength = abs(dr_signal) / max(1.0, self.load.dr_capability_mw)

        logger.debug(f"Agent {self.agent_id} executing demand adjustment: {dr_signal:.2f} MW")

    def get_agent_state(self) -> DemandSwarmState:
        """Get current agent state for swarm coordination.

        Returns:
            Current demand swarm state
        """
        flexibility_factor = self.load.dr_capability_mw / max(1.0, self.load.baseline_demand_mw)

        return DemandSwarmState(
            agent_id=self.agent_id,
            current_demand_mw=self.load.current_demand_mw,
            baseline_demand_mw=self.load.baseline_demand_mw,
            peak_demand_mw=self.load.peak_demand_mw,
            dr_capability_mw=self.load.dr_capability_mw,
            price_elasticity=self.load.price_elasticity,
            current_price=self.load.current_price,
            flexibility_factor=flexibility_factor,
            grid_support_priority=self._calculate_grid_support_priority(),
            coordination_signal=self.coordination_signal,
        )

    def _calculate_grid_support_priority(self) -> float:
        """Calculate priority for grid support based on current conditions.

        Returns:
            Grid support priority (0.0 to 1.0)
        """
        # Higher priority when:
        # 1. High grid stress
        # 2. High demand response capability
        # 3. Low electricity price (cheap to provide support)

        stress_factor = self.local_grid_stress
        capability_factor = self.load.dr_capability_mw / max(1.0, self.load.baseline_demand_mw)
        price_factor = max(0.0, 1.0 - self.load.current_price / 100.0)  # Normalize to $100/MWh

        priority = 0.4 * stress_factor + 0.4 * capability_factor + 0.2 * price_factor

        return min(1.0, max(0.0, priority))

    def update_target_demand_factor(self, factor: float) -> None:
        """Update target demand factor.

        Args:
            factor: Target demand factor (1.0 = baseline)
        """
        self.target_demand_factor = max(0.2, min(factor, 1.8))  # Reasonable bounds

    def get_coordination_signal(self) -> float:
        """Get current coordination signal strength.

        Returns:
            Coordination signal strength (-1.0 to 1.0)
        """
        # Signal based on current demand relative to baseline
        baseline = self.load.baseline_demand_mw
        current = self.load.current_demand_mw

        if baseline > 0:
            signal = (current - baseline) / baseline
            return max(-1.0, min(1.0, signal))
        return 0.0

    def get_pheromone_strength(self) -> float:
        """Get current pheromone strength for swarm communication.

        Returns:
            Pheromone strength (0.0 to 1.0)
        """
        return self.pheromone_strength

    def reset(self) -> None:
        """Reset agent state to initial conditions."""
        self.pheromone_strength = 0.0
        self.local_grid_stress = 0.0
        self.coordination_signal = 0.0
        self.neighbor_signals = []
        self.scheduled_adjustments = {}
        self.target_demand_factor = 1.0

    def __str__(self) -> str:
        """String representation of demand agent."""
        return f"DemandAgent(id={self.agent_id}, load={self.load.baseline_demand_mw:.1f}MW)"

    def __repr__(self) -> str:
        """Detailed string representation of demand agent."""
        return (
            f"DemandAgent(agent_id='{self.agent_id}', "
            f"baseline_demand={self.load.baseline_demand_mw:.1f}MW, "
            f"dr_capability={self.load.dr_capability_mw:.1f}MW, "
            f"coordination_weight={self.coordination_weight:.2f})"
        )
