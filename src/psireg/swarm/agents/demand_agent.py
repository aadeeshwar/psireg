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
from psireg.swarm.pheromone import PheromoneType, SwarmBus
from psireg.utils.types import MW

if TYPE_CHECKING:
    from typing import Any

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

        # Energy request parameters
        self.pending_energy_requests: list[dict[str, Any]] = []  # Active energy requests
        self.received_energy_responses: list[dict[str, Any]] = []  # Responses from supply agents
        self.secured_energy_mw: float = 0.0  # Energy secured through requests
        self.energy_request_threshold: float = 0.6  # Grid stress threshold for requests
        self.max_request_price_multiplier: float = 2.0  # Max price willing to pay as multiple of current price

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
        power_stress = abs(power_imbalance) / 50.0  # Normalize to 50 MW range (more sensitive)
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
        self.pending_energy_requests = []
        self.received_energy_responses = []
        self.secured_energy_mw = 0.0

    # ===============================================
    # ENERGY REQUEST SYSTEM METHODS
    # ===============================================

    def broadcast_energy_request(
        self,
        energy_needed_mw: float,
        urgency: str,
        duration_hours: int,
        max_price_mwh: float,
    ) -> bool:
        """Broadcast energy request via pheromones.

        Args:
            energy_needed_mw: Amount of energy needed in MW
            urgency: Request urgency ("high", "normal", "low")
            duration_hours: Duration of energy need in hours
            max_price_mwh: Maximum price willing to pay in $/MWh

        Returns:
            True if request was broadcast successfully
        """
        if energy_needed_mw <= 0:
            return False

        # Create energy request
        request = {
            "energy_needed_mw": energy_needed_mw,
            "urgency": urgency,
            "duration_hours": duration_hours,
            "max_price_mwh": max_price_mwh,
            "timestamp": self.load.current_time if hasattr(self.load, "current_time") else None,
            "agent_id": self.agent_id,
        }

        # Add to pending requests
        self.pending_energy_requests.append(request)

        logger.info(
            f"Agent {self.agent_id} broadcast energy request: "
            f"{energy_needed_mw:.1f} MW, urgency={urgency}, max_price=${max_price_mwh:.0f}/MWh"
        )

        return True

    def calculate_request_pheromone_strength(
        self,
        energy_needed_mw: float,
        urgency: str,
        grid_stress: float,
    ) -> float:
        """Calculate pheromone strength for energy request.

        Args:
            energy_needed_mw: Amount of energy needed in MW
            urgency: Request urgency ("high", "normal", "low")
            grid_stress: Current grid stress level (0-1)

        Returns:
            Pheromone strength (0.0 to 1.0)
        """
        # Base strength from energy need relative to load capacity
        energy_ratio = energy_needed_mw / max(1.0, self.load.capacity_mw)
        base_strength = min(1.0, energy_ratio)

        # Urgency multiplier
        urgency_multipliers = {"high": 1.0, "normal": 0.7, "low": 0.4}
        urgency_factor = urgency_multipliers.get(urgency, 0.7)

        # Grid stress amplification
        stress_factor = 1.0 + grid_stress  # 1.0 to 2.0

        # Calculate final strength
        strength = base_strength * urgency_factor * stress_factor

        return min(1.0, max(0.0, strength))

    def should_request_energy(
        self,
        forecast_demand: list[float],
        forecast_generation: list[float],
        forecast_prices: list[float],
    ) -> bool:
        """Determine if energy should be requested based on forecasts.

        Args:
            forecast_demand: Forecasted demand in MW
            forecast_generation: Forecasted generation in MW
            forecast_prices: Forecasted prices in $/MWh

        Returns:
            True if energy should be requested
        """
        # Check current grid stress - if high stress, always consider requesting
        if self.local_grid_stress < self.energy_request_threshold:
            return False

        # If we have high current grid stress, request energy proactively
        if self.local_grid_stress > 0.8:
            return True

        # Check if there's an anticipated energy shortage
        if not forecast_demand or not forecast_generation:
            # If no forecast data but high grid stress, err on side of requesting
            return self.local_grid_stress > 0.7

        # Calculate average forecast shortage
        total_shortage = 0.0
        shortage_hours = 0

        for i in range(min(len(forecast_demand), len(forecast_generation))):
            shortage = forecast_demand[i] - forecast_generation[i]
            if shortage > 0:
                total_shortage += shortage
                shortage_hours += 1

        # Request energy if significant shortage is anticipated
        avg_shortage = total_shortage / max(1, len(forecast_demand))
        shortage_ratio = avg_shortage / max(1.0, self.load.baseline_demand_mw)

        # Check price trends (increasing prices indicate scarcity)
        price_trend_rising = False
        if len(forecast_prices) >= 2:
            price_trend = forecast_prices[-1] - forecast_prices[0]
            price_trend_rising = price_trend > 0

        # Request if shortage is significant or prices are rising rapidly or high current stress
        return (
            shortage_ratio > 0.1
            or (price_trend_rising and shortage_ratio > 0.05)
            or (price_trend_rising and self.local_grid_stress > 0.6)  # Proactive on price rises with stress
        )

    def calculate_energy_request(
        self,
        forecast_demand: list[float],
        forecast_generation: list[float],
        forecast_prices: list[float],
    ) -> dict[str, Any]:
        """Calculate energy request details.

        Args:
            forecast_demand: Forecasted demand in MW
            forecast_generation: Forecasted generation in MW
            forecast_prices: Forecasted prices in $/MWh

        Returns:
            Dictionary with energy request details
        """
        # Calculate energy shortage
        total_shortage = 0.0
        max_shortage = 0.0

        for i in range(min(len(forecast_demand), len(forecast_generation))):
            shortage = max(0, forecast_demand[i] - forecast_generation[i])
            total_shortage += shortage
            max_shortage = max(max_shortage, shortage)

        # Calculate energy needed (base it on our own demand and shortage severity)
        if total_shortage > 0:
            # Calculate as a fraction of our DR capability based on shortage severity
            shortage_severity = min(1.0, total_shortage / max(1.0, sum(forecast_demand)))
            energy_needed = shortage_severity * self.load.dr_capability_mw * 0.8  # Use 80% of DR capability
        else:
            # If no forecast shortage, consider current grid stress and other factors
            if self.local_grid_stress > 0.8:
                # High stress: request significant energy
                energy_needed = self.local_grid_stress * self.load.dr_capability_mw * 0.6
            elif self.local_grid_stress > 0.3:
                # Moderate stress: request smaller amount proactively
                energy_needed = self.local_grid_stress * self.load.dr_capability_mw * 0.3
            else:
                energy_needed = 0.0

        # Ensure minimum meaningful request
        if energy_needed > 0 and energy_needed < self.load.dr_capability_mw * 0.1:
            energy_needed = self.load.dr_capability_mw * 0.1  # At least 10% of DR capability

        # Determine urgency based on grid stress and shortage severity
        if self.local_grid_stress > 0.8 or max_shortage > self.load.baseline_demand_mw:
            urgency = "high"
        elif self.local_grid_stress > 0.6 or max_shortage > self.load.baseline_demand_mw * 0.5:
            urgency = "normal"
        else:
            urgency = "low"

        # Duration based on forecast length
        duration_hours = min(len(forecast_demand), 4)  # Max 4 hours

        # Maximum price willing to pay
        current_price = getattr(self.load, "current_price", 50.0)
        max_price = current_price * self.max_request_price_multiplier

        return {
            "energy_needed_mw": energy_needed,
            "urgency": urgency,
            "duration_hours": duration_hours,
            "max_price_mwh": max_price,
        }

    def process_energy_responses(self, energy_responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process energy responses and create matches.

        Args:
            energy_responses: List of energy responses from supply agents

        Returns:
            List of matched energy transactions
        """
        matches = []
        remaining_requests = self.pending_energy_requests.copy()

        for response in energy_responses:
            if not remaining_requests:
                break

            # Find best matching request
            best_request = None
            best_score = -1.0

            for request in remaining_requests:
                # Calculate match score
                energy_match = min(response["can_provide_mw"], request["energy_needed_mw"]) / max(
                    1.0, request["energy_needed_mw"]
                )

                price_acceptable = response["estimated_cost_mwh"] <= request["max_price_mwh"]

                if price_acceptable:
                    score = energy_match * response["response_priority"]
                    if score > best_score:
                        best_score = score
                        best_request = request

            # Create match if suitable request found
            if best_request and best_score > 0.3:  # Minimum match threshold
                agreed_energy = min(response["can_provide_mw"], best_request["energy_needed_mw"])

                match = {
                    "supplier_agent_id": response["agent_id"],
                    "agreed_energy_mw": agreed_energy,
                    "agreed_price_mwh": response["estimated_cost_mwh"],
                    "duration_hours": min(response["response_duration_hours"], best_request["duration_hours"]),
                    "urgency": best_request["urgency"],
                }

                matches.append(match)
                remaining_requests.remove(best_request)

                # Update secured energy
                self.secured_energy_mw += agreed_energy

        return matches

    def execute_energy_coordination(
        self,
        swarm_bus: SwarmBus | None,
        forecast_demand: list[float],
        forecast_generation: list[float],
        forecast_prices: list[float],
    ) -> dict[str, Any]:
        """Execute complete energy coordination protocol.

        Args:
            swarm_bus: Swarm bus for coordination (None for standalone)
            forecast_demand: Forecasted demand in MW
            forecast_generation: Forecasted generation in MW
            forecast_prices: Forecasted prices in $/MWh

        Returns:
            Coordination result dictionary
        """
        requests_sent = 0
        responses_received = 0
        energy_secured = 0.0

        # Check if energy request is needed
        if self.should_request_energy(forecast_demand, forecast_generation, forecast_prices):
            # Calculate energy request
            request_details = self.calculate_energy_request(forecast_demand, forecast_generation, forecast_prices)

            # Broadcast energy request
            if self.broadcast_energy_request(**request_details):
                requests_sent = 1

                # Deposit pheromone if swarm bus is available
                if swarm_bus:
                    pheromone_type = (
                        PheromoneType.ENERGY_REQUEST_HIGH
                        if request_details["urgency"] == "high"
                        else PheromoneType.ENERGY_REQUEST_NORMAL
                    )

                    strength = self.calculate_request_pheromone_strength(
                        request_details["energy_needed_mw"],
                        request_details["urgency"],
                        self.local_grid_stress,
                    )

                    swarm_bus.deposit_pheromone(
                        agent_id=self.agent_id, pheromone_type=pheromone_type, strength=strength
                    )

        # Calculate demand response incorporating secured energy
        demand_response = self._calculate_demand_response_with_secured_energy()

        return {
            "requests_sent": requests_sent,
            "responses_received": responses_received,
            "energy_secured_mw": energy_secured,
            "demand_response_mw": demand_response,
        }

    def calculate_enhanced_demand_response(
        self,
        forecast_demand: list[float],
        forecast_generation: list[float],
        forecast_prices: list[float],
        secured_energy_mw: float = 0.0,
    ) -> dict[str, Any]:
        """Calculate enhanced demand response incorporating energy request logic.

        Args:
            forecast_demand: Forecasted demand in MW
            forecast_generation: Forecasted generation in MW
            forecast_prices: Forecasted prices in $/MWh
            secured_energy_mw: Energy secured through requests in MW

        Returns:
            Enhanced demand response dictionary
        """
        # Calculate traditional demand response
        traditional_response = (
            self.calculate_optimal_demand(
                forecast_prices, forecast_generation, [self.local_grid_stress] * len(forecast_prices)
            )
            - self.load.baseline_demand_mw
        )

        # Calculate request-based adjustment (reduction in demand response)
        request_based_adjustment = 0.0
        if secured_energy_mw > 0:
            # If energy is secured, calculate reduction factor for demand response
            # The secured energy provides a buffer, so less DR is needed
            energy_security_factor = min(1.0, secured_energy_mw / self.load.dr_capability_mw)
            request_based_adjustment = -(traditional_response * energy_security_factor * 0.5)  # Negative = reduction

        # Calculate total response (traditional + request-based adjustment)
        # Note: request_based_adjustment is negative when energy is secured (reduction)
        total_response = traditional_response + request_based_adjustment

        # Calculate confidence based on grid conditions and energy security
        grid_condition_clarity = abs(self.local_grid_stress - 0.5) * 2  # 0 to 1
        energy_security_factor = min(1.0, secured_energy_mw / max(1.0, self.load.dr_capability_mw))
        confidence = (grid_condition_clarity + energy_security_factor) / 2

        return {
            "demand_adjustment_mw": self.load.baseline_demand_mw + total_response,
            "request_based_adjustment_mw": request_based_adjustment,
            "traditional_response_mw": traditional_response,
            "total_response_mw": total_response,
            "confidence": confidence,
        }

    def generate_primary_demand_response(
        self,
        forecast_demand: list[float],
        forecast_generation: list[float],
        forecast_prices: list[float],
        swarm_bus: SwarmBus | None = None,
    ) -> dict[str, Any]:
        """Generate primary demand response output incorporating energy request logic.

        This is the main method that produces the primary output as requested:
        "Primary output is: Demand response"

        Args:
            forecast_demand: Forecasted demand in MW
            forecast_generation: Forecasted generation in MW
            forecast_prices: Forecasted prices in $/MWh
            swarm_bus: Optional swarm bus for coordination

        Returns:
            Primary demand response output dictionary
        """
        # Execute energy coordination if swarm bus is available
        coordination_result = self.execute_energy_coordination(
            swarm_bus, forecast_demand, forecast_generation, forecast_prices
        )

        # Calculate enhanced demand response
        enhanced_response = self.calculate_enhanced_demand_response(
            forecast_demand, forecast_generation, forecast_prices, self.secured_energy_mw
        )

        # Generate coordination signals
        coordination_signals = {
            "energy_request_signal": self.pheromone_strength,
            "grid_support_signal": self.get_coordination_signal(),
            "local_stress_signal": self.local_grid_stress,
        }

        # Calculate action priority
        action_priority = self._calculate_action_priority()

        # PRIMARY OUTPUT: Demand Response
        primary_output = {
            "demand_response_mw": enhanced_response["total_response_mw"],  # PRIMARY OUTPUT
            "energy_requests_sent": coordination_result["requests_sent"],
            "energy_secured_mw": coordination_result["energy_secured_mw"],
            "coordination_signals": coordination_signals,
            "response_confidence": enhanced_response["confidence"],
            "action_priority": action_priority,
        }

        return primary_output

    def _calculate_demand_response_with_secured_energy(self) -> float:
        """Calculate demand response accounting for secured energy."""
        # Base demand response
        base_response = self._calculate_frequency_response()

        # Adjustment for secured energy
        if self.secured_energy_mw > 0:
            # Can reduce demand response if energy is secured
            secured_energy_factor = min(1.0, self.secured_energy_mw / self.load.dr_capability_mw)
            base_response *= 1.0 - secured_energy_factor * 0.5

        return base_response

    def _calculate_action_priority(self) -> float:
        """Calculate action priority for demand response.

        Returns:
            Priority value (0.0 to 1.0)
        """
        # Higher priority for:
        # 1. High grid stress
        # 2. High electricity prices
        # 3. Low secured energy

        stress_factor = self.local_grid_stress
        price_factor = min(1.0, getattr(self.load, "current_price", 50.0) / 100.0)  # Normalize to $100/MWh
        energy_security_factor = 1.0 - min(1.0, self.secured_energy_mw / max(1.0, self.load.dr_capability_mw))

        priority = (stress_factor + price_factor + energy_security_factor) / 3

        return min(1.0, max(0.0, priority))

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
