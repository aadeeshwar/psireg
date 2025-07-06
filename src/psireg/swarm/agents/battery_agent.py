"""Battery storage agent for swarm intelligence coordination.

This module provides the BatteryAgent class that implements intelligent coordination
strategies for battery storage systems in the distributed grid using swarm intelligence
principles like pheromone-based communication and local optimization.

Enhanced with voltage triggers, enhanced pheromone sensitivity, and local stabilization
signal generation as the primary output for grid stability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from psireg.sim.assets.battery import Battery
from psireg.swarm.pheromone import PheromoneType
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
    - Voltage trigger-based local stabilization
    - Peak shaving and load leveling
    - Renewable energy smoothing
    - Emergency grid support
    - Economic optimization
    - Enhanced swarm coordination through pheromone-like signals
    - Local stabilization as primary output
    """

    def __init__(
        self,
        battery: Battery,
        agent_id: str | None = None,
        communication_range: float = 5.0,
        response_time_s: float = 1.0,
        coordination_weight: float = 0.3,
        voltage_deadband_v: float = 10.0,
        voltage_trigger_sensitivity: float = 0.5,
        voltage_regulation_weight: float = 0.4,
    ):
        """Initialize battery storage agent.

        Args:
            battery: Battery asset to control
            agent_id: Unique agent identifier (defaults to battery asset_id)
            communication_range: Communication range for swarm coordination
            response_time_s: Response time for control actions in seconds
            coordination_weight: Weight for coordination vs local optimization (0-1)
            voltage_deadband_v: Voltage deadband in volts for trigger activation
            voltage_trigger_sensitivity: Voltage trigger sensitivity (0-1)
            voltage_regulation_weight: Weight for voltage regulation in optimization (0-1)
        """
        self.battery = battery
        self.agent_id = agent_id or battery.asset_id
        self.communication_range = communication_range
        self.response_time_s = response_time_s
        self.coordination_weight = coordination_weight

        # Voltage trigger parameters
        self.voltage_deadband_v = voltage_deadband_v
        self.voltage_trigger_sensitivity = voltage_trigger_sensitivity
        self.voltage_regulation_weight = voltage_regulation_weight
        self.nominal_voltage_v = battery.nominal_voltage_v
        self.current_voltage_kv = 0.8  # Default grid voltage in kV

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

        # Enhanced pheromone sensitivity
        self.pheromone_sensitivity_types: dict[PheromoneType, float] = {
            PheromoneType.FREQUENCY_SUPPORT: 0.8,
            PheromoneType.COORDINATION: 0.6,
            PheromoneType.RENEWABLE_CURTAILMENT: 0.4,
            PheromoneType.DEMAND_REDUCTION: 0.5,
            PheromoneType.EMERGENCY_RESPONSE: 0.9,
            PheromoneType.ECONOMIC_SIGNAL: 0.3,
        }
        self.pheromone_response_weights: dict[PheromoneType, float] = {
            PheromoneType.FREQUENCY_SUPPORT: 0.4,
            PheromoneType.COORDINATION: 0.3,
            PheromoneType.RENEWABLE_CURTAILMENT: 0.2,
            PheromoneType.DEMAND_REDUCTION: 0.1,
            PheromoneType.EMERGENCY_RESPONSE: 0.5,
            PheromoneType.ECONOMIC_SIGNAL: 0.15,
        }
        self.pheromone_decay_factor: float = 0.95
        self.pheromone_gradient_threshold: float = 0.1
        self.pheromone_memory: dict[PheromoneType, float] = {}
        self.pheromone_gradients: dict[PheromoneType, float] = {}

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

        Raises:
            ValueError: If frequency is negative
        """
        if frequency_hz < 0:
            raise ValueError("Frequency cannot be negative")

        # Store current voltage and frequency for calculations
        self.current_voltage_kv = voltage_kv
        self.current_frequency_hz = frequency_hz

        # Calculate frequency deviation
        nominal_frequency = 60.0  # Hz
        frequency_deviation = frequency_hz - nominal_frequency

        # Calculate local power imbalance
        power_imbalance = local_generation_mw - local_load_mw

        # Update local grid stress indicator
        frequency_stress = abs(frequency_deviation) / 0.5  # Normalize to 0.5 Hz range
        power_stress = abs(power_imbalance) / 100.0  # Normalize to 100 MW range
        voltage_stress = abs(self.calculate_voltage_deviation()) / (
            self.voltage_deadband_v * 3 / 1000.0
        )  # Normalize with 3x deadband

        self.local_grid_stress = min(1.0, max(frequency_stress, power_stress, voltage_stress))

        # Update electricity price
        if electricity_price is not None:
            self.electricity_price = electricity_price

    def calculate_voltage_deviation(self) -> float:
        """Calculate voltage deviation from nominal in kV.

        Returns:
            Voltage deviation in kV
        """
        nominal_kv = self.nominal_voltage_v / 1000.0  # Convert to kV
        return self.current_voltage_kv - nominal_kv

    def calculate_voltage_response(self) -> MW:
        """Calculate power response for voltage regulation.

        Returns:
            Power adjustment in MW for voltage support
        """
        voltage_deviation = self.calculate_voltage_deviation()
        deadband_kv = self.voltage_deadband_v / 1000.0  # Convert to kV

        # Check if within deadband
        if abs(voltage_deviation) <= deadband_kv:
            return 0.0

        # Calculate response magnitude
        deviation_magnitude = abs(voltage_deviation) - deadband_kv
        response_magnitude = deviation_magnitude * self.voltage_trigger_sensitivity

        # Scale by battery capacity
        max_response = self.battery.capacity_mw * 0.5  # Limit to 50% of capacity
        voltage_response = response_magnitude * max_response * 10  # Scale factor

        # Determine direction: high voltage -> charge (positive), low voltage -> discharge (negative)
        if voltage_deviation > deadband_kv:
            # High voltage - charge to absorb reactive power
            voltage_response = min(voltage_response, self.battery.get_max_charge_power())
        else:
            # Low voltage - discharge to provide reactive power
            voltage_response = -min(voltage_response, self.battery.get_max_discharge_power())

        return voltage_response

    def update_pheromone_gradients(self, gradients: dict[PheromoneType, float]) -> None:
        """Update pheromone gradients from swarm coordination.

        Args:
            gradients: Dictionary of pheromone type to gradient values
        """
        # Update memory with decay
        for pheromone_type in self.pheromone_memory:
            self.pheromone_memory[pheromone_type] *= self.pheromone_decay_factor

        # Update with new gradients
        for pheromone_type, gradient in gradients.items():
            self.pheromone_gradients[pheromone_type] = gradient
            self.pheromone_memory[pheromone_type] = gradient

    def calculate_pheromone_response(self, pheromone_type: PheromoneType) -> MW:
        """Calculate power response for specific pheromone type.

        Args:
            pheromone_type: Type of pheromone to respond to

        Returns:
            Power adjustment in MW for pheromone coordination
        """
        if pheromone_type not in self.pheromone_gradients:
            return 0.0

        gradient = self.pheromone_gradients[pheromone_type]

        # Check threshold
        if abs(gradient) < self.pheromone_gradient_threshold:
            return 0.0

        # Get sensitivity and response weight for this pheromone type
        sensitivity = self.pheromone_sensitivity_types.get(pheromone_type, 0.5)
        weight = self.pheromone_response_weights.get(pheromone_type, 0.3)

        # Calculate response based on pheromone type
        if pheromone_type == PheromoneType.FREQUENCY_SUPPORT:
            # Frequency support: provide grid frequency regulation
            response = gradient * sensitivity * weight * self.battery.capacity_mw
        elif pheromone_type == PheromoneType.EMERGENCY_RESPONSE:
            # Emergency response: immediate action required
            response = gradient * sensitivity * weight * self.battery.capacity_mw * 1.2
        elif pheromone_type == PheromoneType.COORDINATION:
            # Coordination: follow swarm behavior
            response = gradient * sensitivity * weight * self.battery.capacity_mw * 0.5
        elif pheromone_type == PheromoneType.RENEWABLE_CURTAILMENT:
            # Renewable curtailment: charge when renewables are curtailed
            response = gradient * sensitivity * weight * self.battery.capacity_mw * 0.3
        elif pheromone_type == PheromoneType.ECONOMIC_SIGNAL:
            # Economic signal: optimize based on price signals
            response = gradient * sensitivity * weight * self.battery.capacity_mw * 0.4
        else:
            # Default response (including DEMAND_REDUCTION)
            response = gradient * sensitivity * weight * self.battery.capacity_mw * 0.2

        return response

    def calculate_combined_pheromone_response(self) -> MW:
        """Calculate combined pheromone response from all types.

        Returns:
            Combined power adjustment in MW
        """
        total_response = 0.0

        for pheromone_type in self.pheromone_gradients:
            response = self.calculate_pheromone_response(pheromone_type)
            total_response += response

        # Limit to battery capacity
        max_charge = self.battery.get_max_charge_power()
        max_discharge = self.battery.get_max_discharge_power()

        if total_response > 0:
            total_response = min(total_response, max_charge)
        else:
            total_response = max(total_response, -max_discharge)

        return total_response

    def update_neighbor_pheromone_strengths(self, neighbor_strengths: dict[PheromoneType, list[float]]) -> None:
        """Update neighbor pheromone strength data.

        Args:
            neighbor_strengths: Dictionary mapping pheromone types to lists of neighbor strengths
        """
        # This would be used for gradient calculation in a real implementation
        # For now, store for testing
        self.neighbor_pheromone_strengths = neighbor_strengths

    def calculate_pheromone_gradients(self) -> dict[PheromoneType, float]:
        """Calculate pheromone gradients based on neighbor data.

        Returns:
            Dictionary of pheromone gradients
        """
        gradients = {}

        if hasattr(self, "neighbor_pheromone_strengths"):
            for pheromone_type, strengths in self.neighbor_pheromone_strengths.items():
                if strengths:
                    # Simple gradient calculation: difference from average
                    avg_strength = sum(strengths) / len(strengths)
                    current_strength = self.pheromone_memory.get(pheromone_type, 0.0)
                    gradient = avg_strength - current_strength
                    gradients[pheromone_type] = max(-1.0, min(1.0, gradient))

        return gradients

    def update_spatial_pheromone_data(self, neighbor_data: list[dict[str, Any]]) -> None:
        """Update spatial pheromone data with distance weighting.

        Args:
            neighbor_data: List of neighbor data with distance and pheromone information
        """
        self.spatial_pheromone_data = neighbor_data

    def calculate_spatial_pheromone_response(self) -> float:
        """Calculate distance-weighted pheromone response.

        Returns:
            Spatial pheromone response
        """
        if not hasattr(self, "spatial_pheromone_data"):
            return 0.0

        total_response = 0.0
        total_weight = 0.0

        for neighbor in self.spatial_pheromone_data:
            distance = neighbor["distance"]
            pheromones = neighbor["pheromone"]

            # Distance weighting (closer neighbors have more influence)
            weight = 1.0 / (1.0 + distance)

            for pheromone_type, strength in pheromones.items():
                response = self.calculate_pheromone_response(pheromone_type)
                total_response += response * weight * strength
                total_weight += weight

        if total_weight > 0:
            return total_response / total_weight
        return 0.0

    def calculate_local_stabilization_signal(self) -> dict[str, Any]:
        """Calculate local stabilization signal as primary output.

        Returns:
            Dictionary containing local stabilization signal components
        """
        # Calculate individual components
        frequency_support = self._calculate_frequency_response()
        voltage_support = self.calculate_voltage_response()
        pheromone_coordination = self.calculate_combined_pheromone_response()
        soc_management = self._calculate_soc_management_response()

        # Combine components with weights
        frequency_weight = 0.35
        voltage_weight = self.voltage_regulation_weight
        pheromone_weight = self.coordination_weight
        soc_weight = 0.15

        total_power = (
            frequency_weight * frequency_support
            + voltage_weight * voltage_support
            + pheromone_weight * pheromone_coordination
            + soc_weight * soc_management
        )

        # Apply power limits
        max_charge = self.battery.get_max_charge_power()
        max_discharge = self.battery.get_max_discharge_power()

        if total_power > 0:
            total_power = min(total_power, max_charge)
        else:
            total_power = max(total_power, -max_discharge)

        # Calculate confidence based on signal consistency
        confidence = self._calculate_stabilization_confidence(
            frequency_support, voltage_support, pheromone_coordination
        )

        # Calculate priority based on grid conditions
        priority = self._calculate_stabilization_priority()

        # Calculate response time based on urgency
        response_time = self._calculate_response_time()

        return {
            "power_mw": total_power,
            "voltage_support_mw": voltage_support,
            "frequency_support_mw": frequency_support,
            "pheromone_coordination_mw": pheromone_coordination,
            "confidence": confidence,
            "priority": priority,
            "response_time_s": response_time,
        }

    def _calculate_stabilization_confidence(
        self, frequency_support: float, voltage_support: float, pheromone_coordination: float
    ) -> float:
        """Calculate confidence in stabilization signal.

        Args:
            frequency_support: Frequency support component
            voltage_support: Voltage support component
            pheromone_coordination: Pheromone coordination component

        Returns:
            Confidence value (0.0 to 1.0)
        """
        # High confidence when signals are strong and consistent
        signals = [frequency_support, voltage_support, pheromone_coordination]
        signals = [s for s in signals if abs(s) > 0.001]  # Filter out near-zero signals

        if not signals:
            return 0.1  # Low confidence when no clear signals

        # Check signal consistency (same direction)
        positive_signals = sum(1 for s in signals if s > 0)
        negative_signals = sum(1 for s in signals if s < 0)

        consistency = max(positive_signals, negative_signals) / len(signals)

        # Higher confidence with stronger signals (normalized by capacity)
        signal_strength = min(1.0, sum(abs(s) for s in signals) / (len(signals) * self.battery.capacity_mw))

        # Calculate grid condition strength (higher for clearer deviations)
        voltage_deviation = abs(self.calculate_voltage_deviation())
        frequency_deviation = 0.0
        if hasattr(self, "current_frequency_hz"):
            frequency_deviation = abs(self.current_frequency_hz - 60.0)

        # Stronger grid conditions = higher confidence
        voltage_strength = min(1.0, voltage_deviation / (self.voltage_deadband_v * 2 / 1000.0))
        frequency_strength = min(1.0, frequency_deviation / 0.1)  # Normalize to 0.1 Hz
        grid_condition_strength = max(voltage_strength, frequency_strength)

        # Combine factors: consistency, signal strength, and grid condition clarity
        confidence = 0.2 + 0.3 * consistency + 0.2 * signal_strength + 0.3 * grid_condition_strength
        return min(1.0, max(0.0, confidence))

    def _calculate_stabilization_priority(self) -> float:
        """Calculate priority for stabilization actions.

        Returns:
            Priority value (0.0 to 1.0)
        """
        # Higher priority for larger deviations
        voltage_deviation = abs(self.calculate_voltage_deviation())
        voltage_priority = min(1.0, voltage_deviation / (self.voltage_deadband_v / 500.0))  # Normalize

        frequency_priority = self.local_grid_stress

        grid_priority = max(voltage_priority, frequency_priority)

        # Battery readiness factor
        soc_readiness = 1.0 - abs(self.battery.current_soc_percent - 50.0) / 50.0
        health_readiness = self.battery.current_health_percent / 100.0

        battery_readiness = 0.7 * soc_readiness + 0.3 * health_readiness

        priority = 0.7 * grid_priority + 0.3 * battery_readiness
        return min(1.0, max(0.0, priority))

    def _calculate_response_time(self) -> float:
        """Calculate response time for stabilization actions.

        Returns:
            Response time in seconds
        """
        # Faster response for emergency conditions
        voltage_deviation = abs(self.calculate_voltage_deviation())
        emergency_voltage_threshold = self.voltage_deadband_v * 8 / 1000.0  # 8x deadband for emergency
        fast_voltage_threshold = self.voltage_deadband_v * 3 / 1000.0  # 3x deadband for fast

        # Check frequency deviation for emergency
        frequency_deviation = 0.0
        if hasattr(self, "current_frequency_hz"):
            frequency_deviation = abs(self.current_frequency_hz - 60.0)

        # Emergency conditions: very high voltage/frequency deviation or very high grid stress
        if (
            voltage_deviation > emergency_voltage_threshold
            or frequency_deviation > 0.25
            or self.local_grid_stress > 0.9
        ):
            # Emergency response
            return 0.2
        elif voltage_deviation > fast_voltage_threshold or frequency_deviation > 0.1 or self.local_grid_stress > 0.6:
            # Fast response
            return 0.5
        else:
            # Normal response
            return self.response_time_s

    def execute_local_stabilization(self, stabilization_signal: dict[str, Any]) -> None:
        """Execute local stabilization actions.

        Args:
            stabilization_signal: Stabilization signal from calculate_local_stabilization_signal
        """
        power_setpoint = stabilization_signal["power_mw"]

        # Set battery power setpoint
        self.battery.set_power_setpoint(power_setpoint)

        # Update pheromone strength based on action
        action_magnitude = abs(power_setpoint) / self.battery.capacity_mw
        self.pheromone_strength = 0.8 * self.pheromone_strength + 0.2 * action_magnitude

        logger.debug(
            f"Agent {self.agent_id} executed stabilization: {power_setpoint:.2f} MW "
            f"(confidence: {stabilization_signal['confidence']:.2f}, "
            f"priority: {stabilization_signal['priority']:.2f})"
        )

    def get_enhanced_state(self) -> dict[str, Any]:
        """Get enhanced agent state including voltage and pheromone information.

        Returns:
            Enhanced state dictionary
        """
        base_state = self.get_agent_state().model_dump()

        enhanced_state = {
            "voltage_deviation_v": self.calculate_voltage_deviation() * 1000.0,  # Convert to V
            "pheromone_gradients": dict(self.pheromone_gradients),
            "pheromone_memory": dict(self.pheromone_memory),
            "stabilization_priority": self._calculate_stabilization_priority(),
            "local_grid_conditions": {
                "voltage_kv": self.current_voltage_kv,
                "grid_stress": self.local_grid_stress,
                "electricity_price": self.electricity_price,
            },
            "voltage_trigger_config": {
                "deadband_v": self.voltage_deadband_v,
                "sensitivity": self.voltage_trigger_sensitivity,
                "regulation_weight": self.voltage_regulation_weight,
            },
        }

        base_state.update(enhanced_state)
        return base_state

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
        voltage_weight = self.voltage_regulation_weight  # Use voltage regulation weight
        economic_weight = 0.2
        coordination_weight = self.coordination_weight
        soc_management_weight = 0.2

        # 1. Frequency regulation objective
        frequency_response = self._calculate_frequency_response()

        # 2. Voltage regulation objective
        voltage_response = self.calculate_voltage_response()

        # 3. Economic optimization objective
        economic_response = self._calculate_economic_response(forecast_prices)

        # 4. Swarm coordination objective
        coordination_response = self._calculate_coordination_response()

        # 5. SoC management objective
        soc_response = self._calculate_soc_management_response()

        # Combine objectives
        total_response = (
            frequency_weight * frequency_response
            + voltage_weight * voltage_response
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
        # Calculate frequency deviation from nominal
        nominal_frequency = 60.0  # Hz
        # Use stored frequency if available, otherwise use grid stress proxy
        if hasattr(self, "current_frequency_hz"):
            frequency_deviation = self.current_frequency_hz - nominal_frequency
        else:
            # Fallback to grid stress proxy (negative stress means low frequency)
            frequency_deviation = self.local_grid_stress * 0.5

        # Frequency response logic:
        # High frequency (>60 Hz) -> charge (positive power) to absorb energy
        # Low frequency (<60 Hz) -> discharge (negative power) to provide energy
        frequency_response = frequency_deviation * self.battery.capacity_mw * 0.5

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
        # Enhanced coordination using pheromone responses
        pheromone_response = self.calculate_combined_pheromone_response()

        # Traditional coordination signal
        basic_coordination = self.coordination_signal * self.battery.capacity_mw * 0.2

        # Combine both approaches
        coordination_response = 0.6 * pheromone_response + 0.4 * basic_coordination

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
        self.pheromone_memory.clear()
        self.pheromone_gradients.clear()

    # ===============================================
    # ENERGY REQUEST RESPONSE METHODS
    # ===============================================

    def detect_energy_requests(self, swarm_bus: Any) -> list[dict[str, Any]]:
        """Detect energy request pheromones in the neighborhood.

        Args:
            swarm_bus: Swarm bus for pheromone field access

        Returns:
            List of detected energy requests
        """
        detected_requests = []

        # Check for energy request pheromones in neighborhood
        for pheromone_type in [PheromoneType.ENERGY_REQUEST_HIGH, PheromoneType.ENERGY_REQUEST_NORMAL]:
            neighborhood_pheromones = swarm_bus.get_neighborhood_pheromones(
                agent_id=self.agent_id, pheromone_type=pheromone_type, radius=2
            )

            for position, strength in neighborhood_pheromones:
                if strength > 0.1:  # Minimum detectable strength
                    # Get agent positions to find requesting agent
                    nearby_agents = swarm_bus.get_agent_positions_in_radius(center=position, radius=1.0)

                    for agent_position, agent_id in nearby_agents:
                        if agent_position == position and agent_id != self.agent_id:
                            request = {
                                "agent_id": agent_id,
                                "position": position,
                                "pheromone_type": pheromone_type,
                                "strength": strength,
                                "distance": self._calculate_distance_to_position(position, swarm_bus),
                                # Mock some request details (in real system, this would come from agent communication)
                                "energy_needed_mw": strength * 20.0,  # Estimate based on strength
                                "max_price_mwh": 150.0,  # Default max price
                                "urgency": "high" if pheromone_type == PheromoneType.ENERGY_REQUEST_HIGH else "normal",
                            }
                            detected_requests.append(request)

        return detected_requests

    def calculate_energy_response(self, energy_requests: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate response to energy requests.

        Args:
            energy_requests: List of energy requests to respond to

        Returns:
            Energy response dictionary
        """
        if not energy_requests:
            return {
                "can_provide_mw": 0.0,
                "response_priority": 0.0,
                "estimated_cost_mwh": 0.0,
                "response_duration_hours": 0.0,
            }

        # Calculate available energy capacity
        current_soc = self.battery.current_soc_percent
        min_soc = 20.0  # Don't discharge below 20%
        available_energy_mwh = max(0.0, (current_soc - min_soc) / 100.0 * self.battery.energy_capacity_mwh)

        # Convert to power capacity (assume 1-hour discharge)
        max_discharge_power = self.battery.get_max_discharge_power()
        can_provide_mw = min(available_energy_mwh, max_discharge_power)

        # Calculate response priority based on multiple factors
        # 1. Battery readiness (SoC and health)
        soc_readiness = min(1.0, (current_soc - 50.0) / 50.0) if current_soc > 50.0 else 0.0
        health_readiness = self.battery.current_health_percent / 100.0

        # 2. Grid support value
        grid_stress_factor = self.local_grid_stress

        # 3. Economic incentive
        electricity_price = getattr(self, "electricity_price", 50.0)
        price_incentive = min(1.0, electricity_price / 100.0)  # Normalize to $100/MWh

        # 4. Request urgency (prioritize high urgency requests)
        urgency_factor = 0.9 if any(req["urgency"] == "high" for req in energy_requests) else 0.6

        # Combine factors
        response_priority = (
            0.3 * soc_readiness
            + 0.2 * health_readiness
            + 0.2 * grid_stress_factor
            + 0.2 * price_incentive
            + 0.1 * urgency_factor
        )

        # Calculate estimated cost
        base_cost = electricity_price
        service_premium = 20.0  # Premium for providing grid service
        urgency_premium = 30.0 if urgency_factor > 0.8 else 10.0
        estimated_cost = base_cost + service_premium + urgency_premium

        # Response duration based on available energy
        response_duration = min(4.0, available_energy_mwh / max(0.1, can_provide_mw))

        return {
            "agent_id": self.agent_id,
            "can_provide_mw": can_provide_mw,
            "response_priority": response_priority,
            "estimated_cost_mwh": estimated_cost,
            "response_duration_hours": response_duration,
        }

    def _calculate_distance_to_position(self, position: Any, swarm_bus: Any) -> float:
        """Calculate distance to a grid position.

        Args:
            position: Grid position
            swarm_bus: Swarm bus for position lookup

        Returns:
            Distance to position
        """
        # Get our position from swarm bus
        agent_info = swarm_bus.get_agent_info(self.agent_id)
        if agent_info and "position" in agent_info:
            our_position = agent_info["position"]
            dx = position.x - our_position.x
            dy = position.y - our_position.y
            return (dx * dx + dy * dy) ** 0.5
        return 0.0

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
