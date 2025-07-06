"""Solar panel agent for swarm intelligence coordination.

This module provides the SolarAgent class that implements intelligent coordination
strategies for solar generation assets using swarm intelligence principles with
PPO forecasting and pheromone-based communication to produce demand response signals.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

from psireg.sim.assets.solar import SolarPanel
from psireg.utils.enums import WeatherCondition
from psireg.utils.types import MW

if TYPE_CHECKING:
    from psireg.rl.predictive_layer import PredictiveLayer

logger = logging.getLogger(__name__)


class SolarSwarmState(BaseModel):
    """State information for solar swarm coordination."""

    agent_id: str = Field(..., description="Unique agent identifier")
    current_generation_mw: MW = Field(..., ge=0.0, description="Current generation in MW")
    capacity_mw: MW = Field(..., gt=0.0, description="Generation capacity in MW")
    panel_efficiency: float = Field(..., ge=0.0, le=1.0, description="Solar panel efficiency")
    current_irradiance_w_m2: float = Field(..., ge=0.0, description="Current solar irradiance in W/m²")
    current_temperature_c: float = Field(..., description="Current temperature in Celsius")
    current_curtailment_factor: float = Field(..., ge=0.0, le=1.0, description="Current curtailment factor")
    panel_area_m2: float = Field(..., gt=0.0, description="Solar panel area in m²")
    grid_support_priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Priority for grid support")
    coordination_signal: float = Field(default=0.0, description="Swarm coordination signal strength")


class SolarAgent:
    """Solar panel agent for swarm intelligence coordination.

    This agent implements intelligent coordination strategies for solar generation
    assets including:
    - PPO-based generation forecasting
    - Pheromone gradient coordination
    - Demand response signal generation through curtailment decisions
    - Grid frequency regulation through renewable curtailment
    - Economic optimization based on price signals
    - Swarm coordination through pheromone-like signals

    The primary output is demand response signals that influence grid load balancing.
    """

    def __init__(
        self,
        solar_panel: SolarPanel,
        agent_id: str | None = None,
        communication_range: float = 5.0,
        response_time_s: float = 2.0,
        coordination_weight: float = 0.3,
    ):
        """Initialize solar agent.

        Args:
            solar_panel: Solar panel asset to control
            agent_id: Unique agent identifier (defaults to solar_panel asset_id)
            communication_range: Communication range for swarm coordination
            response_time_s: Response time for control actions in seconds
            coordination_weight: Weight for coordination vs local optimization (0-1)
        """
        self.solar_panel = solar_panel
        self.agent_id = agent_id or solar_panel.asset_id
        self.communication_range = communication_range
        self.response_time_s = response_time_s
        self.coordination_weight = coordination_weight

        # Control parameters
        self.target_curtailment_factor: float = 0.0
        self.curtailment_deadband_percent: float = 2.0
        self.frequency_deadband_hz: float = 0.03
        self.max_frequency_response_rate: float = 0.08  # MW per 0.1 Hz

        # Swarm coordination
        self.pheromone_strength: float = 0.0
        self.local_grid_stress: float = 0.0
        self.coordination_signal: float = 0.0
        self.neighbor_signals: list[float] = []

        # Economic parameters
        self.electricity_price: float = 60.0  # $/MWh
        self.curtailment_cost: float = 80.0  # $/MWh for curtailed generation
        self.grid_service_value: float = 150.0  # $/MWh for grid services

        # PPO integration
        self.ppo_predictor: PredictiveLayer | None = None
        self.forecast_horizon_hours: int = 24
        self.forecast_cache: dict[str, list[float]] = {}
        self._cache_max_age_s: float = 300.0  # 5 minutes

        logger.debug(f"Initialized SolarAgent {self.agent_id}")

    def set_ppo_predictor(self, predictor: PredictiveLayer) -> None:
        """Set PPO predictor for generation forecasting.

        Args:
            predictor: Trained PPO predictor instance
        """
        self.ppo_predictor = predictor
        logger.info(f"SolarAgent {self.agent_id} connected to PPO predictor")

    def forecast_generation(
        self,
        grid_conditions: dict[str, Any],
        hours: int = 24,
    ) -> list[float]:
        """Forecast solar generation using PPO predictor.

        Args:
            grid_conditions: Current grid conditions for forecasting
            hours: Forecasting horizon in hours

        Returns:
            List of forecasted generation values in MW
        """
        # Create cache key from conditions
        cache_key = self._create_cache_key(grid_conditions, hours)

        # Check cache first
        if cache_key in self.forecast_cache:
            return self.forecast_cache[cache_key]

        if self.ppo_predictor is not None:
            forecast = self._forecast_with_ppo(grid_conditions, hours)
        else:
            forecast = self._forecast_fallback(grid_conditions, hours)

        # Cache the result
        self.forecast_cache[cache_key] = forecast

        # Limit cache size
        if len(self.forecast_cache) > 100:
            # Remove oldest entries
            oldest_key = next(iter(self.forecast_cache))
            del self.forecast_cache[oldest_key]

        return forecast

    def _forecast_with_ppo(
        self,
        grid_conditions: dict[str, Any],
        hours: int,
    ) -> list[float]:
        """Forecast generation using PPO predictor.

        Args:
            grid_conditions: Current grid conditions
            hours: Forecasting horizon in hours

        Returns:
            List of forecasted generation values in MW
        """
        try:
            # Prepare observation vector for PPO
            observation = self._prepare_ppo_observation(grid_conditions)

            # Get PPO prediction
            if self.ppo_predictor is not None:
                action = self.ppo_predictor.predict(observation, deterministic=True)
            else:
                return self._forecast_fallback(grid_conditions, hours)

            # Convert action to generation forecast
            forecast = self._action_to_generation_forecast(action, hours)

            return forecast

        except Exception as e:
            logger.warning(f"PPO forecasting failed for {self.agent_id}: {e}")
            return self._forecast_fallback(grid_conditions, hours)

    def _forecast_fallback(
        self,
        grid_conditions: dict[str, Any],
        hours: int,
    ) -> list[float]:
        """Fallback forecasting when PPO is not available.

        Args:
            grid_conditions: Current grid conditions
            hours: Forecasting horizon in hours

        Returns:
            List of forecasted generation values in MW
        """
        # Simple physics-based forecasting
        forecast = []

        irradiance = grid_conditions.get("irradiance_w_m2", 800.0)
        temperature = grid_conditions.get("temperature_c", 25.0)
        weather = grid_conditions.get("weather_condition", WeatherCondition.CLEAR)

        for hour in range(hours):
            # Simple hour-of-day irradiance model
            hour_factor = max(0.0, np.sin(np.pi * (hour % 24 - 6) / 12))

            # Adjust irradiance
            hour_irradiance = irradiance * hour_factor

            # Set panel conditions
            self.solar_panel.set_irradiance(hour_irradiance)
            self.solar_panel.set_temperature(temperature)
            self.solar_panel.set_weather_condition(weather)

            # Calculate generation
            generation = self.solar_panel.calculate_power_output()
            forecast.append(generation)

        return forecast

    def _prepare_ppo_observation(self, grid_conditions: dict[str, Any]) -> np.ndarray:
        """Prepare observation vector for PPO predictor.

        Args:
            grid_conditions: Current grid conditions

        Returns:
            Observation vector as numpy array
        """
        # Prepare normalized observation vector
        observations = [
            grid_conditions.get("frequency_hz", 60.0) / 60.0,  # Normalized frequency
            grid_conditions.get("voltage_kv", 138.0) / 138.0,  # Normalized voltage
            grid_conditions.get("irradiance_w_m2", 800.0) / 1000.0,  # Normalized irradiance
            grid_conditions.get("temperature_c", 25.0) / 50.0,  # Normalized temperature
            self.solar_panel.current_output_mw / self.solar_panel.capacity_mw,  # Current output ratio
            self.local_grid_stress,  # Grid stress level
            self.coordination_signal,  # Coordination signal
            self.solar_panel.curtailment_factor,  # Current curtailment
        ]

        return np.array(observations, dtype=np.float32)

    def _action_to_generation_forecast(self, action: np.ndarray, hours: int) -> list[float]:
        """Convert PPO action to generation forecast.

        Args:
            action: PPO action vector
            hours: Forecasting horizon in hours

        Returns:
            List of forecasted generation values in MW
        """
        # Map action values to generation forecast
        forecast = []

        # Use action values cyclically if action length < hours
        for hour in range(hours):
            action_idx = hour % len(action)
            action_value = action[action_idx]

            # Map action [-1, 1] to generation [0, capacity]
            # Consider this as curtailment factor adjustment
            curtailment_adjustment = (action_value + 1.0) / 2.0  # Map to [0, 1]

            # Calculate base generation potential
            base_generation = self.solar_panel.calculate_power_output()

            # Apply action as curtailment adjustment
            forecast_generation = base_generation * curtailment_adjustment
            forecast.append(min(forecast_generation, self.solar_panel.capacity_mw))

        return forecast

    def _create_cache_key(self, grid_conditions: dict[str, Any], hours: int) -> str:
        """Create cache key from grid conditions.

        Args:
            grid_conditions: Grid conditions
            hours: Forecasting horizon

        Returns:
            Cache key string
        """
        # Create deterministic hash from conditions
        key_data = str(sorted(grid_conditions.items())) + str(hours)
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def update_grid_conditions(
        self,
        frequency_hz: float,
        voltage_kv: float,
        local_load_mw: float,
        local_generation_mw: float,
        electricity_price: float | None = None,
        irradiance_w_m2: float | None = None,
        temperature_c: float | None = None,
        weather_condition: WeatherCondition | None = None,
    ) -> None:
        """Update current grid conditions for decision making.

        Args:
            frequency_hz: Current grid frequency in Hz
            voltage_kv: Current grid voltage in kV
            local_load_mw: Local load in MW
            local_generation_mw: Local generation in MW
            electricity_price: Current electricity price in $/MWh
            irradiance_w_m2: Current solar irradiance in W/m²
            temperature_c: Current temperature in Celsius
            weather_condition: Current weather condition
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

        # Update solar panel conditions
        if irradiance_w_m2 is not None:
            self.solar_panel.set_irradiance(irradiance_w_m2)

        if temperature_c is not None:
            self.solar_panel.set_temperature(temperature_c)

        if weather_condition is not None:
            self.solar_panel.set_weather_condition(weather_condition)

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

    def calculate_demand_response_signal(
        self,
        forecast_irradiance: list[float],
        forecast_prices: list[float],
        forecast_grid_stress: list[float],
        time_horizon_hours: int = 24,
    ) -> dict[str, Any]:
        """Calculate demand response signal based on forecasts and objectives.

        This is the primary output of the agent - demand response signals that
        influence grid load balancing through renewable curtailment decisions.

        Args:
            forecast_irradiance: Forecasted irradiance in W/m² for next hours
            forecast_prices: Forecasted electricity prices in $/MWh
            forecast_grid_stress: Forecasted grid stress levels (0-1)
            time_horizon_hours: Planning horizon in hours

        Returns:
            Dictionary containing demand response signal information
        """
        # Multi-objective optimization weights
        frequency_weight = 0.3
        economic_weight = 0.4
        coordination_weight = self.coordination_weight
        curtailment_weight = 0.3

        # 1. Frequency regulation objective
        frequency_response = self._calculate_frequency_response()

        # 2. Economic optimization objective
        economic_response = self._calculate_economic_response(forecast_prices)

        # 3. Swarm coordination objective
        coordination_response = self._calculate_coordination_response()

        # 4. Curtailment optimization objective
        curtailment_response = self._calculate_curtailment_response()

        # Combine objectives
        total_curtailment = (
            frequency_weight * frequency_response
            + economic_weight * economic_response
            + coordination_weight * coordination_response
            + curtailment_weight * curtailment_response
        )

        # Clamp curtailment to [0, 1] range
        total_curtailment = max(0.0, min(1.0, total_curtailment))

        # Calculate demand response signal based on curtailment
        # Higher curtailment signals demand increase (load should increase)
        # Lower curtailment signals demand decrease (load should decrease)
        potential_generation = self.solar_panel.calculate_power_output()

        # Curtailed generation represents available capacity for demand response
        curtailed_generation = potential_generation * total_curtailment

        # Signal magnitude based on curtailed capacity
        signal_mw = curtailed_generation * 0.5  # 50% of curtailed capacity as DR signal

        # Calculate confidence based on conditions
        confidence = self._calculate_confidence(forecast_irradiance, forecast_prices)

        # Generate reason for the signal
        reason = self._generate_signal_reason(
            frequency_response, economic_response, coordination_response, curtailment_response
        )

        return {
            "signal_mw": signal_mw,
            "curtailment_factor": total_curtailment,
            "confidence": confidence,
            "reason": reason,
            "frequency_component": frequency_response,
            "economic_component": economic_response,
            "coordination_component": coordination_response,
            "curtailment_component": curtailment_response,
        }

    def _calculate_frequency_response(self) -> float:
        """Calculate curtailment response for frequency regulation.

        Returns:
            Curtailment factor for frequency support (0.0 to 1.0)
        """
        # High grid stress (frequency deviation) should increase curtailment
        # to reduce generation and signal load increase
        frequency_response = self.local_grid_stress * 0.8
        return min(1.0, frequency_response)

    def _calculate_economic_response(self, forecast_prices: list[float]) -> float:
        """Calculate curtailment response for economic optimization.

        Args:
            forecast_prices: Forecasted electricity prices

        Returns:
            Curtailment factor for economic optimization (0.0 to 1.0)
        """
        if not forecast_prices:
            return 0.0

        current_price = self.electricity_price
        avg_future_price = sum(forecast_prices) / len(forecast_prices)

        # Low prices should encourage curtailment to signal demand increase
        # High prices should discourage curtailment to maximize revenue
        price_signal = (avg_future_price - current_price) / avg_future_price

        # Convert to curtailment factor [0, 1]
        economic_response = max(0.0, -price_signal * 0.5 + 0.3)
        return min(1.0, economic_response)

    def _calculate_coordination_response(self) -> float:
        """Calculate curtailment response for swarm coordination.

        Returns:
            Curtailment factor for swarm coordination (0.0 to 1.0)
        """
        # Positive coordination signal encourages curtailment
        # Negative coordination signal discourages curtailment
        coordination_response = max(0.0, self.coordination_signal * 0.4 + 0.2)
        return min(1.0, coordination_response)

    def _calculate_curtailment_response(self) -> float:
        """Calculate baseline curtailment response.

        Returns:
            Baseline curtailment factor (0.0 to 1.0)
        """
        # Base curtailment level considering current conditions
        current_curtailment = self.solar_panel.curtailment_factor

        # Target minimal curtailment under normal conditions
        target_curtailment = self.target_curtailment_factor

        # Smooth adjustment toward target
        curtailment_error = target_curtailment - (1.0 - current_curtailment)
        curtailment_response = max(0.0, curtailment_error * 0.5 + 0.1)

        return min(1.0, curtailment_response)

    def _calculate_confidence(
        self,
        forecast_irradiance: list[float],
        forecast_prices: list[float],
    ) -> float:
        """Calculate confidence in demand response signal.

        Args:
            forecast_irradiance: Forecasted irradiance values
            forecast_prices: Forecasted price values

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []

        # Irradiance variability factor
        if forecast_irradiance:
            irradiance_std = np.std(forecast_irradiance)
            irradiance_mean = np.mean(forecast_irradiance)
            if irradiance_mean > 0:
                irradiance_stability = 1.0 - min(1.0, float(irradiance_std / irradiance_mean))
                confidence_factors.append(irradiance_stability)

        # Price stability factor
        if forecast_prices:
            price_std = np.std(forecast_prices)
            price_mean = np.mean(forecast_prices)
            if price_mean > 0:
                price_stability = 1.0 - min(1.0, float(price_std / price_mean * 0.5))
                confidence_factors.append(price_stability)

        # Grid stress factor
        grid_stability = 1.0 - self.local_grid_stress
        confidence_factors.append(grid_stability)

        # Overall confidence
        if confidence_factors:
            confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            confidence = 0.5  # Default moderate confidence

        return max(0.0, min(1.0, confidence))

    def _generate_signal_reason(
        self,
        frequency_response: float,
        economic_response: float,
        coordination_response: float,
        curtailment_response: float,
    ) -> str:
        """Generate human-readable reason for demand response signal.

        Args:
            frequency_response: Frequency component
            economic_response: Economic component
            coordination_response: Coordination component
            curtailment_response: Curtailment component

        Returns:
            Reason string
        """
        components = []

        if frequency_response > 0.3:
            components.append("grid frequency regulation")

        if economic_response > 0.3:
            components.append("economic optimization")

        if coordination_response > 0.3:
            components.append("swarm coordination")

        if curtailment_response > 0.3:
            components.append("generation curtailment")

        if components:
            return f"Demand response triggered by: {', '.join(components)}"
        else:
            return "Minimal demand response required"

    def execute_control_action(self, generation_factor: float) -> None:
        """Execute the calculated control action.

        Args:
            generation_factor: Generation factor (0.0 to 1.0)
        """
        # Convert generation factor to curtailment factor
        curtailment_factor = 1.0 - generation_factor

        # Set solar panel curtailment
        self.solar_panel.set_curtailment_factor(1.0 - curtailment_factor)

        # Update pheromone strength based on action
        action_magnitude = abs(generation_factor - 0.5) * 2.0  # Normalize to [0, 1]
        self.pheromone_strength = 0.8 * self.pheromone_strength + 0.2 * action_magnitude

        logger.debug(f"Agent {self.agent_id} executing generation factor: {generation_factor:.2f}")

    def get_agent_state(self) -> SolarSwarmState:
        """Get current agent state for swarm coordination.

        Returns:
            Current agent state information
        """
        return SolarSwarmState(
            agent_id=self.agent_id,
            current_generation_mw=self.solar_panel.current_output_mw,
            capacity_mw=self.solar_panel.capacity_mw,
            panel_efficiency=self.solar_panel.panel_efficiency,
            current_irradiance_w_m2=self.solar_panel.current_irradiance_w_m2,
            current_temperature_c=self.solar_panel.current_temperature_c,
            current_curtailment_factor=self.solar_panel.curtailment_factor,
            panel_area_m2=self.solar_panel.panel_area_m2,
            grid_support_priority=self._calculate_grid_support_priority(),
            coordination_signal=self.coordination_signal,
        )

    def _calculate_grid_support_priority(self) -> float:
        """Calculate priority for providing grid support services.

        Returns:
            Priority value between 0.0 and 1.0
        """
        # Higher priority when:
        # - High irradiance (more generation potential)
        # - Low grid stress (stable conditions)
        # - Good weather conditions

        irradiance_factor = min(1.0, self.solar_panel.current_irradiance_w_m2 / 1000.0)
        stability_factor = 1.0 - self.local_grid_stress

        # Weather factor
        weather_factors = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.PARTLY_CLOUDY: 0.8,
            WeatherCondition.CLOUDY: 0.4,
            WeatherCondition.RAINY: 0.2,
            WeatherCondition.SNOWY: 0.1,
            WeatherCondition.STORMY: 0.0,
        }
        weather_factor = weather_factors.get(self.solar_panel.current_weather_condition, 0.5)

        priority = 0.4 * irradiance_factor + 0.3 * stability_factor + 0.3 * weather_factor

        return min(1.0, max(0.0, priority))

    def update_target_curtailment(self, target_curtailment_factor: float) -> None:
        """Update target curtailment factor.

        Args:
            target_curtailment_factor: Target curtailment factor (0.0 to 1.0)
        """
        self.target_curtailment_factor = max(0.0, min(1.0, target_curtailment_factor))

    def get_coordination_signal(self) -> float:
        """Get coordination signal for sharing with other agents.

        Returns:
            Coordination signal strength (-1.0 to 1.0)
        """
        # Signal based on current generation relative to capacity
        if self.solar_panel.capacity_mw > 0:
            generation_ratio = self.solar_panel.current_output_mw / self.solar_panel.capacity_mw
            signal = (generation_ratio - 0.5) * 2.0  # Map to [-1, 1]
            return max(-1.0, min(1.0, signal))
        return 0.0

    def get_pheromone_strength(self) -> float:
        """Get pheromone strength for swarm communication.

        Returns:
            Pheromone strength value (0.0 to 1.0)
        """
        return self.pheromone_strength

    def reset(self) -> None:
        """Reset agent to initial state."""
        self.target_curtailment_factor = 0.0
        self.pheromone_strength = 0.0
        self.local_grid_stress = 0.0
        self.coordination_signal = 0.0
        self.neighbor_signals.clear()
        self.forecast_cache.clear()

    def __str__(self) -> str:
        """String representation of the solar agent."""
        return (
            f"SolarAgent(id={self.agent_id}, "
            f"capacity={self.solar_panel.capacity_mw:.1f} MW, "
            f"generation={self.solar_panel.current_output_mw:.1f} MW, "
            f"signal={self.coordination_signal:.3f})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the solar agent."""
        return (
            f"SolarAgent(id={self.agent_id}, "
            f"solar_capacity={self.solar_panel.capacity_mw:.1f} MW, "
            f"efficiency={self.solar_panel.panel_efficiency:.2f}, "
            f"coordination_weight={self.coordination_weight:.2f})"
        )
