"""Rule-based controller for renewable energy grid control.

This module implements traditional power system control strategies using
established rules and heuristics for grid stability and optimization.
"""

import time
from datetime import datetime
from typing import Any

from psireg.controllers.base import BaseController
from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.assets.wind import WindTurbine
from psireg.sim.engine import GridEngine, GridState
from psireg.utils.enums import AssetType
from psireg.utils.logger import logger


class RuleBasedController(BaseController):
    """Rule-based controller using traditional power system control logic.

    This controller implements established control strategies including:
    - Frequency regulation via droop control
    - Battery state-of-charge (SOC) management
    - Demand response for load balancing
    - Renewable curtailment for over-generation
    - Priority-based control action coordination

    The controller uses deterministic rules based on grid conditions,
    asset states, and pre-defined thresholds to maintain grid stability
    and optimize performance.
    """

    def __init__(self):
        """Initialize rule-based controller."""
        super().__init__()
        self.controller_type = "rule"

        # Control parameters
        self.frequency_deadband_hz = 0.05  # Frequency deadband for regulation
        self.droop_factor = 0.5  # MW per 0.1 Hz deviation
        self.soc_target_percent = 50.0  # Target SOC for batteries
        self.soc_deadband_percent = 10.0  # SOC deadband
        self.max_curtailment_factor = 0.3  # Maximum renewable curtailment
        self.dr_response_factor = 0.8  # Demand response aggressiveness

        # Rule weights for priority handling
        self.rule_weights = {
            "frequency_regulation": 1.0,  # Highest priority
            "emergency_response": 0.9,
            "soc_management": 0.6,
            "demand_response": 0.7,
            "renewable_curtailment": 0.5,
            "economic_optimization": 0.3,  # Lowest priority
        }

        # Internal state tracking
        self.last_grid_state: GridState | None = None
        self.last_frequency_hz = 60.0
        self.frequency_deviation_history: list[float] = []
        self.control_actions_history: list[dict[str, Any]] = []

        logger.info("Rule-based controller initialized")

    def initialize(self, grid_engine: GridEngine) -> bool:
        """Initialize rule-based controller with grid engine.

        Args:
            grid_engine: Grid simulation engine to control

        Returns:
            True if initialization successful
        """
        try:
            self.grid_engine = grid_engine
            self.initialized = True

            # Log controllable assets
            controllable_assets = self.get_controllable_assets()
            logger.info(f"Rule controller initialized with {len(controllable_assets)} controllable assets")

            for asset_id, asset in controllable_assets.items():
                logger.debug(f"Controllable asset: {asset_id} ({asset.asset_type.value})")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize rule controller: {e}")
            return False

    def update(self, grid_state: GridState, dt: float) -> None:
        """Update controller state based on current grid conditions.

        Args:
            grid_state: Current grid state
            dt: Time step duration in seconds
        """
        if not self.is_initialized():
            logger.warning("Controller not initialized")
            return

        if not self._validate_grid_state(grid_state):
            logger.warning("Invalid grid state provided")
            return

        # Update internal state
        self.last_grid_state = grid_state
        self.last_update_time = datetime.now()

        # Track frequency deviation
        frequency_deviation = abs(grid_state.frequency_hz - 60.0)
        self.frequency_deviation_history.append(frequency_deviation)

        # Keep only recent history (last 100 entries)
        if len(self.frequency_deviation_history) > 100:
            self.frequency_deviation_history = self.frequency_deviation_history[-100:]

        self.last_frequency_hz = grid_state.frequency_hz

        logger.debug(
            f"Rule controller updated: f={grid_state.frequency_hz:.3f} Hz, "
            f"balance={grid_state.power_balance_mw:.1f} MW"
        )

    def get_control_actions(self) -> dict[str, dict[str, float]]:
        """Calculate control actions using rule-based logic.

        Returns:
            Dictionary mapping asset IDs to control actions
        """
        if not self.is_initialized() or self.last_grid_state is None:
            return {}

        start_time = time.time()
        actions = {}

        try:
            # Get controllable assets
            controllable_assets = self.get_controllable_assets()

            # Apply control rules in priority order
            for asset_id, asset in controllable_assets.items():
                asset_actions = self._calculate_asset_actions(asset, self.last_grid_state)
                if asset_actions:
                    actions[asset_id] = asset_actions

            # Update control actions count
            self.control_actions_count += len(actions)

            # Record action history
            action_record = {
                "timestamp": datetime.now(),
                "frequency_hz": self.last_grid_state.frequency_hz,
                "power_balance_mw": self.last_grid_state.power_balance_mw,
                "actions_count": len(actions),
                "response_time_s": time.time() - start_time,
            }
            self.control_actions_history.append(action_record)

            # Keep only recent history
            if len(self.control_actions_history) > 1000:
                self.control_actions_history = self.control_actions_history[-1000:]

            logger.debug(f"Rule controller generated {len(actions)} control actions")

        except Exception as e:
            logger.error(f"Error calculating control actions: {e}")

        return actions

    def _calculate_asset_actions(self, asset: Any, grid_state: GridState) -> dict[str, float]:
        """Calculate control actions for a specific asset.

        Args:
            asset: Asset to control
            grid_state: Current grid state

        Returns:
            Dictionary of control actions for the asset
        """
        if asset.asset_type == AssetType.BATTERY:
            return self._calculate_battery_actions(asset, grid_state)
        elif asset.asset_type == AssetType.LOAD:
            return self._calculate_load_actions(asset, grid_state)
        elif asset.asset_type == AssetType.SOLAR:
            return self._calculate_solar_actions(asset, grid_state)
        elif asset.asset_type == AssetType.WIND:
            return self._calculate_wind_actions(asset, grid_state)
        else:
            logger.warning(f"Unknown asset type for control: {asset.asset_type}")
            return {}

    def _calculate_battery_actions(self, battery: Battery, grid_state: GridState) -> dict[str, float]:
        """Calculate battery control actions using rule-based logic.

        Args:
            battery: Battery asset to control
            grid_state: Current grid state

        Returns:
            Dictionary containing battery control actions
        """
        actions = {}

        # Rule 1: Frequency regulation (highest priority)
        frequency_response = self._calculate_frequency_response(battery, grid_state)

        # Rule 2: SOC management
        soc_response = self._calculate_soc_management(battery)

        # Rule 3: Economic optimization (if no critical needs)
        economic_response = self._calculate_economic_response(battery, grid_state)

        # Combine responses with priority weights
        total_response = (
            self.rule_weights["frequency_regulation"] * frequency_response
            + self.rule_weights["soc_management"] * soc_response
            + self.rule_weights["economic_optimization"] * economic_response
        )

        # Apply physical limits
        max_charge = battery.get_max_charge_power()
        max_discharge = battery.get_max_discharge_power()

        if total_response > 0:
            # Charging
            power_setpoint = min(total_response, max_charge)
        else:
            # Discharging
            power_setpoint = max(total_response, -max_discharge)

        # Only command if significant action needed
        if abs(power_setpoint) > 0.1:  # 0.1 MW threshold
            actions["power_setpoint_mw"] = power_setpoint

        return actions

    def _calculate_frequency_response(self, battery: Battery, grid_state: GridState) -> float:
        """Calculate battery frequency regulation response.

        Args:
            battery: Battery asset
            grid_state: Current grid state

        Returns:
            Power response in MW (positive = charge, negative = discharge)
        """
        frequency_deviation = grid_state.frequency_hz - 60.0

        # Only respond if outside deadband
        if abs(frequency_deviation) < self.frequency_deadband_hz:
            return 0.0

        # Droop control: frequency high -> charge, frequency low -> discharge
        droop_response = frequency_deviation * self.droop_factor * battery.capacity_mw

        # Consider SOC limits for frequency response
        soc = battery.current_soc_percent
        if droop_response > 0 and soc > 90:  # Limit charging when SOC high
            droop_response *= 0.3
        elif droop_response < 0 and soc < 10:  # Limit discharging when SOC low
            droop_response *= 0.3

        return droop_response

    def _calculate_soc_management(self, battery: Battery) -> float:
        """Calculate battery SOC management response.

        Args:
            battery: Battery asset

        Returns:
            Power response in MW for SOC management
        """
        soc = battery.current_soc_percent
        soc_error = self.soc_target_percent - soc

        # Only respond if outside deadband
        if abs(soc_error) < self.soc_deadband_percent:
            return 0.0

        # Proportional control for SOC management
        # 1% SOC error = 1% of battery capacity
        soc_response = soc_error * battery.capacity_mw * 0.01

        return soc_response

    def _calculate_economic_response(self, battery: Battery, grid_state: GridState) -> float:
        """Calculate battery economic optimization response.

        Args:
            battery: Battery asset
            grid_state: Current grid state

        Returns:
            Power response in MW for economic optimization
        """
        # Simple economic rule: discharge during high demand, charge during low demand
        # This is a placeholder - real implementation would use price signals

        load_factor = grid_state.total_load_mw / max(grid_state.total_generation_mw, 1.0)

        if load_factor > 1.05:  # High demand
            return -battery.capacity_mw * 0.2  # Discharge
        elif load_factor < 0.95:  # Low demand
            return battery.capacity_mw * 0.2  # Charge

        return 0.0

    def _calculate_load_actions(self, load: Load, grid_state: GridState) -> dict[str, float]:
        """Calculate load control actions using demand response rules.

        Args:
            load: Load asset to control
            grid_state: Current grid state

        Returns:
            Dictionary containing load control actions
        """
        actions = {}

        # Rule 1: Frequency-based demand response
        frequency_deviation = grid_state.frequency_hz - 60.0

        # Rule 2: Power balance-based demand response
        power_imbalance = grid_state.power_balance_mw

        # Calculate demand response signal
        dr_signal = 0.0

        # Frequency response: reduce demand when frequency is low
        if frequency_deviation < -self.frequency_deadband_hz:
            # Low frequency -> reduce demand
            freq_response = abs(frequency_deviation) * self.dr_response_factor * load.dr_capability_mw
            dr_signal -= freq_response

        # Power imbalance response: reduce demand when generation is insufficient
        if power_imbalance < -5.0:  # 5 MW deficit threshold
            imbalance_response = min(abs(power_imbalance) * 0.1, load.dr_capability_mw * 0.5)
            dr_signal -= imbalance_response

        # Apply DR capability limits
        dr_signal = max(-load.dr_capability_mw, min(dr_signal, load.dr_capability_mw))

        # Only command if significant action needed
        if abs(dr_signal) > 0.5:  # 0.5 MW threshold
            actions["dr_signal_mw"] = dr_signal

        return actions

    def _calculate_solar_actions(self, solar: SolarPanel, grid_state: GridState) -> dict[str, float]:
        """Calculate solar panel control actions using curtailment rules.

        Args:
            solar: Solar panel asset
            grid_state: Current grid state

        Returns:
            Dictionary containing solar control actions
        """
        actions = {}

        # Rule: Curtail solar when over-generation occurs
        frequency_deviation = grid_state.frequency_hz - 60.0
        power_balance = grid_state.power_balance_mw

        curtailment_factor = 0.0

        # High frequency indicates over-generation
        if frequency_deviation > self.frequency_deadband_hz:
            freq_curtailment = frequency_deviation * 2.0  # 2% per 0.01 Hz
            curtailment_factor += freq_curtailment

        # Positive power balance indicates excess generation
        if power_balance > 10.0:  # 10 MW excess threshold
            balance_curtailment = min(power_balance / 100.0, 0.2)  # Up to 20%
            curtailment_factor += balance_curtailment

        # Apply curtailment limits
        curtailment_factor = min(curtailment_factor, self.max_curtailment_factor)

        # Only command if significant curtailment needed
        if curtailment_factor > 0.02:  # 2% threshold
            actions["curtailment_factor"] = curtailment_factor

        return actions

    def _calculate_wind_actions(self, wind: WindTurbine, grid_state: GridState) -> dict[str, float]:
        """Calculate wind turbine control actions using curtailment rules.

        Args:
            wind: Wind turbine asset
            grid_state: Current grid state

        Returns:
            Dictionary containing wind control actions
        """
        actions = {}

        # Similar logic to solar curtailment
        frequency_deviation = grid_state.frequency_hz - 60.0
        power_balance = grid_state.power_balance_mw

        curtailment_factor = 0.0

        # High frequency indicates over-generation
        if frequency_deviation > self.frequency_deadband_hz:
            freq_curtailment = frequency_deviation * 1.5  # Slightly less aggressive than solar
            curtailment_factor += freq_curtailment

        # Positive power balance indicates excess generation
        if power_balance > 15.0:  # 15 MW excess threshold (higher than solar)
            balance_curtailment = min(power_balance / 150.0, 0.25)  # Up to 25%
            curtailment_factor += balance_curtailment

        # Apply curtailment limits
        curtailment_factor = min(curtailment_factor, self.max_curtailment_factor)

        # Only command if significant curtailment needed
        if curtailment_factor > 0.02:  # 2% threshold
            actions["curtailment_factor"] = curtailment_factor

        return actions

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for rule-based controller.

        Returns:
            Dictionary containing performance metrics
        """
        if not self.last_grid_state:
            return {
                "controller_type": self.controller_type,
                "initialized": self.initialized,
                "control_actions_count": 0,
                "frequency_deviation_hz": 0.0,
                "power_balance_mw": 0.0,
                "response_time_s": 0.0,
            }

        # Calculate average metrics from history
        avg_frequency_deviation = (
            sum(self.frequency_deviation_history) / len(self.frequency_deviation_history)
            if self.frequency_deviation_history
            else 0.0
        )

        avg_response_time = (
            sum(action["response_time_s"] for action in self.control_actions_history)
            / len(self.control_actions_history)
            if self.control_actions_history
            else 0.0
        )

        # Calculate efficiency based on frequency stability
        efficiency = max(0.0, 1.0 - avg_frequency_deviation / 0.5)  # 0.5 Hz = 0% efficiency

        metrics = {
            "controller_type": self.controller_type,
            "initialized": self.initialized,
            "control_actions_count": self.control_actions_count,
            "frequency_deviation_hz": avg_frequency_deviation,
            "power_balance_mw": self.last_grid_state.power_balance_mw,
            "response_time_s": avg_response_time,
            "efficiency": efficiency,
            "last_frequency_hz": self.last_frequency_hz,
            "rule_weights": self.rule_weights.copy(),
            "history_length": len(self.control_actions_history),
        }

        # Update performance history
        self._update_performance_history(metrics)

        return metrics

    def reset(self) -> None:
        """Reset rule-based controller to initial state."""
        self.last_grid_state = None
        self.last_frequency_hz = 60.0
        self.frequency_deviation_history.clear()
        self.control_actions_history.clear()
        self.performance_history.clear()
        self.control_actions_count = 0
        self.last_update_time = None

        logger.info("Rule-based controller reset to initial state")

    def get_rule_status(self) -> dict[str, Any]:
        """Get detailed status of rule-based control logic.

        Returns:
            Dictionary containing rule status information
        """
        if not self.last_grid_state:
            return {"status": "no_data"}

        status = {
            "frequency_regulation": {
                "active": abs(self.last_grid_state.frequency_hz - 60.0) > self.frequency_deadband_hz,
                "deviation_hz": self.last_grid_state.frequency_hz - 60.0,
                "deadband_hz": self.frequency_deadband_hz,
            },
            "power_balance": {
                "current_mw": self.last_grid_state.power_balance_mw,
                "status": "balanced" if abs(self.last_grid_state.power_balance_mw) < 5.0 else "imbalanced",
            },
            "control_parameters": {
                "droop_factor": self.droop_factor,
                "soc_target_percent": self.soc_target_percent,
                "max_curtailment_factor": self.max_curtailment_factor,
                "dr_response_factor": self.dr_response_factor,
            },
            "recent_performance": {
                "avg_frequency_deviation_hz": (
                    sum(self.frequency_deviation_history[-10:]) / min(len(self.frequency_deviation_history), 10)
                    if self.frequency_deviation_history
                    else 0.0
                ),
                "recent_actions": len([a for a in self.control_actions_history[-10:] if a["actions_count"] > 0]),
            },
        }

        return status
