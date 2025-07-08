"""Abstract base controller interface for renewable energy grid control."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from psireg.sim.engine import GridEngine, GridState
from psireg.utils.logger import logger


class BaseController(ABC):
    """Abstract base class for all grid controllers.

    This class defines the common interface that all controller implementations
    must follow for renewable energy grid control. It provides a standardized
    API for initialization, control action computation, performance tracking,
    and state management.

    Controller implementations include:
    - Rule-based controllers using traditional power system control logic
    - ML-only controllers using reinforcement learning models
    - Swarm-only controllers using distributed swarm intelligence

    Attributes:
        grid_engine: Reference to the grid simulation engine
        controller_type: String identifier for the controller type
        initialized: Flag indicating if controller has been initialized
        performance_history: Historical performance metrics
    """

    def __init__(self):
        """Initialize base controller."""
        self.grid_engine: GridEngine | None = None
        self.controller_type: str = "base"
        self.initialized: bool = False
        self.performance_history: list[dict[str, Any]] = []
        self.last_update_time: datetime | None = None
        self.control_actions_count: int = 0

        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def initialize(self, grid_engine: GridEngine) -> bool:
        """Initialize the controller with a grid engine.

        This method must be called before the controller can be used.
        It sets up any necessary internal state, configures assets,
        and prepares the controller for operation.

        Args:
            grid_engine: The grid simulation engine to control

        Returns:
            True if initialization was successful, False otherwise
        """
        pass

    @abstractmethod
    def update(self, grid_state: GridState, dt: float) -> None:
        """Update controller state based on current grid conditions.

        This method is called at each simulation timestep to update
        the controller's internal state based on current grid conditions.
        It should analyze the grid state and prepare for control actions.

        Args:
            grid_state: Current state of the grid system
            dt: Time step duration in seconds
        """
        pass

    @abstractmethod
    def get_control_actions(self) -> dict[str, dict[str, float]]:
        """Calculate and return control actions for grid assets.

        This method computes optimal control actions for all controllable
        assets based on the current grid state and controller strategy.

        Returns:
            Dictionary mapping asset IDs to their control actions.
            Format: {
                'asset_id': {
                    'parameter_name': value,
                    ...
                },
                ...
            }

        Example:
            {
                'battery_1': {'power_setpoint_mw': 25.0},
                'load_1': {'dr_signal_mw': -5.0},
                'solar_1': {'curtailment_factor': 0.1}
            }
        """
        pass

    @abstractmethod
    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics for the controller.

        This method returns various performance indicators that can
        be used to evaluate controller effectiveness and efficiency.

        Returns:
            Dictionary containing performance metrics such as:
            - efficiency: Overall control efficiency (0.0 to 1.0)
            - frequency_deviation_hz: Average frequency deviation
            - power_balance_mw: Average power balance achieved
            - response_time_s: Average response time to grid events
            - control_actions_count: Total number of control actions taken
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset controller to initial state.

        This method clears all internal state and history,
        returning the controller to its initial condition.
        Useful for starting new simulation runs.
        """
        pass

    # Common utility methods that can be used by all controllers

    def is_initialized(self) -> bool:
        """Check if controller has been properly initialized.

        Returns:
            True if controller is initialized and ready to use
        """
        return self.initialized and self.grid_engine is not None

    def get_controllable_assets(self) -> dict[str, Any]:
        """Get dictionary of controllable assets from grid engine.

        Returns:
            Dictionary mapping asset IDs to asset objects for
            assets that can be controlled by this controller
        """
        if not self.is_initialized():
            return {}

        controllable_assets = {}
        for asset_id, asset in self.grid_engine.assets.items():
            if self._is_asset_controllable(asset):
                controllable_assets[asset_id] = asset

        return controllable_assets

    def _is_asset_controllable(self, asset: Any) -> bool:
        """Check if an asset can be controlled by this controller.

        This default implementation considers batteries and loads as controllable.
        Subclasses can override this to define their own controllability rules.

        Args:
            asset: Asset object to check

        Returns:
            True if asset is controllable by this controller
        """
        from psireg.utils.enums import AssetType

        # Default: batteries and loads are controllable
        # Solar and wind are typically controllable via curtailment
        controllable_types = {AssetType.BATTERY, AssetType.LOAD, AssetType.SOLAR, AssetType.WIND}

        return hasattr(asset, "asset_type") and asset.asset_type in controllable_types

    def _validate_grid_state(self, grid_state: GridState) -> bool:
        """Validate that grid state contains required information.

        Args:
            grid_state: Grid state to validate

        Returns:
            True if grid state is valid
        """
        required_attrs = ["frequency_hz", "total_generation_mw", "total_load_mw", "power_balance_mw"]

        for attr in required_attrs:
            if not hasattr(grid_state, attr):
                logger.warning(f"Grid state missing required attribute: {attr}")
                return False

        return True

    def _update_performance_history(self, metrics: dict[str, Any]) -> None:
        """Update performance history with latest metrics.

        Args:
            metrics: Latest performance metrics to record
        """
        timestamp_metrics = {"timestamp": datetime.now(), **metrics}
        self.performance_history.append(timestamp_metrics)

        # Keep only last 1000 entries to prevent memory growth
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary statistics of controller performance over time.

        Returns:
            Dictionary containing summary statistics like mean, std dev,
            min, max values for key performance metrics
        """
        if not self.performance_history:
            return {}

        # Extract numeric metrics
        numeric_metrics = {}
        for entry in self.performance_history:
            for key, value in entry.items():
                if isinstance(value, int | float) and key != "timestamp":
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)

        # Calculate summary statistics
        summary = {}
        for metric, values in numeric_metrics.items():
            if values:
                import statistics

                summary[metric] = {
                    "mean": statistics.mean(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        summary["total_entries"] = len(self.performance_history)
        summary["time_span_seconds"] = (
            (self.performance_history[-1]["timestamp"] - self.performance_history[0]["timestamp"]).total_seconds()
            if len(self.performance_history) > 1
            else 0
        )

        return summary

    def __str__(self) -> str:
        """String representation of controller."""
        status = "initialized" if self.initialized else "not initialized"
        return f"{self.__class__.__name__}({status}, actions={self.control_actions_count})"

    def __repr__(self) -> str:
        """Detailed string representation of controller."""
        return (
            f"{self.__class__.__name__}("
            f"type='{self.controller_type}', "
            f"initialized={self.initialized}, "
            f"actions={self.control_actions_count}, "
            f"history_entries={len(self.performance_history)})"
        )
