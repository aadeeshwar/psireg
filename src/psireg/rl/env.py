"""Gym-compatible environment wrapper for GridEngine.

This module provides a Gym-compatible environment wrapper around the GridEngine
for reinforcement learning applications. The environment enables:

- Multi-asset control through continuous action spaces
- Comprehensive observation space including grid state and asset states
- Multi-objective reward function for grid optimization
- Episode-based training with configurable horizons
- Integration with stable-baselines3 for PPO training

The environment is designed for renewable energy grid optimization tasks including:
- Asset dispatch optimization
- Demand response coordination
- Grid stability maintenance
- Economic optimization
"""

import warnings
from datetime import datetime, timedelta
from typing import Any

import numpy as np

# Optional gym dependency
try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False
    warnings.warn("gymnasium not available. GridEnv will not be functional.", stacklevel=2)

from psireg.config.schema import GridConfig, SimulationConfig
from psireg.sim.assets.base import Asset
from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.assets.wind import WindTurbine
from psireg.sim.engine import GridEngine
from psireg.utils.enums import AssetStatus, AssetType
from psireg.utils.logger import logger


class GridEnv(gym.Env if _GYM_AVAILABLE else object):  # type: ignore[misc]
    """Gym-compatible environment wrapper for GridEngine.

    This environment provides a standardized interface for reinforcement learning
    agents to interact with the GridEngine simulation. It supports:

    - Continuous action spaces for asset control
    - Comprehensive observation spaces including grid and asset states
    - Multi-objective reward functions
    - Episode-based training with configurable horizons
    - Integration with popular RL frameworks

    Attributes:
        grid_engine: The underlying GridEngine simulation
        observation_space: Gym observation space specification
        action_space: Gym action space specification
        reward_weights: Weights for multi-objective reward function
        current_step: Current step in the episode
        max_steps: Maximum steps per episode
        episode_length_hours: Length of each episode in hours
        controllable_assets: Dictionary of controllable assets
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        simulation_config: SimulationConfig | None = None,
        grid_config: GridConfig | None = None,
        episode_length_hours: int = 24,
        max_steps: int | None = None,
        reward_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        """Initialize the GridEnv.

        Args:
            simulation_config: Configuration for simulation parameters
            grid_config: Configuration for grid system parameters
            episode_length_hours: Length of each episode in hours
            max_steps: Maximum steps per episode (overrides episode_length_hours)
            reward_weights: Weights for multi-objective reward function
            seed: Random seed for reproducibility
        """
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium is required for GridEnv. Please install with: pip install gymnasium")

        super().__init__()

        # Initialize configuration
        self.simulation_config = simulation_config or SimulationConfig()
        self.grid_config = grid_config or GridConfig()
        self.episode_length_hours = episode_length_hours

        # Initialize GridEngine
        self.grid_engine = GridEngine(simulation_config=self.simulation_config, grid_config=self.grid_config)

        # Episode configuration
        timestep_minutes = self.simulation_config.timestep_minutes
        steps_per_hour = 60 // timestep_minutes
        self.max_steps = max_steps or (episode_length_hours * steps_per_hour)
        self.current_step = 0

        # Reward configuration
        self.reward_weights = reward_weights or {
            "frequency": 0.3,
            "economics": 0.3,
            "stability": 0.2,
            "efficiency": 0.2,
        }

        # Asset tracking
        self.controllable_assets: dict[str, Asset] = {}
        self.observation_history: list[np.ndarray] = []

        # Initialize spaces (will be updated when assets are added)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Set random seed
        if seed is not None:
            self.seed(seed)

        logger.info(f"GridEnv initialized with {episode_length_hours}h episodes, {self.max_steps} max steps")

    def seed(self, seed: int | None = None) -> list[int]:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value

        Returns:
            List containing the used seed
        """
        if seed is not None:
            np.random.seed(seed)
            return [seed]
        return [np.random.randint(0, 2**32 - 1)]

    def add_asset(self, asset: Asset) -> None:
        """Add an asset to the environment.

        Args:
            asset: Asset to add to the environment
        """
        # Add to GridEngine
        self.grid_engine.add_asset(asset)

        # Track controllable assets
        if self._is_controllable(asset):
            self.controllable_assets[asset.asset_id] = asset

        # Update spaces
        self._update_spaces()

        logger.debug(f"Added asset {asset.asset_id} to GridEnv")

    def _is_controllable(self, asset: Asset) -> bool:
        """Check if an asset is controllable by the RL agent.

        Args:
            asset: Asset to check

        Returns:
            True if asset is controllable
        """
        # Batteries and loads are typically controllable
        # Solar and wind are usually not directly controllable (weather dependent)
        return asset.asset_type in {AssetType.BATTERY, AssetType.LOAD}

    def _update_spaces(self) -> None:
        """Update observation and action spaces based on current assets."""
        # Update action space based on controllable assets
        num_actions = len(self.controllable_assets)
        if num_actions > 0:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Update observation space
        obs_size = self._calculate_observation_size()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        logger.debug(f"Updated spaces: action={self.action_space.shape}, observation={self.observation_space.shape}")

    def _calculate_observation_size(self) -> int:
        """Calculate the size of the observation space.

        Returns:
            Size of observation vector
        """
        # Grid state: 6 components
        # - frequency_hz, total_generation_mw, total_load_mw, total_storage_mw, grid_losses_mw, power_balance_mw
        grid_state_size = 6

        # Asset states: variable per asset
        asset_state_size = 0
        for asset in self.grid_engine.assets.values():
            if isinstance(asset, Battery):
                asset_state_size += 4  # current_output_mw, current_soc_percent, efficiency, temperature
            elif isinstance(asset, Load):
                asset_state_size += 3  # current_demand_mw, baseline_demand_mw, dr_capability_mw
            elif isinstance(asset, SolarPanel | WindTurbine):
                asset_state_size += 3  # current_output_mw, capacity_mw, efficiency
            else:
                asset_state_size += 2  # current_output_mw, capacity_mw

        # Time features: 4 components
        # - hour_of_day, day_of_week, month, season
        time_features_size = 4

        # Episode features: 2 components
        # - step_ratio, time_remaining_ratio
        episode_features_size = 2

        total_size = grid_state_size + asset_state_size + time_features_size + episode_features_size
        return max(total_size, 10)  # Minimum size to avoid edge cases

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Tuple of (observation, info)
        """
        # Set seed if provided
        if seed is not None:
            self.seed(seed)

        # Reset episode state
        self.current_step = 0
        self.observation_history.clear()

        # Store assets before reset (to preserve them)
        stored_assets = list(self.grid_engine.assets.values())
        stored_nodes = list(self.grid_engine.nodes.values())

        # Reset GridEngine (this clears all assets)
        start_time = datetime.now()
        self.grid_engine.reset(start_time=start_time)

        # Restore nodes and assets
        for node in stored_nodes:
            self.grid_engine.add_node(node)

        # Clear controllable assets before restoring
        self.controllable_assets.clear()

        for asset in stored_assets:
            self.grid_engine.add_asset(asset)
            # Add to controllable assets if applicable
            if self._is_controllable(asset):
                self.controllable_assets[asset.asset_id] = asset
            # Reset asset to online status
            if asset.is_online() or asset.status == AssetStatus.OFFLINE:
                asset.set_status(AssetStatus.ONLINE)

        # Update action and observation spaces to reflect current asset state
        self._update_spaces()

        # Get initial observation
        observation = self._get_observation()
        self.observation_history.append(observation)

        info = {
            "step": self.current_step,
            "time": self.grid_engine.current_time.isoformat(),
            "assets": len(self.grid_engine.assets),
            "controllable_assets": len(self.controllable_assets),
        }

        logger.debug(f"Environment reset at step {self.current_step}")
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one time step within the environment.

        Args:
            action: Action to take (normalized to [-1, 1])

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        if action.shape != self.action_space.shape:
            raise ValueError(f"Action shape {action.shape} does not match action space {self.action_space.shape}")

        # Apply action to controllable assets
        self._apply_action(action)

        # Step the simulation
        timestep_delta = timedelta(minutes=self.simulation_config.timestep_minutes)
        self.grid_engine.step(timestep_delta)

        # Update step counter
        self.current_step += 1

        # Get observation
        observation = self._get_observation()
        self.observation_history.append(observation)

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps

        # Prepare info
        info = self._get_info(action, reward)

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action to controllable assets.

        Args:
            action: Normalized action vector [-1, 1]
        """
        if len(self.controllable_assets) == 0:
            return

        # Map actions to assets
        asset_ids = list(self.controllable_assets.keys())

        for i, asset_id in enumerate(asset_ids):
            if i >= len(action):
                break

            asset = self.controllable_assets[asset_id]
            action_value = action[i]

            # Apply action based on asset type
            if isinstance(asset, Battery):
                self._apply_battery_action(asset, action_value)
            elif isinstance(asset, Load):
                self._apply_load_action(asset, action_value)
            else:
                logger.warning(f"Unknown controllable asset type: {type(asset)}")

    def _apply_battery_action(self, battery: Battery, action_value: float) -> None:
        """Apply action to battery asset.

        Args:
            battery: Battery asset to control
            action_value: Action value in [-1, 1]
        """
        # Map action to power setpoint
        # -1 = maximum discharge, +1 = maximum charge
        max_charge = battery.get_max_charge_power()
        max_discharge = battery.get_max_discharge_power()

        if action_value >= 0:
            # Charging
            power_setpoint = action_value * max_charge
        else:
            # Discharging
            power_setpoint = action_value * max_discharge

        # Apply power setpoint
        battery.set_power_setpoint(power_setpoint)

    def _apply_load_action(self, load: Load, action_value: float) -> None:
        """Apply action to load asset.

        Args:
            load: Load asset to control
            action_value: Action value in [-1, 1]
        """
        # Map action to demand response signal
        # -1 = maximum reduction, +1 = maximum increase
        baseline_demand = load.baseline_demand_mw
        dr_capability = getattr(load, "dr_capability_mw", baseline_demand * 0.2)

        # Calculate demand response signal
        dr_signal = action_value * dr_capability

        # Apply demand response signal
        load.set_demand_response_signal(dr_signal)

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector.

        Returns:
            Observation vector as numpy array
        """
        observations = []

        # Grid state
        grid_state = self.grid_engine.get_state()
        observations.extend(
            [
                grid_state.frequency_hz / 60.0,  # Normalized frequency
                grid_state.total_generation_mw / 1000.0,  # Normalized generation
                grid_state.total_load_mw / 1000.0,  # Normalized load
                grid_state.total_storage_mw / 1000.0,  # Normalized storage
                grid_state.grid_losses_mw / 1000.0,  # Normalized losses
                grid_state.power_balance_mw / 1000.0,  # Normalized balance
            ]
        )

        # Asset states
        for asset in self.grid_engine.assets.values():
            if isinstance(asset, Battery):
                observations.extend(
                    [
                        asset.current_output_mw / asset.capacity_mw,  # Normalized power
                        asset.current_soc_percent / 100.0,  # Normalized SoC
                        asset.get_current_charge_efficiency(),  # Efficiency
                        asset.current_temperature_c / 50.0,  # Normalized temperature
                    ]
                )
            elif isinstance(asset, Load):
                observations.extend(
                    [
                        asset.current_demand_mw / asset.capacity_mw,  # Normalized demand
                        asset.baseline_demand_mw / asset.capacity_mw,  # Normalized baseline
                        getattr(asset, "dr_capability_mw", 0) / asset.capacity_mw,  # Normalized DR capability
                    ]
                )
            elif isinstance(asset, SolarPanel | WindTurbine):
                observations.extend(
                    [
                        asset.current_output_mw / asset.capacity_mw,  # Normalized output
                        1.0,  # Capacity factor (always 1 for normalization)
                        getattr(asset, "efficiency", 0.2),  # Efficiency
                    ]
                )
            else:
                observations.extend(
                    [asset.current_output_mw / max(asset.capacity_mw, 1.0), 1.0]  # Normalized output  # Placeholder
                )

        # Time features
        current_time = self.grid_engine.current_time
        observations.extend(
            [
                current_time.hour / 24.0,  # Hour of day
                current_time.weekday() / 7.0,  # Day of week
                current_time.month / 12.0,  # Month
                (current_time.month - 1) // 3 / 4.0,  # Season
            ]
        )

        # Episode features
        observations.extend(
            [
                self.current_step / self.max_steps,  # Step ratio
                max(0, (self.max_steps - self.current_step) / self.max_steps),  # Time remaining ratio
            ]
        )

        # Pad or truncate to match observation space
        obs_array = np.array(observations, dtype=np.float32)
        expected_size = self.observation_space.shape[0]

        if len(obs_array) < expected_size:
            # Pad with zeros
            obs_array = np.pad(obs_array, (0, expected_size - len(obs_array)), mode="constant", constant_values=0)
        elif len(obs_array) > expected_size:
            # Truncate
            obs_array = obs_array[:expected_size]

        return obs_array

    def _calculate_reward(self) -> float:
        """Calculate reward based on grid state and objectives.

        Returns:
            Reward value
        """
        grid_state = self.grid_engine.get_state()

        # Initialize reward components
        reward_components = {}

        # 1. Frequency stability reward
        frequency_deviation = abs(grid_state.frequency_hz - self.grid_config.frequency_hz)
        frequency_reward = -frequency_deviation * 10  # Penalty for deviation
        reward_components["frequency"] = frequency_reward

        # 2. Economic efficiency reward
        # Reward for high renewable utilization and low grid losses
        renewable_generation = 0.0
        total_generation = max(grid_state.total_generation_mw, 1.0)

        for asset in self.grid_engine.assets.values():
            if asset.asset_type in {AssetType.SOLAR, AssetType.WIND}:
                renewable_generation += asset.current_output_mw

        renewable_ratio = renewable_generation / total_generation
        loss_ratio = grid_state.grid_losses_mw / total_generation

        economic_reward = renewable_ratio * 10 - loss_ratio * 5
        reward_components["economics"] = economic_reward

        # 3. Grid stability reward
        # Reward for balanced supply and demand
        power_balance = abs(grid_state.power_balance_mw)
        balance_reward = -power_balance * 0.1  # Penalty for imbalance
        reward_components["stability"] = balance_reward

        # 4. Efficiency reward
        # Reward for optimal asset utilization
        asset_efficiency = 0.0
        num_assets = len(self.grid_engine.assets)

        for asset in self.grid_engine.assets.values():
            if isinstance(asset, Battery):
                # Reward for keeping battery SoC in optimal range
                soc_dev = abs(asset.current_soc_percent - 50.0)
                asset_efficiency += max(0, (50.0 - soc_dev) / 50.0)
            else:
                # Reward for high capacity utilization
                utilization = asset.current_output_mw / max(asset.capacity_mw, 1.0)
                asset_efficiency += min(1.0, abs(utilization))

        if num_assets > 0:
            asset_efficiency /= num_assets

        efficiency_reward = asset_efficiency * 5
        reward_components["efficiency"] = efficiency_reward

        # Combine reward components
        total_reward = 0.0
        for component, weight in self.reward_weights.items():
            if component in reward_components:
                total_reward += weight * reward_components[component]

        # Store reward components for analysis
        self._last_reward_components = reward_components

        return total_reward

    def _check_terminated(self) -> bool:
        """Check if episode should be terminated early.

        Returns:
            True if episode should be terminated
        """
        grid_state = self.grid_engine.get_state()

        # Terminate if grid becomes unstable
        frequency_deviation = abs(grid_state.frequency_hz - self.grid_config.frequency_hz)
        if frequency_deviation > 5.0:  # 5 Hz deviation is critical
            return True

        # Terminate if power balance becomes critical
        if abs(grid_state.power_balance_mw) > 1000.0:  # 1000 MW imbalance is critical
            return True

        return False

    def _get_info(self, action: np.ndarray, reward: float) -> dict[str, Any]:
        """Get additional information about the current step.

        Args:
            action: Action taken
            reward: Reward received

        Returns:
            Information dictionary
        """
        grid_state = self.grid_engine.get_state()

        info = {
            "step": self.current_step,
            "time": self.grid_engine.current_time.isoformat(),
            "grid_state": {
                "frequency_hz": grid_state.frequency_hz,
                "total_generation_mw": grid_state.total_generation_mw,
                "total_load_mw": grid_state.total_load_mw,
                "total_storage_mw": grid_state.total_storage_mw,
                "power_balance_mw": grid_state.power_balance_mw,
                "grid_losses_mw": grid_state.grid_losses_mw,
            },
            "action": action.tolist(),
            "reward": reward,
            "reward_components": getattr(self, "_last_reward_components", {}),
            "controllable_assets": len(self.controllable_assets),
        }

        return info

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Render the environment.

        Args:
            mode: Rendering mode ("human" or "rgb_array")

        Returns:
            Rendered image if mode is "rgb_array", None otherwise
        """
        if mode == "human":
            # Print current state
            grid_state = self.grid_engine.get_state()
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Time: {self.grid_engine.current_time}")
            print(f"Frequency: {grid_state.frequency_hz:.2f} Hz")
            print(f"Generation: {grid_state.total_generation_mw:.2f} MW")
            print(f"Load: {grid_state.total_load_mw:.2f} MW")
            print(f"Balance: {grid_state.power_balance_mw:.2f} MW")
            print("-" * 50)
            return None
        elif mode == "rgb_array":
            # Return simple visualization as RGB array
            # This is a placeholder - could be enhanced with actual plotting
            return np.zeros((400, 600, 3), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        """Close the environment and cleanup resources."""
        # Reset GridEngine
        self.grid_engine.reset()

        # Clear state
        self.controllable_assets.clear()
        self.observation_history.clear()

        logger.info("GridEnv closed")

    def get_grid_summary(self) -> dict[str, Any]:
        """Get summary of current grid state.

        Returns:
            Dictionary with grid summary information
        """
        return self.grid_engine.get_grid_summary()

    def get_controllable_assets(self) -> dict[str, Asset]:
        """Get dictionary of controllable assets.

        Returns:
            Dictionary of controllable assets
        """
        return self.controllable_assets.copy()

    def get_observation_history(self) -> list[np.ndarray]:
        """Get history of observations.

        Returns:
            List of observation arrays
        """
        return self.observation_history.copy()


# Export classes when gym is available
if _GYM_AVAILABLE:
    __all__ = ["GridEnv"]
else:
    __all__ = []
