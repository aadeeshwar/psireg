"""Tests for RL Environment Wrapper around GridEngine.

This test suite covers:
- GridEnv initialization and configuration
- Gym interface compliance (reset, step, render)
- Observation space and action space definition
- Reward function implementation
- Integration with GridEngine
- Asset control through actions
- Multi-step episodes and termination conditions
"""

import numpy as np
import pytest
from psireg.config.schema import GridConfig, SimulationConfig
from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.engine import GridEngine, NetworkNode
from psireg.utils.enums import AssetStatus

# Try to import RL dependencies
try:
    import gymnasium as gym
    from psireg.rl.env import GridEnv

    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False
    pytest.skip("RL dependencies not available", allow_module_level=True)


class TestGridEnvInitialization:
    """Test GridEnv initialization and configuration."""

    def test_grid_env_basic_initialization(self):
        """Test basic GridEnv initialization with default settings."""
        env = GridEnv()

        assert env is not None
        assert isinstance(env.grid_engine, GridEngine)
        assert env.max_steps > 0
        assert env.current_step == 0
        assert env.episode_length_hours > 0

    def test_grid_env_custom_initialization(self):
        """Test GridEnv initialization with custom configuration."""
        sim_config = SimulationConfig(timestep_minutes=15)
        grid_config = GridConfig(frequency_hz=50.0)

        env = GridEnv(
            simulation_config=sim_config,
            grid_config=grid_config,
            episode_length_hours=48,
            max_steps=1000,
            reward_weights={"frequency": 0.4, "economics": 0.3, "stability": 0.3},
        )

        assert env.grid_engine.simulation_config == sim_config
        assert env.grid_engine.grid_config == grid_config
        assert env.episode_length_hours == 48
        assert env.max_steps == 1000
        assert env.reward_weights["frequency"] == 0.4

    def test_grid_env_with_assets(self):
        """Test GridEnv initialization with pre-configured assets."""
        env = GridEnv()

        # Add network nodes
        node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
        env.grid_engine.add_node(node)

        # Add assets
        solar = SolarPanel(
            asset_id="solar_1",
            name="Solar Panel 1",
            node_id="test_node",
            capacity_mw=100.0,
            panel_efficiency=0.2,
            panel_area_m2=50000.0,
        )
        battery = Battery(
            asset_id="battery_1", name="Battery 1", node_id="test_node", capacity_mw=50.0, energy_capacity_mwh=200.0
        )

        env.add_asset(solar)
        env.add_asset(battery)

        assert len(env.grid_engine.assets) == 2
        # Solar panels are not controllable (weather dependent)
        assert "solar_1" not in env.controllable_assets
        assert "battery_1" in env.controllable_assets


class TestGymInterface:
    """Test Gym interface compliance."""

    def setup_method(self):
        """Set up test environment."""
        self.env = GridEnv()
        self._setup_basic_grid()

    def _setup_basic_grid(self):
        """Set up basic grid configuration for testing."""
        # Add node
        node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
        self.env.grid_engine.add_node(node)

        # Add controllable assets
        solar = SolarPanel(
            asset_id="solar_1",
            name="Solar Panel 1",
            node_id="test_node",
            capacity_mw=100.0,
            panel_efficiency=0.2,
            panel_area_m2=50000.0,
        )
        battery = Battery(
            asset_id="battery_1", name="Battery 1", node_id="test_node", capacity_mw=50.0, energy_capacity_mwh=200.0
        )
        load = Load(asset_id="load_1", name="Load 1", node_id="test_node", capacity_mw=80.0, baseline_demand_mw=60.0)

        self.env.add_asset(solar)
        self.env.add_asset(battery)
        self.env.add_asset(load)

        # Set assets online
        solar.set_status(AssetStatus.ONLINE)
        battery.set_status(AssetStatus.ONLINE)
        load.set_status(AssetStatus.ONLINE)

    def test_reset_function(self):
        """Test environment reset functionality."""
        obs, info = self.env.reset()

        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == self.env.observation_space.shape
        assert isinstance(info, dict)
        assert self.env.current_step == 0
        assert self.env.grid_engine.current_time is not None

    def test_step_function(self):
        """Test environment step functionality."""
        obs, info = self.env.reset()

        # Create valid action
        action = self.env.action_space.sample()

        # Take step
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == self.env.observation_space.shape
        assert isinstance(reward, int | float | np.floating)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert self.env.current_step == 1

    def test_observation_space(self):
        """Test observation space definition."""
        obs_space = self.env.observation_space

        assert isinstance(obs_space, gym.spaces.Box)
        assert obs_space.dtype == np.float32
        assert len(obs_space.shape) == 1
        assert obs_space.low.shape == obs_space.high.shape

    def test_action_space(self):
        """Test action space definition."""
        action_space = self.env.action_space

        assert isinstance(action_space, gym.spaces.Box)
        assert action_space.dtype == np.float32
        assert len(action_space.shape) == 1
        assert action_space.low.shape == action_space.high.shape

    def test_multiple_steps(self):
        """Test multiple simulation steps."""
        obs, info = self.env.reset()

        for step in range(5):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            assert self.env.current_step == step + 1
            assert not terminated  # Should not terminate early

    def test_episode_termination(self):
        """Test episode termination conditions."""
        self.env.max_steps = 3  # Set short episode
        obs, info = self.env.reset()

        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            step += 1

            # Prevent infinite loop
            if step > 10:
                break

        assert terminated or truncated
        assert step <= self.env.max_steps


class TestObservationSpace:
    """Test observation space construction and values."""

    def setup_method(self):
        """Set up test environment."""
        self.env = GridEnv()
        self._setup_basic_grid()

    def _setup_basic_grid(self):
        """Set up basic grid configuration for testing."""
        # Add node
        node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
        self.env.grid_engine.add_node(node)

        # Add assets
        solar = SolarPanel(
            asset_id="solar_1",
            name="Solar Panel 1",
            node_id="test_node",
            capacity_mw=100.0,
            panel_efficiency=0.2,
            panel_area_m2=50000.0,
        )
        battery = Battery(
            asset_id="battery_1", name="Battery 1", node_id="test_node", capacity_mw=50.0, energy_capacity_mwh=200.0
        )

        self.env.add_asset(solar)
        self.env.add_asset(battery)

        solar.set_status(AssetStatus.ONLINE)
        battery.set_status(AssetStatus.ONLINE)

    def test_observation_construction(self):
        """Test observation vector construction."""
        obs, info = self.env.reset()

        # Check observation structure
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

        # Check observation bounds
        assert np.all(obs >= self.env.observation_space.low)
        assert np.all(obs <= self.env.observation_space.high)

    def test_observation_consistency(self):
        """Test observation consistency across steps."""
        obs1, info = self.env.reset()

        # Take a step
        action = self.env.action_space.sample()
        obs2, reward, terminated, truncated, info = self.env.step(action)

        # Observations should be different after step
        assert not np.array_equal(obs1, obs2)

        # But should maintain same shape and bounds
        assert obs1.shape == obs2.shape
        assert np.all(obs2 >= self.env.observation_space.low)
        assert np.all(obs2 <= self.env.observation_space.high)


class TestActionSpace:
    """Test action space and asset control."""

    def setup_method(self):
        """Set up test environment."""
        self.env = GridEnv()
        self._setup_basic_grid()

    def _setup_basic_grid(self):
        """Set up basic grid configuration for testing."""
        # Add node (if not already present)
        if "test_node" not in self.env.grid_engine.nodes:
            node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
            self.env.grid_engine.add_node(node)

        # Add controllable assets (if not already present)
        if "battery_1" not in self.env.grid_engine.assets:
            battery = Battery(
                asset_id="battery_1", name="Battery 1", node_id="test_node", capacity_mw=50.0, energy_capacity_mwh=200.0
            )
            self.env.add_asset(battery)
            battery.set_status(AssetStatus.ONLINE)

        if "load_1" not in self.env.grid_engine.assets:
            load = Load(
                asset_id="load_1", name="Load 1", node_id="test_node", capacity_mw=80.0, baseline_demand_mw=60.0
            )
            self.env.add_asset(load)
            load.set_status(AssetStatus.ONLINE)

    def test_action_space_bounds(self):
        """Test action space bounds correspond to asset capabilities."""
        action_space = self.env.action_space

        # Should have one action per controllable asset
        expected_actions = len(self.env.controllable_assets)
        assert action_space.shape[0] == expected_actions

        # Action bounds should be normalized [-1, 1]
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)

    def test_action_application(self):
        """Test that actions are properly applied to assets."""
        obs, info = self.env.reset()

        # Re-add assets after reset (since reset clears them)
        self._setup_basic_grid()

        # Create specific action
        action = np.array([0.5, -0.3], dtype=np.float32)  # Battery charge, load reduction

        # Get initial asset states
        battery = self.env.grid_engine.assets["battery_1"]
        load = self.env.grid_engine.assets["load_1"]

        initial_battery_power = battery.current_output_mw
        initial_load_power = load.current_demand_mw

        # Apply action
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Check that assets were affected
        # (Exact values depend on implementation)
        assert battery.current_output_mw != initial_battery_power or load.current_demand_mw != initial_load_power

    def test_invalid_action_handling(self):
        """Test handling of invalid actions."""
        obs, info = self.env.reset()

        # Create action with wrong shape
        with pytest.raises((ValueError, AssertionError)):
            invalid_action = np.array([1.0])  # Wrong size
            self.env.step(invalid_action)


class TestRewardFunction:
    """Test reward function implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.env = GridEnv()
        self._setup_basic_grid()

    def _setup_basic_grid(self):
        """Set up basic grid configuration for testing."""
        # Add node
        node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
        self.env.grid_engine.add_node(node)

        # Add mixed assets
        solar = SolarPanel(
            asset_id="solar_1",
            name="Solar Panel 1",
            node_id="test_node",
            capacity_mw=100.0,
            panel_efficiency=0.2,
            panel_area_m2=50000.0,
        )
        battery = Battery(
            asset_id="battery_1", name="Battery 1", node_id="test_node", capacity_mw=50.0, energy_capacity_mwh=200.0
        )
        load = Load(asset_id="load_1", name="Load 1", node_id="test_node", capacity_mw=80.0, baseline_demand_mw=60.0)

        self.env.add_asset(solar)
        self.env.add_asset(battery)
        self.env.add_asset(load)

        solar.set_status(AssetStatus.ONLINE)
        battery.set_status(AssetStatus.ONLINE)
        load.set_status(AssetStatus.ONLINE)

    def test_reward_calculation(self):
        """Test reward calculation returns valid values."""
        obs, info = self.env.reset()

        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        assert isinstance(reward, int | float | np.floating)
        assert not np.isnan(reward)
        assert not np.isinf(reward)

    def test_reward_components(self):
        """Test individual reward components."""
        obs, info = self.env.reset()

        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Should have reward components in info
        assert "reward_components" in info
        assert isinstance(info["reward_components"], dict)

        # Check expected components
        expected_components = ["frequency", "economics", "stability"]
        for component in expected_components:
            if component in self.env.reward_weights:
                assert component in info["reward_components"]

    def test_reward_consistency(self):
        """Test reward consistency for similar states."""
        obs1, info = self.env.reset()

        # Take same action twice from reset
        action = np.array([0.0, 0.0], dtype=np.float32)  # Neutral action

        obs2, reward1, terminated1, truncated1, info1 = self.env.step(action)

        # Reset and take same action again
        obs3, info = self.env.reset()
        obs4, reward2, terminated2, truncated2, info2 = self.env.step(action)

        # Rewards should be similar (but not necessarily identical due to randomness)
        assert abs(reward1 - reward2) < 1.0  # Allow for some variation


class TestGridEngineIntegration:
    """Test integration with GridEngine."""

    def setup_method(self):
        """Set up test environment."""
        self.env = GridEnv()
        self._setup_basic_grid()

    def _setup_basic_grid(self):
        """Set up basic grid configuration for testing."""
        # Add node (if not already present)
        if "test_node" not in self.env.grid_engine.nodes:
            node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
            self.env.grid_engine.add_node(node)

        # Add assets (if not already present)
        if "solar_1" not in self.env.grid_engine.assets:
            solar = SolarPanel(
                asset_id="solar_1",
                name="Solar Panel 1",
                node_id="test_node",
                capacity_mw=100.0,
                panel_efficiency=0.2,
                panel_area_m2=50000.0,
            )
            self.env.add_asset(solar)
            solar.set_status(AssetStatus.ONLINE)

        if "battery_1" not in self.env.grid_engine.assets:
            battery = Battery(
                asset_id="battery_1", name="Battery 1", node_id="test_node", capacity_mw=50.0, energy_capacity_mwh=200.0
            )
            self.env.add_asset(battery)
            battery.set_status(AssetStatus.ONLINE)

    def test_engine_time_progression(self):
        """Test that GridEngine time progresses correctly."""
        obs, info = self.env.reset()

        initial_time = self.env.grid_engine.current_time

        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        final_time = self.env.grid_engine.current_time

        # Time should have progressed
        assert final_time > initial_time

    def test_engine_state_consistency(self):
        """Test GridEngine state consistency."""
        obs, info = self.env.reset()

        # Check initial state
        initial_state = self.env.grid_engine.get_state()
        assert initial_state is not None

        # Take step
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Check state after step
        final_state = self.env.grid_engine.get_state()
        assert final_state is not None
        assert final_state.timestamp != initial_state.timestamp

    def test_asset_state_updates(self):
        """Test that asset states are updated correctly."""
        obs, info = self.env.reset()

        # Re-add assets after reset (since reset clears them)
        self._setup_basic_grid()

        # Get initial asset states
        solar = self.env.grid_engine.assets["solar_1"]
        battery = self.env.grid_engine.assets["battery_1"]

        initial_solar_output = solar.current_output_mw
        initial_battery_soc = battery.current_soc_percent

        # Take step
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Check that assets were updated (at least time-wise)
        # Solar output might change due to time progression
        # Battery SoC might change due to actions
        final_solar_output = solar.current_output_mw
        final_battery_soc = battery.current_soc_percent

        # At least one should have changed
        assert final_solar_output != initial_solar_output or final_battery_soc != initial_battery_soc


class TestEnvironmentEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_grid_environment(self):
        """Test environment with no assets."""
        env = GridEnv()

        # Should still be able to reset
        obs, info = env.reset()
        assert obs is not None

        # Action space should be empty or minimal
        assert env.action_space.shape[0] >= 0

    def test_single_asset_environment(self):
        """Test environment with single asset."""
        env = GridEnv()

        # Add single asset
        node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
        env.grid_engine.add_node(node)

        battery = Battery(
            asset_id="battery_1", name="Battery 1", node_id="test_node", capacity_mw=50.0, energy_capacity_mwh=200.0
        )
        env.add_asset(battery)
        battery.set_status(AssetStatus.ONLINE)

        # Should work with single asset
        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert next_obs is not None
        assert isinstance(reward, int | float | np.floating)

    def test_disconnected_assets(self):
        """Test environment with disconnected assets."""
        env = GridEnv()

        # Add assets but don't set them online
        node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
        env.grid_engine.add_node(node)

        battery = Battery(
            asset_id="battery_1", name="Battery 1", node_id="test_node", capacity_mw=50.0, energy_capacity_mwh=200.0
        )
        env.add_asset(battery)
        # Don't set status to ONLINE

        # Should handle gracefully
        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert next_obs is not None
