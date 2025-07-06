"""Integration tests for RL module components.

This test suite covers:
- Complete RL module integration
- GridEnv with PPOTrainer integration
- GridPredictor integration with trained models
- End-to-end training and inference pipeline
- Performance and robustness testing
"""

import os
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest
from psireg.config.schema import GridConfig, RLConfig, SimulationConfig
from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.engine import GridEngine, NetworkNode
from psireg.utils.enums import AssetStatus


class TestRLModuleIntegration:
    """Test complete RL module integration."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_rl_module_complete_import(self):
        """Test complete RL module can be imported."""
        # Test main package import
        import psireg

        # Check if RL module is available
        assert hasattr(psireg, "_rl_available")

        if psireg._rl_available:
            assert "rl" in psireg.__all__
            assert hasattr(psireg, "rl")
        else:
            pytest.skip("RL dependencies not available")

    def test_rl_config_integration(self):
        """Test RL configuration integration with other configs."""
        from psireg.config.schema import PSIREGConfig

        # Create complete configuration
        config = PSIREGConfig(
            simulation=SimulationConfig(timestep_minutes=15),
            rl=RLConfig(learning_rate=0.001, training_episodes=100),
            grid=GridConfig(frequency_hz=60.0),
        )

        assert config.simulation.timestep_minutes == 15
        assert config.rl.learning_rate == 0.001
        assert config.rl.training_episodes == 100
        assert config.grid.frequency_hz == 60.0

    def test_grid_env_complete_workflow(self):
        """Test complete GridEnv workflow without RL dependencies."""
        try:
            from psireg.rl.env import _GYM_AVAILABLE, GridEnv

            if not _GYM_AVAILABLE:
                pytest.skip("Gym dependencies not available")

            # Create environment
            env = GridEnv()

            # Set up basic grid
            self._setup_test_grid(env)

            # Test complete workflow
            obs, info = env.reset()
            assert obs is not None
            assert isinstance(info, dict)

            # Run several steps
            total_reward = 0
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                assert obs is not None
                assert isinstance(reward, int | float | np.floating)
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)

                if terminated or truncated:
                    break

            # Verify environment state
            assert env.current_step > 0
            assert len(env.observation_history) > 0
            assert isinstance(total_reward, int | float | np.floating)

            # Test environment cleanup
            env.close()

        except ImportError:
            pytest.skip("RL environment dependencies not available")

    def _setup_test_grid(self, env):
        """Set up test grid configuration."""
        from psireg.sim.assets.battery import Battery
        from psireg.sim.assets.load import Load
        from psireg.sim.engine import NetworkNode
        from psireg.utils.enums import AssetStatus

        # Add node
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
        load = Load(asset_id="load_1", name="Load 1", node_id="test_node", capacity_mw=80.0, baseline_demand_mw=60.0)

        for asset in [solar, battery, load]:
            env.add_asset(asset)
            asset.set_status(AssetStatus.ONLINE)


class TestPPOTrainerIntegration:
    """Test PPO trainer integration (if dependencies available)."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "ppo_logs")

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_ppo_trainer_initialization(self):
        """Test PPO trainer can be initialized."""
        try:
            from psireg.rl.train import _SB3_AVAILABLE, PPOTrainer

            if not _SB3_AVAILABLE:
                pytest.skip("Stable-baselines3 dependencies not available")

            # Create trainer with minimal configuration
            rl_config = RLConfig(learning_rate=0.001, training_episodes=2, batch_size=4)  # Very small for testing

            trainer = PPOTrainer(
                rl_config=rl_config, log_dir=self.log_dir, n_envs=1, seed=42  # Single environment for testing
            )

            assert trainer.config.learning_rate == 0.001
            assert trainer.log_dir.exists()
            assert trainer.n_envs == 1
            assert trainer.seed == 42

            # Test configuration saving
            trainer.save_config()
            config_path = trainer.log_dir / "training_config.pkl"
            assert config_path.exists()

            # Test cleanup
            trainer.cleanup()

        except ImportError:
            pytest.skip("PPO trainer dependencies not available")

    def test_ppo_trainer_environment_creation(self):
        """Test PPO trainer environment creation."""
        try:
            from psireg.rl.train import _SB3_AVAILABLE, PPOTrainer

            if not _SB3_AVAILABLE:
                pytest.skip("Stable-baselines3 dependencies not available")

            trainer = PPOTrainer(log_dir=self.log_dir, n_envs=1, seed=42)

            # Create environments
            trainer.create_environments()

            assert trainer.env is not None
            assert trainer.eval_env is not None

            # Test environment functionality
            obs = trainer.env.reset()
            assert obs is not None

            # Test action space
            action = trainer.env.action_space.sample()
            # Reshape action for vectorized environment if needed
            if len(action.shape) == 1:
                action = action.reshape(1, -1)
            obs, reward, done, info = trainer.env.step(action)

            assert obs is not None
            assert isinstance(reward, int | float | np.ndarray)

            # Cleanup
            trainer.cleanup()

        except ImportError:
            pytest.skip("PPO trainer dependencies not available")


class TestGridPredictorIntegration:
    """Test GridPredictor integration (if dependencies available)."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_grid_predictor_mock_model(self):
        """Test GridPredictor with mock model."""
        try:
            from psireg.rl.infer import _SB3_AVAILABLE, GridPredictor

            if not _SB3_AVAILABLE:
                pytest.skip("Stable-baselines3 dependencies not available")

            # This test would require creating a mock model file
            # For now, just test that the class can be imported
            assert GridPredictor is not None

        except ImportError:
            pytest.skip("GridPredictor dependencies not available")

    def test_grid_predictor_functionality_concepts(self):
        """Test GridPredictor functionality concepts without dependencies."""
        # Test the concepts that would be used in GridPredictor

        # Mock observation
        observation = np.array(
            [
                1.0,  # frequency (normalized)
                0.15,  # generation (normalized)
                0.12,  # load (normalized)
                0.02,  # storage (normalized)
                0.003,  # losses (normalized)
                0.03,  # balance (normalized)
                0.5,  # battery soc
                0.8,  # load factor
                0.5,  # hour of day
                0.3,  # day of week
            ],
            dtype=np.float32,
        )

        # Mock action
        action = np.array([0.3, -0.2], dtype=np.float32)  # Battery charge, load reduce

        # Test observation validation
        assert observation.dtype == np.float32
        assert len(observation) == 10
        assert np.all(np.isfinite(observation))

        # Test action validation
        assert action.dtype == np.float32
        assert len(action) == 2
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

        # Test action interpretation concept
        battery_action = action[0]
        load_action = action[1]

        if battery_action > 0:
            battery_interpretation = f"Charge at {battery_action*100:.1f}% rate"
        else:
            battery_interpretation = f"Discharge at {abs(battery_action)*100:.1f}% rate"

        if load_action > 0:
            load_interpretation = f"Increase demand by {load_action*100:.1f}%"
        else:
            load_interpretation = f"Reduce demand by {abs(load_action)*100:.1f}%"

        assert "Charge" in battery_interpretation
        assert "Reduce" in load_interpretation


class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_complete_workflow_concept(self):
        """Test complete workflow concept without requiring RL dependencies."""
        # This tests the overall workflow that would be used with RL

        # 1. Create grid configuration
        sim_config = SimulationConfig(timestep_minutes=15)
        grid_config = GridConfig(frequency_hz=60.0)
        rl_config = RLConfig(learning_rate=0.001, training_episodes=10)

        # Verify RL config was created
        assert rl_config.learning_rate == 0.001
        assert rl_config.training_episodes == 10

        # 2. Create GridEngine
        engine = GridEngine(sim_config, grid_config)

        # 3. Set up grid
        node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
        engine.add_node(node)

        battery = Battery(
            asset_id="battery_1", name="Battery 1", node_id="test_node", capacity_mw=50.0, energy_capacity_mwh=200.0
        )
        load = Load(asset_id="load_1", name="Load 1", node_id="test_node", capacity_mw=80.0, baseline_demand_mw=60.0)

        engine.add_asset(battery)
        engine.add_asset(load)

        battery.set_status(AssetStatus.ONLINE)
        load.set_status(AssetStatus.ONLINE)

        # 4. Simulate RL-like control loop
        for step in range(5):
            # Get state (would be observation in RL)
            grid_state = engine.get_state()
            assert grid_state is not None

            # Simulate action prediction (mock RL agent decision)
            battery_action = 0.5 if step % 2 == 0 else -0.3
            load_action = -0.2 if step % 3 == 0 else 0.1

            # Apply actions to assets
            max_charge = battery.get_max_charge_power()
            max_discharge = battery.get_max_discharge_power()

            if battery_action >= 0:
                power_setpoint = battery_action * max_charge
            else:
                power_setpoint = battery_action * max_discharge

            battery.set_power_setpoint(power_setpoint)

            # Apply load action
            baseline_demand = load.baseline_demand_mw
            dr_capability = baseline_demand * 0.2
            dr_signal = load_action * dr_capability
            load.set_demand_response_signal(dr_signal)

            # Step simulation
            engine.step(timedelta(minutes=15))

            # Verify state changed
            new_state = engine.get_state()
            assert new_state.timestamp != grid_state.timestamp

        # 5. Verify final state
        final_state = engine.get_state()
        assert isinstance(final_state.frequency_hz, float)
        assert isinstance(final_state.total_generation_mw, float)
        assert isinstance(final_state.total_load_mw, float)

    def test_performance_benchmarks(self):
        """Test performance benchmarks for RL components."""
        # Test observation space construction performance
        sim_config = SimulationConfig(timestep_minutes=15)
        grid_config = GridConfig(frequency_hz=60.0)

        engine = GridEngine(sim_config, grid_config)
        node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
        engine.add_node(node)

        # Add multiple assets
        for i in range(10):
            battery = Battery(
                asset_id=f"battery_{i}",
                name=f"Battery {i}",
                node_id="test_node",
                capacity_mw=50.0,
                energy_capacity_mwh=200.0,
            )
            engine.add_asset(battery)
            battery.set_status(AssetStatus.ONLINE)

        # Benchmark observation construction
        start_time = datetime.now()

        for _ in range(100):
            grid_state = engine.get_state()

            # Construct observation-like vector
            observations = [
                grid_state.frequency_hz / 60.0,
                grid_state.total_generation_mw / 1000.0,
                grid_state.total_load_mw / 1000.0,
                grid_state.total_storage_mw / 1000.0,
                grid_state.grid_losses_mw / 1000.0,
                grid_state.power_balance_mw / 1000.0,
            ]

            # Add asset states
            for asset in engine.assets.values():
                observations.extend([asset.current_output_mw / asset.capacity_mw, 1.0])  # placeholder

            obs_array = np.array(observations, dtype=np.float32)
            assert len(obs_array) > 0

        construction_time = (datetime.now() - start_time).total_seconds()

        # Should be able to construct 100 observations in under 1 second
        assert construction_time < 1.0

        # Benchmark action application
        start_time = datetime.now()

        for _ in range(100):
            for i, asset in enumerate(engine.assets.values()):
                if isinstance(asset, Battery):
                    action_value = 0.5 if i % 2 == 0 else -0.3
                    max_charge = asset.get_max_charge_power()
                    max_discharge = asset.get_max_discharge_power()

                    if action_value >= 0:
                        power_setpoint = action_value * max_charge
                    else:
                        power_setpoint = action_value * max_discharge

                    asset.set_power_setpoint(power_setpoint)

        action_time = (datetime.now() - start_time).total_seconds()

        # Should be able to apply 100 action sets in under 1 second
        assert action_time < 1.0

    def test_configuration_consistency(self):
        """Test configuration consistency across RL components."""
        # Test that configurations are consistent between components
        rl_config = RLConfig(learning_rate=0.001, gamma=0.95, batch_size=32, training_episodes=1000)

        sim_config = SimulationConfig(timestep_minutes=15, horizon_hours=24)

        grid_config = GridConfig(frequency_hz=60.0, max_power_mw=1000.0)

        # Verify configuration values
        assert rl_config.learning_rate == 0.001
        assert rl_config.gamma == 0.95
        assert sim_config.timestep_minutes == 15
        assert grid_config.frequency_hz == 60.0

        # Test configuration serialization/deserialization
        rl_dict = rl_config.model_dump()
        sim_dict = sim_config.model_dump()
        grid_dict = grid_config.model_dump()

        # Reconstruct from dictionaries
        rl_config_2 = RLConfig(**rl_dict)
        sim_config_2 = SimulationConfig(**sim_dict)
        grid_config_2 = GridConfig(**grid_dict)

        assert rl_config_2.learning_rate == rl_config.learning_rate
        assert sim_config_2.timestep_minutes == sim_config.timestep_minutes
        assert grid_config_2.frequency_hz == grid_config.frequency_hz
