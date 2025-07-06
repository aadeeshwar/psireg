"""Tests for PPO Inference Service - GridPredictor prediction functionality.

This test suite covers:
- GridPredictor model loading and initialization
- Core predict(obs) functionality
- Batch prediction capabilities
- Performance monitoring and metrics
- Integration with GridEngine
- Edge cases and error handling
"""

import importlib.util
import os
import pickle
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psireg.config.schema import GridConfig, RLConfig, SimulationConfig
from psireg.sim.assets.battery import Battery
from psireg.sim.engine import GridEngine, GridState, NetworkNode
from psireg.utils.enums import AssetStatus

# Check if RL dependencies are available
_RL_AVAILABLE = (
    importlib.util.find_spec("gymnasium") is not None
    and importlib.util.find_spec("stable_baselines3") is not None
    and importlib.util.find_spec("torch") is not None
)


class TestGridPredictorInitialization:
    """Test GridPredictor initialization and model loading."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")
        self.config_path = os.path.join(self.temp_dir, "test_config.pkl")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predictor_initialization_with_model(self):
        """Test GridPredictor initialization with model file."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        # Create mock model and config files
        Path(self.model_path).touch()
        self._create_mock_config()

        # Test initialization should load model and config
        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            assert predictor.model_path == Path(self.model_path)
            assert predictor.config_path == Path(self.config_path)
            assert predictor.model is not None
            assert predictor.config is not None
            assert isinstance(predictor.prediction_history, list)
            assert isinstance(predictor.performance_metrics, dict)

    def test_predictor_initialization_without_config(self):
        """Test GridPredictor initialization without config file."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        # Create mock model file only
        Path(self.model_path).touch()

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path)

            assert predictor.model is not None
            assert predictor.config is not None  # Should use default config
            assert isinstance(predictor.config, RLConfig)

    def test_predictor_initialization_missing_model(self):
        """Test GridPredictor initialization with missing model file."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        # Test with non-existent model file
        with pytest.raises(FileNotFoundError):
            GridPredictor(model_path="nonexistent_model.zip")

    def test_predictor_initialization_without_dependencies(self):
        """Test GridPredictor initialization without RL dependencies."""
        # Test the import error when dependencies aren't available
        with patch("psireg.rl.infer._SB3_AVAILABLE", False):
            from psireg.rl.infer import GridPredictor

            with pytest.raises(ImportError, match="stable-baselines3 is required"):
                GridPredictor(model_path=self.model_path)

    def _create_mock_config(self):
        """Create mock configuration file."""
        config_data = {
            "rl_config": RLConfig().model_dump(),
            "simulation_config": SimulationConfig().model_dump(),
            "grid_config": GridConfig().model_dump(),
        }

        with open(self.config_path, "wb") as f:
            pickle.dump(config_data, f)


class TestGridPredictorPrediction:
    """Test core prediction functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")
        self.config_path = os.path.join(self.temp_dir, "test_config.pkl")

        # Create mock files
        Path(self.model_path).touch()
        self._create_mock_config()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predict_action_with_observation(self):
        """Test predict_action with direct observation input."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            # Setup mock model
            mock_model = MagicMock()
            mock_action = np.array([0.5, -0.3])
            mock_model.predict.return_value = (mock_action, None)
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            # Test prediction with observation
            observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)
            action, info = predictor.predict_action(observation=observation)

            # Verify outputs
            assert isinstance(action, np.ndarray)
            assert action.shape == (2,)  # Flattened action
            assert isinstance(info, dict)
            assert "timestamp" in info
            assert "deterministic" in info
            assert "prediction_time_s" in info
            assert "observation_shape" in info
            assert "action_shape" in info

            # Verify model was called correctly
            mock_model.predict.assert_called_once()
            call_args = mock_model.predict.call_args
            assert np.array_equal(call_args[0][0], observation.reshape(1, -1))

    def test_predict_action_deterministic_vs_stochastic(self):
        """Test predict_action with deterministic vs stochastic policy."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_action = np.array([0.5, -0.3])
            mock_model.predict.return_value = (mock_action, None)
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

            # Test deterministic prediction
            action_det, info_det = predictor.predict_action(observation=observation, deterministic=True)

            # Test stochastic prediction
            action_stoch, info_stoch = predictor.predict_action(observation=observation, deterministic=False)

            # Verify deterministic flag was passed correctly
            assert info_det["deterministic"] is True
            assert info_stoch["deterministic"] is False

            # Verify model was called with correct deterministic flag
            assert mock_model.predict.call_count == 2
            calls = mock_model.predict.call_args_list
            assert calls[0][1]["deterministic"] is True
            assert calls[1][1]["deterministic"] is False

    def test_predict_action_with_grid_state(self):
        """Test predict_action with GridState input."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_action = np.array([0.5, -0.3])
            mock_model.predict.return_value = (mock_action, None)
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            # Mock GridEnv for observation computation
            mock_grid_env = MagicMock()
            mock_observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)
            mock_grid_env._get_observation.return_value = mock_observation
            predictor.grid_env = mock_grid_env

            # Create mock GridState
            grid_state = GridState(
                timestamp=datetime.now(),
                frequency_hz=60.0,
                total_generation_mw=1000.0,
                total_load_mw=950.0,
                total_storage_mw=50.0,
                grid_losses_mw=25.0,
            )

            # Test prediction with grid state
            action, info = predictor.predict_action(grid_state=grid_state)

            # Verify observation was computed from grid state
            mock_grid_env._get_observation.assert_called_once()
            assert isinstance(action, np.ndarray)
            assert isinstance(info, dict)

    def test_predict_action_missing_inputs(self):
        """Test predict_action with missing required inputs."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            # Test with no observation and no grid_state
            with pytest.raises(ValueError, match="Either observation or grid_state must be provided"):
                predictor.predict_action()

    def test_predict_action_history_tracking(self):
        """Test that prediction history is properly tracked."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_action = np.array([0.5, -0.3])
            mock_model.predict.return_value = (mock_action, None)
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

            # Make multiple predictions
            for _ in range(3):
                action, info = predictor.predict_action(observation=observation)

            # Verify history tracking
            assert len(predictor.prediction_history) == 3

            for entry in predictor.prediction_history:
                assert "observation" in entry
                assert "action" in entry
                assert "info" in entry
                assert isinstance(entry["observation"], np.ndarray)
                assert isinstance(entry["action"], np.ndarray)
                assert isinstance(entry["info"], dict)

    def test_predict_action_history_limit(self):
        """Test that prediction history is limited to prevent memory issues."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_action = np.array([0.5, -0.3])
            mock_model.predict.return_value = (mock_action, None)
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

            # Make more predictions than the history limit (1000)
            for _ in range(1050):
                action, info = predictor.predict_action(observation=observation)

            # Verify history is limited
            assert len(predictor.prediction_history) == 1000

    def _create_mock_config(self):
        """Create mock configuration file."""
        config_data = {
            "rl_config": RLConfig().model_dump(),
            "simulation_config": SimulationConfig().model_dump(),
            "grid_config": GridConfig().model_dump(),
        }

        with open(self.config_path, "wb") as f:
            pickle.dump(config_data, f)


class TestGridPredictorBatchPrediction:
    """Test batch prediction functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")
        self.config_path = os.path.join(self.temp_dir, "test_config.pkl")

        # Create mock files
        Path(self.model_path).touch()
        self._create_mock_config()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predict_batch_basic(self):
        """Test basic batch prediction functionality."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_action = np.array([0.5, -0.3])
            mock_model.predict.return_value = (mock_action, None)
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            # Create batch of observations
            observations = [
                np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32),
                np.array([0.8, 0.6, 0.4, 0.3, 0.2], dtype=np.float32),
                np.array([0.9, 0.4, 0.2, 0.1, 0.05], dtype=np.float32),
            ]

            # Test batch prediction
            actions, infos = predictor.predict_batch(observations)

            # Verify outputs
            assert isinstance(actions, list)
            assert isinstance(infos, list)
            assert len(actions) == len(observations)
            assert len(infos) == len(observations)

            # Verify each action and info
            for action, info in zip(actions, infos, strict=True):
                assert isinstance(action, np.ndarray)
                assert isinstance(info, dict)
                assert "timestamp" in info
                assert "deterministic" in info
                assert "prediction_time_s" in info

    def test_predict_batch_performance_metrics(self):
        """Test batch prediction performance metrics tracking."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_action = np.array([0.5, -0.3])
            mock_model.predict.return_value = (mock_action, None)
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            # Create batch of observations
            observations = [
                np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32),
                np.array([0.8, 0.6, 0.4, 0.3, 0.2], dtype=np.float32),
            ]

            # Test batch prediction
            actions, infos = predictor.predict_batch(observations)

            # Verify performance metrics were updated
            assert "batch_size" in predictor.performance_metrics
            assert "total_batch_time_s" in predictor.performance_metrics
            assert "avg_prediction_time_s" in predictor.performance_metrics
            assert "predictions_per_second" in predictor.performance_metrics

            assert predictor.performance_metrics["batch_size"] == 2
            assert predictor.performance_metrics["total_batch_time_s"] >= 0
            assert predictor.performance_metrics["avg_prediction_time_s"] >= 0
            assert predictor.performance_metrics["predictions_per_second"] >= 0

    def test_predict_batch_empty_input(self):
        """Test batch prediction with empty input."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            # Test with empty batch
            observations = []
            actions, infos = predictor.predict_batch(observations)

            # Verify outputs
            assert isinstance(actions, list)
            assert isinstance(infos, list)
            assert len(actions) == 0
            assert len(infos) == 0

    def _create_mock_config(self):
        """Create mock configuration file."""
        config_data = {
            "rl_config": RLConfig().model_dump(),
            "simulation_config": SimulationConfig().model_dump(),
            "grid_config": GridConfig().model_dump(),
        }

        with open(self.config_path, "wb") as f:
            pickle.dump(config_data, f)


class TestGridPredictorScenarioEvaluation:
    """Test scenario evaluation functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")
        self.config_path = os.path.join(self.temp_dir, "test_config.pkl")

        # Create mock files
        Path(self.model_path).touch()
        self._create_mock_config()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_evaluate_scenario_basic(self):
        """Test basic scenario evaluation functionality."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_action = np.array([0.5, -0.3])
            mock_model.predict.return_value = (mock_action, None)
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            # Create test grid engine
            grid_engine = self._create_test_grid_engine()

            # Mock GridEnv
            mock_grid_env = MagicMock()
            mock_observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)
            mock_grid_env.reset.return_value = (mock_observation, {})
            mock_grid_env.step.return_value = (
                mock_observation,  # next_observation
                1.0,  # reward
                False,  # terminated
                False,  # truncated
                {"grid_state": {"frequency_hz": 60.0}},  # info
            )

            with patch("psireg.rl.infer.GridEnv", return_value=mock_grid_env):
                # Test scenario evaluation
                results = predictor.evaluate_scenario(
                    grid_engine=grid_engine, duration_hours=1, timestep_minutes=15, deterministic=True
                )

                # Verify results structure
                assert isinstance(results, dict)
                assert "total_reward" in results
                assert "mean_reward" in results
                assert "std_reward" in results
                assert "episode_length" in results

                # Verify evaluation ran
                assert mock_grid_env.reset.called
                assert mock_grid_env.step.called

    def test_evaluate_scenario_with_termination(self):
        """Test scenario evaluation with early termination."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_action = np.array([0.5, -0.3])
            mock_model.predict.return_value = (mock_action, None)
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            # Create test grid engine
            grid_engine = self._create_test_grid_engine()

            # Mock GridEnv with early termination
            mock_grid_env = MagicMock()
            mock_observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)
            mock_grid_env.reset.return_value = (mock_observation, {})

            # First step normal, second step terminates
            mock_grid_env.step.side_effect = [
                (mock_observation, 1.0, False, False, {"grid_state": {}}),
                (mock_observation, 1.0, True, False, {"grid_state": {}}),  # terminated
            ]

            with patch("psireg.rl.infer.GridEnv", return_value=mock_grid_env):
                # Test scenario evaluation
                results = predictor.evaluate_scenario(
                    grid_engine=grid_engine, duration_hours=1, timestep_minutes=15, deterministic=True
                )

                # Verify results structure
                assert isinstance(results, dict)
                assert results["episode_length"] == 2  # Should stop after 2 steps

    def _create_test_grid_engine(self):
        """Create a test grid engine."""
        sim_config = SimulationConfig(timestep_minutes=15)
        grid_config = GridConfig(frequency_hz=60.0)

        engine = GridEngine(sim_config, grid_config)

        # Add basic components
        node = NetworkNode(node_id="test_node", name="Test Node", voltage_kv=138.0)
        engine.add_node(node)

        battery = Battery(
            asset_id="battery_1", name="Battery 1", node_id="test_node", capacity_mw=50.0, energy_capacity_mwh=200.0
        )
        engine.add_asset(battery)
        battery.set_status(AssetStatus.ONLINE)

        return engine

    def _create_mock_config(self):
        """Create mock configuration file."""
        config_data = {
            "rl_config": RLConfig().model_dump(),
            "simulation_config": SimulationConfig().model_dump(),
            "grid_config": GridConfig().model_dump(),
        }

        with open(self.config_path, "wb") as f:
            pickle.dump(config_data, f)


class TestGridPredictorErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")
        self.config_path = os.path.join(self.temp_dir, "test_config.pkl")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predict_without_model(self):
        """Test prediction when model is not loaded."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        # Create predictor but don't load model
        Path(self.model_path).touch()

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_ppo.load.return_value = None

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)
            predictor.model = None  # Simulate failed model loading

            observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Model not loaded"):
                predictor.predict_action(observation=observation)

    def test_invalid_observation_format(self):
        """Test prediction with invalid observation format."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        Path(self.model_path).touch()
        self._create_mock_config()

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_action = np.array([0.5, -0.3])
            mock_model.predict.return_value = (mock_action, None)
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            # Test with invalid observation types
            invalid_observations = [
                "invalid_string",
                None,
                [1, 2, 3],  # list instead of numpy array
                np.array([]),  # empty array
                np.array([np.nan, 0.5, 0.3]),  # contains NaN
                np.array([np.inf, 0.5, 0.3]),  # contains inf
            ]

            for invalid_obs in invalid_observations:
                try:
                    action, info = predictor.predict_action(observation=invalid_obs)
                    # If it doesn't raise an error, the prediction should still work
                    # (due to numpy's type conversion capabilities)
                    assert isinstance(action, np.ndarray)
                    assert isinstance(info, dict)
                except (ValueError, TypeError):
                    # Expected for some invalid inputs
                    pass

    def test_model_prediction_failure(self):
        """Test handling of model prediction failures."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import GridPredictor

        Path(self.model_path).touch()
        self._create_mock_config()

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_model.predict.side_effect = RuntimeError("Model prediction failed")
            mock_ppo.load.return_value = mock_model

            predictor = GridPredictor(model_path=self.model_path, config_path=self.config_path)

            observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

            # Should propagate the model prediction error
            with pytest.raises(RuntimeError, match="Model prediction failed"):
                predictor.predict_action(observation=observation)

    def _create_mock_config(self):
        """Create mock configuration file."""
        config_data = {
            "rl_config": RLConfig().model_dump(),
            "simulation_config": SimulationConfig().model_dump(),
            "grid_config": GridConfig().model_dump(),
        }

        with open(self.config_path, "wb") as f:
            pickle.dump(config_data, f)


class TestGridPredictorConvenienceFunction:
    """Test convenience function for loading trained models."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")
        self.config_path = os.path.join(self.temp_dir, "test_config.pkl")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_trained_model_function(self):
        """Test load_trained_model convenience function."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import load_trained_model

        # Create mock files
        Path(self.model_path).touch()
        self._create_mock_config()

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_ppo.load.return_value = mock_model

            # Test convenience function
            predictor = load_trained_model(model_path=self.model_path, config_path=self.config_path, device="cpu")

            assert predictor is not None
            assert predictor.model_path == Path(self.model_path)
            assert predictor.config_path == Path(self.config_path)
            assert predictor.device == "cpu"

    def test_load_trained_model_without_config(self):
        """Test load_trained_model without configuration file."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        from psireg.rl.infer import load_trained_model

        # Create mock model file only
        Path(self.model_path).touch()

        with patch("psireg.rl.infer.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_ppo.load.return_value = mock_model

            # Test convenience function
            predictor = load_trained_model(model_path=self.model_path)

            assert predictor is not None
            assert predictor.model_path == Path(self.model_path)
            assert predictor.config_path is None

    def _create_mock_config(self):
        """Create mock configuration file."""
        config_data = {
            "rl_config": RLConfig().model_dump(),
            "simulation_config": SimulationConfig().model_dump(),
            "grid_config": GridConfig().model_dump(),
        }

        with open(self.config_path, "wb") as f:
            pickle.dump(config_data, f)


# Skip all tests if RL dependencies not available
pytestmark = pytest.mark.skipif(
    not _RL_AVAILABLE, reason="RL dependencies (gymnasium, stable-baselines3, torch) not available"
)
