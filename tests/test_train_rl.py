"""Tests for train_rl.py CLI training script.

This module tests the standalone CLI training script functionality including:
- Command-line argument parsing
- Configuration management
- Integration with existing PPOTrainer
- Model training execution
- Predictor model output
- Error handling and validation
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psireg.config.schema import GridConfig, RLConfig, SimulationConfig


class TestTrainRLCLI:
    """Test CLI argument parsing and configuration."""

    def test_parse_basic_arguments(self):
        """Test basic argument parsing."""
        # Import the argument parser (this will be implemented)
        from train_rl import create_argument_parser

        parser = create_argument_parser()

        # Test default arguments
        args = parser.parse_args([])
        assert args.log_dir == "logs/ppo_training"
        assert args.episodes == 1000
        assert args.timesteps is None
        assert args.n_envs == 4
        assert args.learning_rate == 0.001
        assert args.gamma == 0.95
        assert args.resume is False
        assert args.config_file is None

    def test_parse_custom_arguments(self):
        """Test parsing custom arguments."""
        from train_rl import create_argument_parser

        parser = create_argument_parser()

        args = parser.parse_args(
            [
                "--log-dir",
                "custom_logs",
                "--episodes",
                "500",
                "--timesteps",
                "50000",
                "--n-envs",
                "8",
                "--learning-rate",
                "0.0001",
                "--gamma",
                "0.99",
                "--resume",
                "--config-file",
                "config.yaml",
            ]
        )

        assert args.log_dir == "custom_logs"
        assert args.episodes == 500
        assert args.timesteps == 50000
        assert args.n_envs == 8
        assert args.learning_rate == 0.0001
        assert args.gamma == 0.99
        assert args.resume is True
        assert args.config_file == "config.yaml"

    def test_parse_model_path_arguments(self):
        """Test model path arguments."""
        from train_rl import create_argument_parser

        parser = create_argument_parser()

        args = parser.parse_args(
            ["--model-path", "models/existing_model.zip", "--output-path", "models/trained_model.zip"]
        )

        assert args.model_path == "models/existing_model.zip"
        assert args.output_path == "models/trained_model.zip"


class TestTrainRLConfiguration:
    """Test configuration management for training script."""

    def test_create_config_from_args(self):
        """Test creating configuration from command line arguments."""
        from train_rl import create_configs_from_args

        # Mock args object
        args = argparse.Namespace(learning_rate=0.001, gamma=0.95, episodes=1000, batch_size=32, config_file=None)

        rl_config, sim_config, grid_config = create_configs_from_args(args)

        assert isinstance(rl_config, RLConfig)
        assert isinstance(sim_config, SimulationConfig)
        assert isinstance(grid_config, GridConfig)

        assert rl_config.learning_rate == 0.001
        assert rl_config.gamma == 0.95
        assert rl_config.training_episodes == 1000

    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        from train_rl import load_config_from_file

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
rl:
  learning_rate: 0.0005
  gamma: 0.98
  training_episodes: 2000
  batch_size: 64
simulation:
  timestep_minutes: 15
  start_time: "2023-01-01T00:00:00"
grid:
  frequency_hz: 60.0
  voltage_tolerance: 0.05
"""
            )
            config_file = f.name

        try:
            rl_config, sim_config, grid_config = load_config_from_file(config_file)

            assert rl_config.learning_rate == 0.0005
            assert rl_config.gamma == 0.98
            assert rl_config.training_episodes == 2000
            assert rl_config.batch_size == 64
            assert sim_config.timestep_minutes == 15
            assert grid_config.frequency_hz == 60.0

        finally:
            os.unlink(config_file)

    def test_config_validation(self):
        """Test configuration validation."""
        from train_rl import validate_config

        # Test valid config
        config = RLConfig(learning_rate=0.001, gamma=0.95)
        validate_config(config)  # Should not raise

        # Test invalid config - Pydantic will catch this during creation
        with pytest.raises(ValueError):  # Catch the broader ValidationError
            invalid_config = RLConfig(learning_rate=-0.001)
            validate_config(invalid_config)


class TestTrainRLExecution:
    """Test training execution functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("train_rl.PPOTrainer")
    def test_training_execution(self, mock_trainer_class):
        """Test actual training execution."""
        from train_rl import run_training

        # Mock trainer instance
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train.return_value = None
        mock_trainer.evaluate.return_value = {"mean_reward": 100.0}

        # Create expected model file
        os.makedirs(self.log_dir, exist_ok=True)
        final_model_path = os.path.join(self.log_dir, "final_model.zip")
        Path(final_model_path).touch()

        # Mock args
        args = argparse.Namespace(
            log_dir=self.log_dir,
            episodes=100,
            timesteps=None,
            n_envs=2,
            learning_rate=0.001,
            gamma=0.95,
            resume=False,
            model_path=None,
            output_path=None,
            config_file=None,
            seed=42,
            no_eval=False,
            eval_episodes=10,
        )

        # Run training
        result = run_training(args)

        # Verify trainer was created with correct parameters
        mock_trainer_class.assert_called_once()
        call_args = mock_trainer_class.call_args
        assert call_args[1]["log_dir"] == self.log_dir
        assert call_args[1]["n_envs"] == 2
        assert call_args[1]["seed"] == 42

        # Verify training methods were called
        mock_trainer.train.assert_called_once()
        mock_trainer.evaluate.assert_called_once()

        # Verify result
        assert result is not None
        assert "model_path" in result
        assert "evaluation_results" in result

    @patch("train_rl.PPOTrainer")
    def test_resume_training(self, mock_trainer_class):
        """Test resuming training from existing model."""
        from train_rl import run_training

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        # Create dummy model file
        model_path = os.path.join(self.temp_dir, "existing_model.zip")
        Path(model_path).touch()

        # Create expected final model file
        os.makedirs(self.log_dir, exist_ok=True)
        final_model_path = os.path.join(self.log_dir, "final_model.zip")
        Path(final_model_path).touch()

        args = argparse.Namespace(
            log_dir=self.log_dir,
            episodes=100,
            timesteps=None,
            n_envs=2,
            learning_rate=0.001,
            gamma=0.95,
            resume=True,
            model_path=model_path,
            output_path=None,
            config_file=None,
            seed=42,
            no_eval=False,
            eval_episodes=10,
        )

        run_training(args)

        # Verify training was called with resume=True
        mock_trainer.train.assert_called_once()
        train_call_args = mock_trainer.train.call_args
        assert train_call_args[1]["resume"] is True
        assert train_call_args[1]["model_path"] == model_path

    def test_error_handling_missing_model(self):
        """Test error handling when model file doesn't exist."""
        from train_rl import run_training

        args = argparse.Namespace(
            log_dir=self.log_dir,
            episodes=100,
            timesteps=None,
            n_envs=2,
            learning_rate=0.001,
            gamma=0.95,
            resume=True,
            model_path="nonexistent_model.zip",
            output_path=None,
            config_file=None,
            seed=42,
        )

        with pytest.raises(FileNotFoundError):
            run_training(args)


class TestTrainRLOutput:
    """Test output generation and predictor model creation."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predictor_model_creation(self):
        """Test that predictor model is created correctly."""
        from train_rl import create_predictor_model

        # Create dummy trained model
        model_path = os.path.join(self.log_dir, "final_model.zip")
        config_path = os.path.join(self.log_dir, "training_config.pkl")

        Path(model_path).touch()
        Path(config_path).touch()

        # Test predictor creation
        predictor_info = create_predictor_model(model_path, config_path)

        assert predictor_info is not None
        assert predictor_info["model_path"] == model_path
        assert predictor_info["config_path"] == config_path
        assert "creation_time" in predictor_info

    def test_training_summary_generation(self):
        """Test generation of training summary."""
        from train_rl import generate_training_summary

        # Mock training results
        training_results = {
            "model_path": os.path.join(self.log_dir, "final_model.zip"),
            "evaluation_results": {
                "mean_reward": 150.0,
                "std_reward": 25.0,
                "mean_episode_length": 96.0,
                "mean_frequency_deviation": 0.05,
                "mean_power_imbalance": 2.5,
            },
            "training_time": 3600.0,
            "total_timesteps": 50000,
        }

        summary = generate_training_summary(training_results)

        assert summary is not None
        assert "Training completed successfully" in summary
        assert "Model saved to" in summary
        assert "Mean reward: 150.0" in summary
        assert "Training time: 1.0 hours" in summary

    def test_model_validation(self):
        """Test validation of trained model."""
        from train_rl import validate_trained_model

        # Create mock model file
        model_path = os.path.join(self.log_dir, "final_model.zip")

        # Test with non-existent model
        with pytest.raises(FileNotFoundError):
            validate_trained_model("nonexistent_model.zip")

        # Test with existing model (empty file for test)
        Path(model_path).touch()

        # Should not raise exception
        validate_trained_model(model_path)


class TestTrainRLIntegration:
    """Test integration with existing components."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_integration_with_ppo_trainer(self):
        """Test integration with existing PPOTrainer."""
        from train_rl import setup_trainer

        rl_config = RLConfig(learning_rate=0.001, gamma=0.95)
        sim_config = SimulationConfig()
        grid_config = GridConfig()

        # Mock trainer setup
        with patch("train_rl.PPOTrainer") as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer_class.return_value = mock_trainer

            trainer = setup_trainer(
                rl_config=rl_config,
                sim_config=sim_config,
                grid_config=grid_config,
                log_dir=self.temp_dir,
                n_envs=4,
                seed=42,
            )

            # Verify trainer was created correctly
            mock_trainer_class.assert_called_once_with(
                rl_config=rl_config,
                simulation_config=sim_config,
                grid_config=grid_config,
                log_dir=self.temp_dir,
                n_envs=4,
                seed=42,
            )

            assert trainer == mock_trainer

    def test_integration_with_grid_predictor(self):
        """Test integration with GridPredictor."""
        from train_rl import setup_predictor

        model_path = os.path.join(self.temp_dir, "model.zip")
        config_path = os.path.join(self.temp_dir, "config.pkl")

        # Create dummy files
        Path(model_path).touch()
        Path(config_path).touch()

        # Mock predictor setup
        with patch("train_rl.GridPredictor") as mock_predictor_class:
            mock_predictor = MagicMock()
            mock_predictor_class.return_value = mock_predictor

            predictor = setup_predictor(model_path, config_path)

            # Verify predictor was created correctly
            mock_predictor_class.assert_called_once_with(model_path=model_path, config_path=config_path)

            assert predictor == mock_predictor


class TestTrainRLMainFunction:
    """Test main execution function."""

    def test_main_function_execution(self):
        """Test main function execution flow."""
        from train_rl import main

        # Mock command line arguments
        test_args = ["--log-dir", "test_logs", "--episodes", "50", "--n-envs", "1", "--learning-rate", "0.001"]

        with patch("sys.argv", ["train_rl.py"] + test_args):
            with patch("train_rl.run_training") as mock_run_training:
                mock_run_training.return_value = {
                    "model_path": "test_model.zip",
                    "evaluation_results": {"mean_reward": 100.0},
                    "training_time": 3600.0,
                    "total_timesteps": 50000,
                    "predictor_info": {
                        "predictor_ready": True,
                        "model_path": "test_model.zip",
                        "config_path": "test_config.pkl",
                        "creation_time": "2023-01-01T00:00:00",
                    },
                }

                # Should not raise exception
                main()

                # Verify run_training was called
                mock_run_training.assert_called_once()

    def test_main_function_error_handling(self):
        """Test main function error handling."""
        from train_rl import main

        with patch("sys.argv", ["train_rl.py", "--invalid-arg"]):
            with pytest.raises(SystemExit):
                main()


# Skip tests if RL dependencies not available
_RL_AVAILABLE = True
try:
    # Check if RL dependencies are available
    import importlib.util

    for lib in ["gymnasium", "stable_baselines3", "torch"]:
        if importlib.util.find_spec(lib) is None:
            _RL_AVAILABLE = False
            break
except ImportError:
    _RL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _RL_AVAILABLE, reason="RL dependencies (gymnasium, stable-baselines3, torch) not available"
)
