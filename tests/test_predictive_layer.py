"""Tests for PredictiveLayer - PPO Inference Service API.

This test suite covers:
- PredictiveLayer initialization and model loading
- Core predict(obs) functionality
- Batch prediction capabilities
- Performance monitoring
- Fallback behavior when RL dependencies unavailable
- Error handling and edge cases
- Integration with existing GridPredictor
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psireg.rl.predictive_layer import (
    FallbackPredictor,
    GridPredictorWrapper,
    PredictiveLayer,
    load_predictor,
    predict_action,
)

# Check if RL dependencies are available
_RL_AVAILABLE = (
    importlib.util.find_spec("gymnasium") is not None
    and importlib.util.find_spec("stable_baselines3") is not None
    and importlib.util.find_spec("torch") is not None
)


class TestPredictiveLayerInitialization:
    """Test PredictiveLayer initialization and model loading."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")
        self.config_path = os.path.join(self.temp_dir, "test_config.pkl")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predictive_layer_basic_initialization(self):
        """Test basic PredictiveLayer initialization."""
        # Test with no predictor
        layer = PredictiveLayer()
        assert layer.predictor is None
        assert not layer.is_ready
        assert layer.model_path is None
        assert layer.performance_tracking is True
        assert layer._prediction_count == 0

    def test_predictive_layer_with_predictor(self):
        """Test PredictiveLayer initialization with predictor."""
        # Mock predictor
        mock_predictor = MagicMock()

        layer = PredictiveLayer(predictor=mock_predictor)
        assert layer.predictor is mock_predictor
        assert layer.is_ready is True
        assert layer.performance_tracking is True

    def test_load_model_with_rl_dependencies(self):
        """Test loading model when RL dependencies are available."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        # Create mock model file
        Path(self.model_path).touch()

        with patch("psireg.rl.predictive_layer.load_trained_model") as mock_load:
            mock_grid_predictor = MagicMock()
            mock_load.return_value = mock_grid_predictor

            layer = PredictiveLayer.load_model(model_path=self.model_path, config_path=self.config_path)

            assert layer.is_ready is True
            assert layer.model_path == Path(self.model_path)
            assert isinstance(layer.predictor, GridPredictorWrapper)
            mock_load.assert_called_once_with(model_path=self.model_path, config_path=self.config_path, device="auto")

    def test_load_model_without_rl_dependencies(self):
        """Test loading model when RL dependencies are not available."""
        with patch("psireg.rl.predictive_layer._RL_AVAILABLE", False):
            layer = PredictiveLayer.load_model(model_path=self.model_path)

            assert layer.is_ready is True
            assert layer.model_path == Path(self.model_path)
            assert isinstance(layer.predictor, FallbackPredictor)

    def test_load_model_failure_fallback(self):
        """Test fallback when model loading fails."""
        if not _RL_AVAILABLE:
            pytest.skip("RL dependencies not available")

        with patch("psireg.rl.predictive_layer.load_trained_model") as mock_load:
            mock_load.side_effect = Exception("Model loading failed")

            layer = PredictiveLayer.load_model(model_path=self.model_path)

            assert layer.is_ready is True
            assert isinstance(layer.predictor, FallbackPredictor)

    def test_model_info_and_status(self):
        """Test model information and status methods."""
        layer = PredictiveLayer()

        # Test when no model loaded
        assert not layer.is_model_loaded()

        info = layer.get_model_info()
        assert info["model_path"] is None
        assert info["is_ready"] is False
        assert info["predictor_type"] is None
        assert "rl_dependencies_available" in info
        assert info["total_predictions"] == 0

        # Test with mock predictor
        mock_predictor = MagicMock()
        layer = PredictiveLayer(predictor=mock_predictor)
        layer.model_path = Path(self.model_path)

        assert layer.is_model_loaded()

        info = layer.get_model_info()
        assert info["model_path"] == str(Path(self.model_path))
        assert info["is_ready"] is True
        assert info["predictor_type"] == "MagicMock"


class TestPredictiveLayerPrediction:
    """Test core prediction functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predict_basic_functionality(self):
        """Test basic predict functionality."""
        # Mock predictor
        mock_predictor = MagicMock()
        expected_action = np.array([0.5, -0.3], dtype=np.float32)
        mock_predictor.predict.return_value = expected_action

        layer = PredictiveLayer(predictor=mock_predictor)

        # Test prediction
        observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)
        action = layer.predict(observation)

        assert np.array_equal(action, expected_action)
        assert layer._prediction_count == 1
        mock_predictor.predict.assert_called_once_with(observation, deterministic=True)

    def test_predict_deterministic_vs_stochastic(self):
        """Test deterministic vs stochastic prediction."""
        mock_predictor = MagicMock()
        expected_action = np.array([0.5, -0.3], dtype=np.float32)
        mock_predictor.predict.return_value = expected_action

        layer = PredictiveLayer(predictor=mock_predictor)
        observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

        # Test deterministic
        action_det = layer.predict(observation, deterministic=True)
        assert np.array_equal(action_det, expected_action)

        # Test stochastic
        action_stoch = layer.predict(observation, deterministic=False)
        assert np.array_equal(action_stoch, expected_action)

        # Verify calls
        calls = mock_predictor.predict.call_args_list
        assert calls[0][1]["deterministic"] is True
        assert calls[1][1]["deterministic"] is False

    def test_predict_without_model(self):
        """Test prediction when no model is loaded."""
        layer = PredictiveLayer()
        observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

        with pytest.raises(RuntimeError, match="Predictor not ready"):
            layer.predict(observation)

    def test_predict_input_validation(self):
        """Test prediction input validation."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.array([0.5, -0.3])
        layer = PredictiveLayer(predictor=mock_predictor)

        # Test with list input (should be converted to numpy)
        observation_list = [1.0, 0.5, 0.3, 0.2, 0.1]
        action = layer.predict(observation_list)
        assert isinstance(action, np.ndarray)

        # Test with empty observation
        with pytest.raises(ValueError, match="Observation cannot be empty"):
            layer.predict(np.array([]))

    def test_predict_batch_basic(self):
        """Test basic batch prediction functionality."""
        mock_predictor = MagicMock()
        expected_actions = [
            np.array([0.5, -0.3], dtype=np.float32),
            np.array([0.2, -0.1], dtype=np.float32),
        ]
        mock_predictor.predict_batch.return_value = expected_actions

        layer = PredictiveLayer(predictor=mock_predictor)

        observations = [
            np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32),
            np.array([0.8, 0.6, 0.4, 0.3, 0.2], dtype=np.float32),
        ]

        actions = layer.predict_batch(observations)

        assert len(actions) == 2
        assert np.array_equal(actions[0], expected_actions[0])
        assert np.array_equal(actions[1], expected_actions[1])
        assert layer._prediction_count == 2

    def test_predict_batch_empty_input(self):
        """Test batch prediction with empty input."""
        mock_predictor = MagicMock()
        layer = PredictiveLayer(predictor=mock_predictor)

        actions = layer.predict_batch([])
        assert actions == []
        assert layer._prediction_count == 0

    def test_predict_batch_without_model(self):
        """Test batch prediction when no model is loaded."""
        layer = PredictiveLayer()
        observations = [np.array([1.0, 0.5, 0.3, 0.2, 0.1])]

        with pytest.raises(RuntimeError, match="Predictor not ready"):
            layer.predict_batch(observations)

    def test_performance_tracking_disabled(self):
        """Test prediction with performance tracking disabled."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.array([0.5, -0.3])

        layer = PredictiveLayer(predictor=mock_predictor, performance_tracking=False)
        observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

        layer.predict(observation)
        assert layer._prediction_count == 0  # Should not increment


class TestPredictiveLayerPerformanceMetrics:
    """Test performance metrics functionality."""

    def test_get_performance_metrics_with_predictor(self):
        """Test getting performance metrics when predictor is available."""
        mock_predictor = MagicMock()
        mock_metrics = {"avg_prediction_time": 0.01, "total_predictions": 100, "model_accuracy": 0.95}
        mock_predictor.get_performance_metrics.return_value = mock_metrics

        layer = PredictiveLayer(predictor=mock_predictor)
        layer.model_path = Path("test_model.zip")
        layer._prediction_count = 50

        metrics = layer.get_performance_metrics()

        # Should include predictor metrics
        assert metrics["avg_prediction_time"] == 0.01
        assert metrics["model_accuracy"] == 0.95

        # Should include layer-specific metrics
        assert metrics["total_predictions"] == 50  # Layer count, not predictor count
        assert metrics["model_path"] == "test_model.zip"
        assert metrics["predictor_ready"] is True
        assert metrics["performance_tracking_enabled"] is True

    def test_get_performance_metrics_without_predictor(self):
        """Test getting performance metrics when no predictor is available."""
        layer = PredictiveLayer()
        metrics = layer.get_performance_metrics()
        assert metrics["error"] == "Predictor not ready"


class TestFallbackPredictor:
    """Test fallback predictor functionality."""

    def test_fallback_predictor_initialization(self):
        """Test fallback predictor initialization."""
        model_path = "test_model.zip"
        predictor = FallbackPredictor(model_path)

        assert predictor.model_path == model_path
        assert predictor.prediction_count == 0

    def test_fallback_predict_functionality(self):
        """Test fallback prediction functionality."""
        predictor = FallbackPredictor("test_model.zip")
        observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

        action = predictor.predict(observation)

        assert isinstance(action, np.ndarray)
        assert action.dtype == np.float32
        assert len(action) >= 2  # Should estimate reasonable action size
        assert np.all(action >= -0.1) and np.all(action <= 0.1)  # Conservative actions
        assert predictor.prediction_count == 1

    def test_fallback_predict_batch(self):
        """Test fallback batch prediction."""
        predictor = FallbackPredictor("test_model.zip")
        observations = [
            np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32),
            np.array([0.8, 0.6, 0.4, 0.3, 0.2], dtype=np.float32),
        ]

        actions = predictor.predict_batch(observations)

        assert len(actions) == 2
        assert all(isinstance(action, np.ndarray) for action in actions)
        assert predictor.prediction_count == 2

    def test_fallback_performance_metrics(self):
        """Test fallback performance metrics."""
        predictor = FallbackPredictor("test_model.zip")
        observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

        # Make some predictions
        predictor.predict(observation)
        predictor.predict(observation)

        metrics = predictor.get_performance_metrics()

        assert metrics["type"] == "fallback"
        assert metrics["model_path"] == "test_model.zip"
        assert metrics["prediction_count"] == 2
        assert "warning" in metrics


class TestGridPredictorWrapper:
    """Test GridPredictor wrapper functionality."""

    def test_wrapper_predict_functionality(self):
        """Test wrapper prediction functionality."""
        # Mock GridPredictor
        mock_grid_predictor = MagicMock()
        expected_action = np.array([0.5, -0.3], dtype=np.float32)
        mock_grid_predictor.predict_action.return_value = (expected_action, {"info": "test"})

        wrapper = GridPredictorWrapper(mock_grid_predictor)
        observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

        action = wrapper.predict(observation, deterministic=True)

        assert np.array_equal(action, expected_action)
        mock_grid_predictor.predict_action.assert_called_once_with(observation=observation, deterministic=True)

    def test_wrapper_predict_batch(self):
        """Test wrapper batch prediction functionality."""
        mock_grid_predictor = MagicMock()
        expected_actions = [
            np.array([0.5, -0.3], dtype=np.float32),
            np.array([0.2, -0.1], dtype=np.float32),
        ]
        mock_grid_predictor.predict_batch.return_value = (expected_actions, [{"info": "test"}] * 2)

        wrapper = GridPredictorWrapper(mock_grid_predictor)
        observations = [
            np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32),
            np.array([0.8, 0.6, 0.4, 0.3, 0.2], dtype=np.float32),
        ]

        actions = wrapper.predict_batch(observations, deterministic=False)

        assert len(actions) == 2
        assert np.array_equal(actions[0], expected_actions[0])
        mock_grid_predictor.predict_batch.assert_called_once_with(observations=observations, deterministic=False)

    def test_wrapper_performance_metrics(self):
        """Test wrapper performance metrics."""
        mock_grid_predictor = MagicMock()
        mock_metrics = {"accuracy": 0.95, "speed": 100}
        mock_grid_predictor.performance_metrics = mock_metrics

        wrapper = GridPredictorWrapper(mock_grid_predictor)
        metrics = wrapper.get_performance_metrics()

        assert metrics == mock_metrics


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_predictor_function(self):
        """Test load_predictor convenience function."""
        with patch.object(PredictiveLayer, "load_model") as mock_load:
            mock_layer = MagicMock()
            mock_load.return_value = mock_layer

            result = load_predictor(model_path=self.model_path, config_path="config.pkl", device="cpu")

            assert result == mock_layer
            mock_load.assert_called_once_with(model_path=self.model_path, config_path="config.pkl", device="cpu")

    def test_predict_action_function(self):
        """Test predict_action convenience function."""
        observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)
        expected_action = np.array([0.5, -0.3], dtype=np.float32)

        with patch("psireg.rl.predictive_layer.load_predictor") as mock_load:
            mock_layer = MagicMock()
            mock_layer.predict.return_value = expected_action
            mock_load.return_value = mock_layer

            action = predict_action(
                model_path=self.model_path, observation=observation, config_path="config.pkl", deterministic=False
            )

            assert np.array_equal(action, expected_action)
            mock_load.assert_called_once_with(self.model_path, "config.pkl")
            mock_layer.predict.assert_called_once_with(observation, False)


class TestPredictiveLayerEdgeCases:
    """Test edge cases and error scenarios."""

    def test_predict_with_invalid_observation_types(self):
        """Test prediction with various invalid observation types."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.array([0.5, -0.3])
        layer = PredictiveLayer(predictor=mock_predictor)

        # Test with problematic inputs - numpy is very robust, so we focus on our validation
        # Test with empty observation (our validation should catch this)
        with pytest.raises(ValueError, match="Observation cannot be empty"):
            layer.predict(np.array([]))

    def test_predictive_layer_with_disabled_performance_tracking(self):
        """Test PredictiveLayer with performance tracking disabled."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.array([0.5, -0.3])
        mock_predictor.get_performance_metrics.return_value = {}

        layer = PredictiveLayer(predictor=mock_predictor, performance_tracking=False)
        observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

        # Make predictions
        layer.predict(observation)
        layer.predict(observation)

        # Performance count should not increase
        assert layer._prediction_count == 0

        # But metrics should still be available
        metrics = layer.get_performance_metrics()
        assert metrics["performance_tracking_enabled"] is False
        assert metrics["total_predictions"] == 0

    def test_layer_resilience_to_predictor_errors(self):
        """Test that layer handles predictor errors gracefully."""
        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = RuntimeError("Predictor failed")

        layer = PredictiveLayer(predictor=mock_predictor)
        observation = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)

        # Should propagate the error from predictor
        with pytest.raises(RuntimeError, match="Predictor failed"):
            layer.predict(observation)
