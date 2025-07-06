"""Predictive Layer for PPO Inference Service.

This module provides a clean, simplified API interface for PPO model inference
in renewable energy grid optimization. It wraps the comprehensive GridPredictor
functionality with a focus on ease of use and integration.

The Predictive Layer serves as the primary output interface for the PPO
Inference Service, providing:

- Simple predict(obs) interface
- Model lifecycle management
- Performance monitoring
- Error handling and graceful degradation
- Integration with existing grid systems

Usage:
    # Basic usage
    predictor = PredictiveLayer.load_model("path/to/model.zip")
    action = predictor.predict(observation)

    # Advanced usage with configuration
    predictor = PredictiveLayer.load_model(
        model_path="path/to/model.zip",
        config_path="path/to/config.pkl"
    )
    actions = predictor.predict_batch([obs1, obs2, obs3])

    # Performance monitoring
    metrics = predictor.get_performance_metrics()
"""

import importlib.util
import warnings
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from psireg.sim.engine import GridEngine  # noqa: F401 - Used in type hints
from psireg.utils.logger import logger

# Optional dependencies
_RL_AVAILABLE = (
    importlib.util.find_spec("stable_baselines3") is not None
    and importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("gymnasium") is not None
)

if _RL_AVAILABLE:
    from psireg.rl.infer import GridPredictor, load_trained_model
else:
    warnings.warn(
        "RL dependencies not available. PredictiveLayer will use fallback implementation.", UserWarning, stacklevel=2
    )


class PredictorProtocol(Protocol):
    """Protocol for predictor implementations."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given observation."""
        ...

    def predict_batch(self, observations: list[np.ndarray], deterministic: bool = True) -> list[np.ndarray]:
        """Predict actions for batch of observations."""
        ...

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        ...


class PredictiveLayer:
    """Predictive Layer for PPO Inference Service.

    This class provides a clean, simplified API for PPO model inference
    that focuses on the core prediction functionality while maintaining
    access to advanced features when needed.

    Attributes:
        predictor: Underlying predictor implementation
        model_path: Path to the loaded model
        is_ready: Whether the predictor is ready for inference
        performance_tracking: Whether to track performance metrics
    """

    def __init__(self, predictor: PredictorProtocol | None = None, performance_tracking: bool = True):
        """Initialize Predictive Layer.

        Args:
            predictor: Underlying predictor implementation
            performance_tracking: Whether to track performance metrics
        """
        self.predictor = predictor
        self.model_path: Path | None = None
        self.is_ready = predictor is not None
        self.performance_tracking = performance_tracking
        self._prediction_count = 0

        if self.is_ready:
            logger.info("PredictiveLayer initialized and ready for inference")
        else:
            logger.info("PredictiveLayer initialized in fallback mode")

    @classmethod
    def load_model(
        cls, model_path: str, config_path: str | None = None, device: str = "auto", performance_tracking: bool = True
    ) -> "PredictiveLayer":
        """Load a trained model and create PredictiveLayer.

        Args:
            model_path: Path to trained PPO model
            config_path: Path to training configuration (optional)
            device: Device for inference ("cpu", "cuda", or "auto")
            performance_tracking: Whether to track performance metrics

        Returns:
            Configured PredictiveLayer instance
        """
        if not _RL_AVAILABLE:
            logger.warning("RL dependencies not available, creating fallback predictor")
            return cls._create_fallback_predictor(model_path, performance_tracking)

        try:
            # Load using existing GridPredictor
            grid_predictor = load_trained_model(model_path=model_path, config_path=config_path, device=device)

            # Create wrapper
            wrapper = GridPredictorWrapper(grid_predictor)
            layer = cls(predictor=wrapper, performance_tracking=performance_tracking)
            layer.model_path = Path(model_path)

            logger.info(f"PredictiveLayer loaded model: {model_path}")
            return layer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return cls._create_fallback_predictor(model_path, performance_tracking)

    @classmethod
    def _create_fallback_predictor(cls, model_path: str, performance_tracking: bool = True) -> "PredictiveLayer":
        """Create fallback predictor when RL dependencies are not available.

        Args:
            model_path: Path to model (for reference only)
            performance_tracking: Whether to track performance metrics

        Returns:
            PredictiveLayer with fallback predictor
        """
        fallback = FallbackPredictor(model_path)
        layer = cls(predictor=fallback, performance_tracking=performance_tracking)
        layer.model_path = Path(model_path)
        return layer

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given observation.

        This is the primary interface method that provides simple
        predict(obs) functionality as requested.

        Args:
            observation: Grid state observation vector
            deterministic: Whether to use deterministic policy

        Returns:
            Action vector for grid asset control

        Raises:
            RuntimeError: If predictor is not ready
        """
        if not self.is_ready or self.predictor is None:
            raise RuntimeError("Predictor not ready. Load a model first.")

        # Validate observation
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)

        if observation.size == 0:
            raise ValueError("Observation cannot be empty")

        # Make prediction
        action = self.predictor.predict(observation, deterministic=deterministic)

        # Track performance if enabled
        if self.performance_tracking:
            self._prediction_count += 1

        return action

    def predict_batch(self, observations: list[np.ndarray], deterministic: bool = True) -> list[np.ndarray]:
        """Predict actions for batch of observations.

        Args:
            observations: List of observation vectors
            deterministic: Whether to use deterministic policy

        Returns:
            List of action vectors

        Raises:
            RuntimeError: If predictor is not ready
        """
        if not self.is_ready or self.predictor is None:
            raise RuntimeError("Predictor not ready. Load a model first.")

        if not observations:
            return []

        # Make batch prediction
        actions = self.predictor.predict_batch(observations, deterministic=deterministic)

        # Track performance if enabled
        if self.performance_tracking:
            self._prediction_count += len(observations)

        return actions

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the predictor.

        Returns:
            Dictionary with performance metrics
        """
        if not self.is_ready or self.predictor is None:
            return {"error": "Predictor not ready"}

        metrics = self.predictor.get_performance_metrics()

        # Add layer-specific metrics
        metrics.update(
            {
                "total_predictions": self._prediction_count,
                "model_path": str(self.model_path) if self.model_path else None,
                "predictor_ready": self.is_ready,
                "performance_tracking_enabled": self.performance_tracking,
            }
        )

        return metrics

    def is_model_loaded(self) -> bool:
        """Check if a model is loaded and ready for inference.

        Returns:
            True if model is loaded and ready
        """
        return self.is_ready and self.predictor is not None

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_path": str(self.model_path) if self.model_path else None,
            "is_ready": self.is_ready,
            "predictor_type": type(self.predictor).__name__ if self.predictor else None,
            "rl_dependencies_available": _RL_AVAILABLE,
            "total_predictions": self._prediction_count,
        }


class GridPredictorWrapper:
    """Wrapper around GridPredictor to implement PredictorProtocol."""

    def __init__(self, grid_predictor: "GridPredictor"):
        """Initialize wrapper.

        Args:
            grid_predictor: GridPredictor instance to wrap
        """
        self.grid_predictor = grid_predictor

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given observation."""
        action, _ = self.grid_predictor.predict_action(observation=observation, deterministic=deterministic)
        return action

    def predict_batch(self, observations: list[np.ndarray], deterministic: bool = True) -> list[np.ndarray]:
        """Predict actions for batch of observations."""
        actions, _ = self.grid_predictor.predict_batch(observations=observations, deterministic=deterministic)
        return actions

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        return self.grid_predictor.performance_metrics.copy()


class FallbackPredictor:
    """Fallback predictor for when RL dependencies are not available."""

    def __init__(self, model_path: str):
        """Initialize fallback predictor.

        Args:
            model_path: Path to model (for reference only)
        """
        self.model_path = model_path
        self.prediction_count = 0

        logger.warning(f"Using fallback predictor for {model_path}. " "Install RL dependencies for full functionality.")

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action using fallback strategy."""
        self.prediction_count += 1

        # Simple heuristic: return small random actions
        # In practice, this could be replaced with rule-based control
        action_size = max(2, len(observation) // 5)  # Estimate action size
        action = np.random.uniform(-0.1, 0.1, size=action_size).astype(np.float32)

        logger.debug(f"Fallback prediction {self.prediction_count}: {action}")
        return action

    def predict_batch(self, observations: list[np.ndarray], deterministic: bool = True) -> list[np.ndarray]:
        """Predict actions for batch using fallback strategy."""
        return [self.predict(obs, deterministic) for obs in observations]

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get fallback performance metrics."""
        return {
            "type": "fallback",
            "model_path": self.model_path,
            "prediction_count": self.prediction_count,
            "warning": "Using fallback predictor - install RL dependencies for full functionality",
        }


# Convenience functions
def load_predictor(model_path: str, config_path: str | None = None, device: str = "auto") -> PredictiveLayer:
    """Load a trained model and create PredictiveLayer.

    Convenience function for quick model loading.

    Args:
        model_path: Path to trained PPO model
        config_path: Path to training configuration (optional)
        device: Device for inference

    Returns:
        Configured PredictiveLayer instance
    """
    return PredictiveLayer.load_model(model_path=model_path, config_path=config_path, device=device)


def predict_action(
    model_path: str, observation: np.ndarray, config_path: str | None = None, deterministic: bool = True
) -> np.ndarray:
    """Quick prediction with model loading.

    Convenience function for one-off predictions.

    Args:
        model_path: Path to trained PPO model
        observation: Grid state observation
        config_path: Path to training configuration (optional)
        deterministic: Whether to use deterministic policy

    Returns:
        Action vector
    """
    predictor = load_predictor(model_path, config_path)
    return predictor.predict(observation, deterministic)


__all__ = [
    "PredictiveLayer",
    "PredictorProtocol",
    "load_predictor",
    "predict_action",
]
