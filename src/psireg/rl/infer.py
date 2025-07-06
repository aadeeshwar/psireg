"""Inference service for trained PPO models on GridEnv.

This module provides the GridPredictor service for using trained reinforcement learning
models to control grid assets in real-time or simulation scenarios. It includes:

- GridPredictor class for model loading and inference
- Real-time prediction capabilities
- Batch prediction for multiple scenarios
- Model performance monitoring
- Integration with GridEngine for live control
- Policy interpretation and explainability

The predictor service enables:
- Real-time grid control using trained RL policies
- Scenario analysis and "what-if" simulations
- Policy performance evaluation
- Integration with existing grid management systems
"""

import importlib.util
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from psireg.config.schema import GridConfig, RLConfig, SimulationConfig
from psireg.rl.env import GridEnv
from psireg.sim.assets.base import Asset
from psireg.sim.engine import GridEngine, GridState
from psireg.utils.logger import logger

# Optional dependencies
_SB3_AVAILABLE = (
    importlib.util.find_spec("stable_baselines3") is not None and importlib.util.find_spec("torch") is not None
)

if _SB3_AVAILABLE:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
else:
    warnings.warn("stable-baselines3 not available. GridPredictor will not be functional.", stacklevel=2)


class GridPredictor:
    """Predictor service for trained PPO models on GridEnv.

    This class provides a service for using trained reinforcement learning models
    to control grid assets and predict optimal actions. It supports:

    - Loading trained PPO models
    - Real-time action prediction
    - Batch prediction for scenario analysis
    - Model performance monitoring
    - Integration with GridEngine for live control
    - Policy interpretation and analysis

    Attributes:
        model: Loaded PPO model
        model_path: Path to the loaded model
        config: RL configuration used for training
        grid_env: GridEnv instance for predictions
        prediction_history: History of predictions and outcomes
        performance_metrics: Performance tracking metrics
    """

    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        device: str = "auto",
    ):
        """Initialize GridPredictor.

        Args:
            model_path: Path to trained PPO model
            config_path: Path to training configuration (optional)
            device: Device for model inference ("cpu", "cuda", or "auto")
        """
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for GridPredictor. Please install with: pip install stable-baselines3"
            )

        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = device

        # Initialize attributes
        self.model: PPO | None = None
        self.config: RLConfig | None = None
        self.simulation_config: SimulationConfig | None = None
        self.grid_config: GridConfig | None = None
        self.grid_env: GridEnv | None = None

        # Performance tracking
        self.prediction_history: list[dict[str, Any]] = []
        self.performance_metrics: dict[str, float] = {}

        # Load model and configuration
        self._load_model()
        self._load_config()

        logger.info(f"GridPredictor initialized with model: {model_path}")

    def _load_model(self) -> None:
        """Load trained PPO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            # Create temporary environment for model loading
            temp_env = DummyVecEnv([lambda: GridEnv()])

            # Load model
            self.model = PPO.load(str(self.model_path), env=temp_env, device=self.device)

            # Close temporary environment
            temp_env.close()

            logger.info(f"Successfully loaded PPO model from {self.model_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def _load_config(self) -> None:
        """Load training configuration if available."""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "rb") as f:
                    config_data = pickle.load(f)

                self.config = RLConfig(**config_data.get("rl_config", {}))
                self.simulation_config = SimulationConfig(**config_data.get("simulation_config", {}))
                self.grid_config = GridConfig(**config_data.get("grid_config", {}))

                logger.info(f"Loaded training configuration from {self.config_path}")

            except Exception as e:
                logger.warning(f"Failed to load configuration: {e}")
                self._use_default_config()
        else:
            logger.info("No configuration file provided, using defaults")
            self._use_default_config()

    def _use_default_config(self) -> None:
        """Use default configuration."""
        self.config = RLConfig()
        self.simulation_config = SimulationConfig()
        self.grid_config = GridConfig()

    def setup_environment(
        self,
        grid_engine: GridEngine | None = None,
        simulation_config: SimulationConfig | None = None,
        grid_config: GridConfig | None = None,
    ) -> None:
        """Set up GridEnv for predictions.

        Args:
            grid_engine: Existing GridEngine instance (creates new if None)
            simulation_config: Simulation configuration (uses loaded config if None)
            grid_config: Grid configuration (uses loaded config if None)
        """
        sim_config = simulation_config or self.simulation_config
        grd_config = grid_config or self.grid_config

        if grid_engine:
            # Use existing GridEngine
            self.grid_env = GridEnv(simulation_config=sim_config, grid_config=grd_config)
            # Replace the engine with the provided one
            self.grid_env.grid_engine = grid_engine

            # Update controllable assets
            self.grid_env.controllable_assets.clear()
            for asset in grid_engine.assets.values():
                if self.grid_env._is_controllable(asset):
                    self.grid_env.controllable_assets[asset.asset_id] = asset

            # Update spaces
            self.grid_env._update_spaces()

        else:
            # Create new environment
            self.grid_env = GridEnv(simulation_config=sim_config, grid_config=grd_config)

        logger.info("GridEnv environment set up for predictions")

    def predict_action(
        self,
        observation: np.ndarray | None = None,
        grid_state: GridState | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict optimal action for current grid state.

        Args:
            observation: Direct observation vector (if None, will be computed from grid_state)
            grid_state: Current grid state (required if observation is None)
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, prediction_info)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Get observation
        if observation is None:
            if self.grid_env is None:
                raise RuntimeError("GridEnv not set up. Call setup_environment() first")
            observation = self.grid_env._get_observation()

        # Ensure observation is in correct format
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)

        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)

        # Predict action
        start_time = datetime.now()
        action, _ = self.model.predict(observation, deterministic=deterministic)
        prediction_time = (datetime.now() - start_time).total_seconds()

        # Prepare prediction info
        prediction_info = {
            "timestamp": datetime.now().isoformat(),
            "deterministic": deterministic,
            "prediction_time_s": prediction_time,
            "observation_shape": observation.shape,
            "action_shape": action.shape,
        }

        # Store in history
        self.prediction_history.append(
            {
                "observation": observation.copy(),
                "action": action.copy(),
                "info": prediction_info.copy(),
            }
        )

        # Keep history limited to last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]

        return action.flatten(), prediction_info

    def predict_batch(
        self,
        observations: list[np.ndarray],
        deterministic: bool = True,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        """Predict actions for batch of observations.

        Args:
            observations: List of observation vectors
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (actions_list, prediction_info_list)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        actions = []
        prediction_infos = []

        start_time = datetime.now()

        for obs in observations:
            action, info = self.predict_action(obs, deterministic=deterministic)
            actions.append(action)
            prediction_infos.append(info)

        total_time = (datetime.now() - start_time).total_seconds()

        # Update performance metrics
        self.performance_metrics.update(
            {
                "batch_size": len(observations),
                "total_batch_time_s": total_time,
                "avg_prediction_time_s": total_time / len(observations),
                "predictions_per_second": len(observations) / total_time if total_time > 0 else 0,
            }
        )

        return actions, prediction_infos

    def evaluate_scenario(
        self,
        grid_engine: GridEngine,
        duration_hours: int = 24,
        timestep_minutes: int = 15,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """Evaluate policy performance on a specific scenario.

        Args:
            grid_engine: GridEngine with configured scenario
            duration_hours: Simulation duration in hours
            timestep_minutes: Timestep in minutes
            deterministic: Whether to use deterministic policy

        Returns:
            Dictionary with evaluation results
        """
        # Set up environment with provided grid engine
        self.setup_environment(grid_engine=grid_engine)

        if self.grid_env is None:
            raise RuntimeError("Failed to set up environment")

        # Reset environment
        observation, _ = self.grid_env.reset()

        # Run simulation
        total_steps = (duration_hours * 60) // timestep_minutes
        episode_data: dict[str, list] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "grid_states": [],
            "asset_states": [],
        }

        for _ in range(total_steps):
            # Predict action
            action, _ = self.predict_action(observation, deterministic=deterministic)

            # Take step
            next_observation, reward, terminated, truncated, info = self.grid_env.step(action)

            # Store data
            episode_data["observations"].append(observation.copy())
            episode_data["actions"].append(action.copy())
            episode_data["rewards"].append(reward)
            episode_data["grid_states"].append(info.get("grid_state", {}))
            episode_data["asset_states"].append(self._get_asset_states())

            observation = next_observation

            if terminated or truncated:
                break

        # Calculate evaluation metrics
        results = self._calculate_scenario_metrics(episode_data)

        return results

    def _get_asset_states(self) -> dict[str, dict[str, Any]]:
        """Get current state of all assets.

        Returns:
            Dictionary mapping asset IDs to their states
        """
        if not self.grid_env or not self.grid_env.grid_engine:
            return {}

        asset_states = {}
        for asset_id, asset in self.grid_env.grid_engine.assets.items():
            asset_states[asset_id] = asset.get_state()

        return asset_states

    def _calculate_scenario_metrics(self, episode_data: dict[str, list]) -> dict[str, Any]:
        """Calculate metrics from scenario evaluation.

        Args:
            episode_data: Data collected during scenario evaluation

        Returns:
            Dictionary with calculated metrics
        """
        rewards = episode_data["rewards"]
        grid_states = episode_data["grid_states"]

        # Basic metrics
        metrics = {
            "total_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "episode_length": len(rewards),
        }

        # Grid stability metrics
        if grid_states:
            frequency_deviations = [
                abs(state.get("frequency_hz", 60.0) - 60.0) for state in grid_states if "frequency_hz" in state
            ]
            power_imbalances = [
                abs(state.get("power_balance_mw", 0.0)) for state in grid_states if "power_balance_mw" in state
            ]

            if frequency_deviations:
                metrics.update(
                    {
                        "mean_frequency_deviation": float(np.mean(frequency_deviations)),
                        "max_frequency_deviation": float(np.max(frequency_deviations)),
                        "frequency_stability_score": float(1.0 / (1.0 + np.mean(frequency_deviations))),
                    }
                )

            if power_imbalances:
                metrics.update(
                    {
                        "mean_power_imbalance": float(np.mean(power_imbalances)),
                        "max_power_imbalance": float(np.max(power_imbalances)),
                        "power_balance_score": float(1.0 / (1.0 + np.mean(power_imbalances))),
                    }
                )

        return metrics

    def explain_action(
        self,
        observation: np.ndarray,
        action: np.ndarray,
    ) -> dict[str, Any]:
        """Provide explanation for predicted action.

        Args:
            observation: Input observation
            action: Predicted action

        Returns:
            Dictionary with action explanation
        """
        explanation = {
            "action_interpretation": self._interpret_action(action),
            "observation_summary": self._summarize_observation(observation),
            "action_confidence": self._estimate_action_confidence(observation, action),
        }

        return explanation

    def _interpret_action(self, action: np.ndarray) -> dict[str, Any]:
        """Interpret action values in grid context.

        Args:
            action: Action vector

        Returns:
            Dictionary with action interpretation
        """
        if not self.grid_env:
            return {"error": "GridEnv not set up"}

        interpretation = {}
        asset_ids = list(self.grid_env.controllable_assets.keys())

        for i, asset_id in enumerate(asset_ids):
            if i < len(action):
                asset = self.grid_env.controllable_assets[asset_id]
                action_value = action[i]

                interpretation[asset_id] = {
                    "action_value": float(action_value),
                    "asset_type": asset.asset_type.value,
                    "interpretation": self._interpret_asset_action(asset, action_value),
                }

        return interpretation

    def _interpret_asset_action(self, asset: Asset, action_value: float) -> str:
        """Interpret action value for specific asset.

        Args:
            asset: Asset to interpret action for
            action_value: Action value in [-1, 1]

        Returns:
            Human-readable interpretation
        """
        from psireg.sim.assets.battery import Battery
        from psireg.sim.assets.load import Load

        if isinstance(asset, Battery):
            if action_value > 0.5:
                return f"Charge at {action_value*100:.1f}% of max rate"
            elif action_value < -0.5:
                return f"Discharge at {abs(action_value)*100:.1f}% of max rate"
            else:
                return "Maintain current state (idle)"

        elif isinstance(asset, Load):
            if action_value > 0.1:
                return f"Increase demand by {action_value*100:.1f}%"
            elif action_value < -0.1:
                return f"Reduce demand by {abs(action_value)*100:.1f}%"
            else:
                return "Maintain baseline demand"

        else:
            return f"Control signal: {action_value:.3f}"

    def _summarize_observation(self, observation: np.ndarray) -> dict[str, Any]:
        """Summarize observation in human-readable format.

        Args:
            observation: Observation vector

        Returns:
            Dictionary with observation summary
        """
        if len(observation) < 6:
            return {"error": "Observation too short"}

        # Basic grid state (first 6 elements)
        summary = {
            "frequency_hz": float(observation[0] * 60.0),  # Denormalize
            "generation_mw": float(observation[1] * 1000.0),  # Denormalize
            "load_mw": float(observation[2] * 1000.0),  # Denormalize
            "storage_mw": float(observation[3] * 1000.0),  # Denormalize
            "losses_mw": float(observation[4] * 1000.0),  # Denormalize
            "balance_mw": float(observation[5] * 1000.0),  # Denormalize
        }

        # Add interpretation
        summary["grid_status"] = "stable" if abs(summary["balance_mw"]) < 10 else "imbalanced"  # type: ignore[assignment]
        summary["frequency_status"] = "normal" if abs(summary["frequency_hz"] - 60.0) < 0.5 else "abnormal"  # type: ignore[assignment]

        return summary

    def _estimate_action_confidence(self, observation: np.ndarray, action: np.ndarray) -> float:
        """Estimate confidence in predicted action.

        Args:
            observation: Input observation
            action: Predicted action

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Simple heuristic based on action magnitude
        # In practice, this could use model internals or ensembles
        action_magnitude = np.linalg.norm(action)
        confidence = min(1.0, float(action_magnitude + 0.5))  # Basic heuristic

        return float(confidence)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of predictor performance.

        Returns:
            Dictionary with performance metrics
        """
        summary = {
            "model_path": str(self.model_path),
            "total_predictions": len(self.prediction_history),
            "performance_metrics": self.performance_metrics.copy(),
        }

        if self.prediction_history:
            prediction_times = [
                pred["info"]["prediction_time_s"]
                for pred in self.prediction_history
                if "prediction_time_s" in pred["info"]
            ]

            if prediction_times:
                summary["prediction_time_stats"] = {
                    "mean_s": float(np.mean(prediction_times)),
                    "std_s": float(np.std(prediction_times)),
                    "min_s": float(np.min(prediction_times)),
                    "max_s": float(np.max(prediction_times)),
                }

        return summary

    def save_prediction_history(self, filepath: str) -> None:
        """Save prediction history to file.

        Args:
            filepath: Path to save history
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.prediction_history, f)

        logger.info(f"Prediction history saved to {filepath}")

    def load_prediction_history(self, filepath: str) -> None:
        """Load prediction history from file.

        Args:
            filepath: Path to load history from
        """
        with open(filepath, "rb") as f:
            self.prediction_history = pickle.load(f)

        logger.info(f"Prediction history loaded from {filepath}")

    def clear_history(self) -> None:
        """Clear prediction history and reset metrics."""
        self.prediction_history.clear()
        self.performance_metrics.clear()

        logger.info("Prediction history and metrics cleared")


def load_trained_model(
    model_path: str,
    config_path: str | None = None,
    device: str = "auto",
) -> GridPredictor:
    """Load a trained PPO model for inference.

    Convenience function for loading trained models.

    Args:
        model_path: Path to trained model
        config_path: Path to training configuration
        device: Device for inference

    Returns:
        Configured GridPredictor instance
    """
    return GridPredictor(
        model_path=model_path,
        config_path=config_path,
        device=device,
    )


def predict_grid_actions(
    model_path: str,
    grid_engine: GridEngine,
    duration_hours: int = 1,
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    """Predict grid actions for a given scenario.

    Convenience function for quick predictions.

    Args:
        model_path: Path to trained model
        grid_engine: GridEngine with scenario setup
        duration_hours: Duration to predict for
        config_path: Path to training configuration

    Returns:
        List of prediction results
    """
    predictor = GridPredictor(model_path, config_path)
    results = predictor.evaluate_scenario(
        grid_engine=grid_engine,
        duration_hours=duration_hours,
    )

    return [results]


# Export classes when dependencies are available
if _SB3_AVAILABLE:
    __all__ = ["GridPredictor", "load_trained_model", "predict_grid_actions"]
else:
    __all__ = []
