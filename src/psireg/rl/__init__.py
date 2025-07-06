"""Reinforcement Learning module for PSIREG renewable energy grid system.

This module provides:
- GridEnv: Gym-compatible environment wrapper around GridEngine
- PPO training functionality for renewable energy grid optimization
- Predictor service for inference with trained models

The RL module enables:
- Intelligent grid control through reinforcement learning
- Asset dispatch optimization
- Demand response coordination
- Grid stability and economic optimization
"""

# Optional dependencies - check availability without importing
import importlib.util

_RL_AVAILABLE = (
    importlib.util.find_spec("gymnasium") is not None
    and importlib.util.find_spec("stable_baselines3") is not None
    and importlib.util.find_spec("torch") is not None
)

# Import main classes when dependencies are available
if _RL_AVAILABLE:
    from .env import GridEnv
    from .infer import GridPredictor
    from .predictive_layer import PredictiveLayer
    from .train import PPOTrainer

    __all__ = ["GridEnv", "PPOTrainer", "GridPredictor", "PredictiveLayer"]
else:
    # Always import PredictiveLayer as it has fallback functionality
    from .predictive_layer import PredictiveLayer

    __all__ = ["PredictiveLayer"]

__version__ = "0.1.0"
