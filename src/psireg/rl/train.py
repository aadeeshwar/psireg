"""PPO training script for GridEnv using stable-baselines3.

This module provides training functionality for PPO agents on the GridEnv environment.
It includes:

- PPOTrainer class for training configuration and execution
- Training progress monitoring and logging
- Model checkpointing and evaluation
- Hyperparameter optimization support
- Multi-environment training support
- Custom callbacks for grid-specific metrics

The training process optimizes renewable energy grid control policies for:
- Frequency stability
- Economic efficiency
- Grid stability
- Asset utilization optimization
"""

import importlib.util
import os
import pickle
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from psireg.config.schema import GridConfig, RLConfig, SimulationConfig
from psireg.rl.env import GridEnv
from psireg.utils.logger import logger

# Optional dependencies
_SB3_AVAILABLE = (
    importlib.util.find_spec("stable_baselines3") is not None and importlib.util.find_spec("torch") is not None
)

if _SB3_AVAILABLE:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
else:
    warnings.warn("stable-baselines3 not available. PPOTrainer will not be functional.", stacklevel=2)


class GridMetricsCallback(BaseCallback if _SB3_AVAILABLE else object):  # type: ignore[misc]
    """Custom callback for logging grid-specific metrics during training."""

    def __init__(self, eval_env: "VecEnv", eval_freq: int = 10000, verbose: int = 0):
        """Initialize the callback.

        Args:
            eval_env: Environment for evaluation
            eval_freq: Frequency of evaluation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_count = 0

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_grid_metrics()
        return True

    def _evaluate_grid_metrics(self) -> None:
        """Evaluate grid-specific metrics."""
        if not hasattr(self.eval_env, "envs"):
            return

        env = self.eval_env.envs[0] if hasattr(self.eval_env, "envs") else self.eval_env

        if not hasattr(env, "unwrapped"):
            return

        grid_env = env.unwrapped
        if not isinstance(grid_env, GridEnv):
            return

        # Reset environment and run evaluation episode
        obs = self.eval_env.reset()
        episode_rewards = []
        episode_frequency_deviations = []
        episode_power_imbalances = []

        for _ in range(100):  # Evaluate for 100 steps
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)

            episode_rewards.append(reward[0] if isinstance(reward, np.ndarray) else reward)

            # Extract grid metrics from info
            if hasattr(info, "__len__") and len(info) > 0:
                grid_info = info[0] if isinstance(info, list) else info
                if isinstance(grid_info, dict) and "grid_state" in grid_info:
                    grid_state = grid_info["grid_state"]
                    episode_frequency_deviations.append(abs(grid_state["frequency_hz"] - 60.0))
                    episode_power_imbalances.append(abs(grid_state["power_balance_mw"]))

            if done:
                break

        # Log metrics
        if episode_rewards:
            self.logger.record("eval/mean_reward", np.mean(episode_rewards))

        if episode_frequency_deviations:
            self.logger.record("eval/mean_frequency_deviation", np.mean(episode_frequency_deviations))

        if episode_power_imbalances:
            self.logger.record("eval/mean_power_imbalance", np.mean(episode_power_imbalances))

        self.eval_count += 1


class PPOTrainer:
    """PPO trainer for GridEnv renewable energy grid optimization.

    This class provides a comprehensive training framework for PPO agents
    on the GridEnv environment. It supports:

    - Multi-environment training with parallelization
    - Custom hyperparameter configurations
    - Training progress monitoring and logging
    - Model checkpointing and evaluation
    - Grid-specific metrics tracking
    - Resume training from checkpoints

    Attributes:
        config: RL configuration parameters
        model: PPO model instance
        env: Training environment
        eval_env: Evaluation environment
        callbacks: List of training callbacks
        log_dir: Directory for training logs and checkpoints
    """

    def __init__(
        self,
        rl_config: RLConfig | None = None,
        simulation_config: SimulationConfig | None = None,
        grid_config: GridConfig | None = None,
        log_dir: str = "logs/ppo_training",
        n_envs: int = 4,
        seed: int | None = None,
    ):
        """Initialize PPO trainer.

        Args:
            rl_config: RL configuration parameters
            simulation_config: Simulation configuration
            grid_config: Grid configuration
            log_dir: Directory for logs and checkpoints
            n_envs: Number of parallel environments
            seed: Random seed for reproducibility
        """
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for PPOTrainer. Please install with: pip install stable-baselines3"
            )

        self.config = rl_config or RLConfig()
        self.simulation_config = simulation_config or SimulationConfig()
        self.grid_config = grid_config or GridConfig()
        self.log_dir = Path(log_dir)
        self.n_envs = n_envs
        self.seed = seed

        # Initialize attributes
        self.model: PPO | None = None
        self.env: VecEnv | None = None
        self.eval_env: VecEnv | None = None
        self.callbacks: list[BaseCallback] = []

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        if seed is not None:
            set_random_seed(seed)

        logger.info(f"PPOTrainer initialized with {n_envs} environments, log_dir: {log_dir}")

    def _make_env(self, rank: int = 0) -> Callable[[], GridEnv]:
        """Create environment factory function.

        Args:
            rank: Environment rank for parallel training

        Returns:
            Environment factory function
        """

        def _init() -> GridEnv:
            env = GridEnv(
                simulation_config=self.simulation_config,
                grid_config=self.grid_config,
                episode_length_hours=24,
                seed=self.seed + rank if self.seed else None,
            )

            # Add some basic assets for training
            self._setup_basic_grid(env)

            # Wrap with Monitor for logging
            monitor_path = self.log_dir / f"monitor_{rank}.csv"
            env = Monitor(env, str(monitor_path))  # type: ignore[assignment]

            return env

        return _init

    def _setup_basic_grid(self, env: GridEnv) -> None:
        """Set up basic grid configuration for training.

        Args:
            env: GridEnv environment to configure
        """
        from psireg.sim.assets.battery import Battery
        from psireg.sim.assets.load import Load
        from psireg.sim.assets.solar import SolarPanel
        from psireg.sim.assets.wind import WindTurbine
        from psireg.sim.engine import NetworkNode
        from psireg.utils.enums import AssetStatus

        # Add network nodes
        nodes = [
            NetworkNode(node_id="gen_hub", name="Generation Hub", voltage_kv=230.0),
            NetworkNode(node_id="load_center", name="Load Center", voltage_kv=138.0),
            NetworkNode(node_id="storage_hub", name="Storage Hub", voltage_kv=138.0),
        ]
        for node in nodes:
            env.grid_engine.add_node(node)

        # Add renewable generation
        solar = SolarPanel(
            asset_id="solar_farm",
            name="Solar Farm",
            node_id="gen_hub",
            capacity_mw=150.0,
            panel_efficiency=0.22,
            panel_area_m2=75000.0,
        )
        wind = WindTurbine(
            asset_id="wind_farm",
            name="Wind Farm",
            node_id="gen_hub",
            capacity_mw=100.0,
            rotor_diameter_m=80.0,
            hub_height_m=80.0,
        )

        # Add controllable assets
        battery = Battery(
            asset_id="grid_battery",
            name="Grid Battery",
            node_id="storage_hub",
            capacity_mw=50.0,
            energy_capacity_mwh=200.0,
        )
        load = Load(
            asset_id="city_load", name="City Load", node_id="load_center", capacity_mw=200.0, baseline_demand_mw=150.0
        )

        # Add assets to environment
        for asset in [solar, wind, battery, load]:
            env.add_asset(asset)
            asset.set_status(AssetStatus.ONLINE)

    def create_environments(self) -> None:
        """Create training and evaluation environments."""
        # Create training environment
        if self.n_envs > 1:
            self.env = SubprocVecEnv([self._make_env(i) for i in range(self.n_envs)])
        else:
            self.env = DummyVecEnv([self._make_env(0)])

        # Create evaluation environment
        self.eval_env = DummyVecEnv([self._make_env(self.n_envs)])

        logger.info(f"Created {self.n_envs} training environments and 1 evaluation environment")

    def create_model(self, model_path: str | None = None) -> None:
        """Create or load PPO model.

        Args:
            model_path: Path to existing model to load
        """
        if self.env is None:
            raise ValueError("Environment must be created before model")

        if model_path and os.path.exists(model_path):
            # Load existing model
            self.model = PPO.load(model_path, env=self.env)
            logger.info(f"Loaded existing model from {model_path}")
        else:
            # Create new model
            policy_kwargs = {
                "net_arch": [{"pi": [256, 256], "vf": [256, 256]}],
                "activation_fn": torch.nn.ReLU,
            }

            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                batch_size=self.config.batch_size,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(self.log_dir / "tensorboard"),
                seed=self.seed,
            )
            logger.info("Created new PPO model")

    def setup_callbacks(self) -> None:
        """Set up training callbacks."""
        self.callbacks = []

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(self.log_dir / "checkpoints"),
            name_prefix="ppo_grid_model",
        )
        self.callbacks.append(checkpoint_callback)

        # Evaluation callback
        if self.eval_env is not None:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.log_dir / "best_model"),
                log_path=str(self.log_dir / "evaluations"),
                eval_freq=5000,
                deterministic=True,
                render=False,
            )
            self.callbacks.append(eval_callback)

            # Grid metrics callback
            grid_metrics_callback = GridMetricsCallback(self.eval_env, eval_freq=5000, verbose=1)
            self.callbacks.append(grid_metrics_callback)

        logger.info(f"Set up {len(self.callbacks)} training callbacks")

    def train(self, total_timesteps: int | None = None, resume: bool = False, model_path: str | None = None) -> None:
        """Train the PPO model.

        Args:
            total_timesteps: Total training timesteps (uses config default if None)
            resume: Whether to resume from existing model
            model_path: Path to model for resuming training
        """
        if total_timesteps is None:
            # Calculate total timesteps from episodes and episode length
            steps_per_episode = 24 * 4  # 24 hours * 4 steps per hour (15min timesteps)
            total_timesteps = self.config.training_episodes * steps_per_episode

        # Create environments if not already created
        if self.env is None:
            self.create_environments()

        # Create or load model
        if resume and model_path:
            self.create_model(model_path)
        elif self.model is None:
            self.create_model()

        # Setup callbacks
        self.setup_callbacks()

        # Configure logger
        sb3_logger = configure(str(self.log_dir / "sb3_logs"), ["stdout", "csv", "tensorboard"])
        if self.model is not None:
            self.model.set_logger(sb3_logger)

        # Start training
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        start_time = datetime.now()

        try:
            if self.model is not None:
                self.model.learn(
                    total_timesteps=total_timesteps,
                    callback=self.callbacks,
                    reset_num_timesteps=not resume,
                    tb_log_name="ppo_grid_training",
                )

                # Save final model
                final_model_path = self.log_dir / "final_model.zip"
                self.model.save(str(final_model_path))

            training_time = datetime.now() - start_time
            logger.info(f"Training completed in {training_time}. Model saved to {final_model_path}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save current model
            interrupted_model_path = self.log_dir / "interrupted_model.zip"
            if self.model:
                self.model.save(str(interrupted_model_path))
                logger.info(f"Model saved to {interrupted_model_path}")

    def evaluate(
        self, model_path: str | None = None, n_eval_episodes: int = 10, deterministic: bool = True
    ) -> dict[str, float]:
        """Evaluate trained model.

        Args:
            model_path: Path to model to evaluate (uses current model if None)
            n_eval_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy

        Returns:
            Dictionary with evaluation metrics
        """
        # Load model if path provided
        if model_path:
            if self.eval_env is None:
                self.create_environments()
            model = PPO.load(model_path, env=self.eval_env)
        else:
            model = self.model  # type: ignore[assignment]

        if model is None:
            raise ValueError("No model available for evaluation")

        if self.eval_env is None:
            self.create_environments()

        # Run evaluation
        logger.info(f"Evaluating model for {n_eval_episodes} episodes")

        mean_reward, std_reward = evaluate_policy(
            model,
            self.eval_env,  # type: ignore[arg-type]
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=False,
        )

        # Collect detailed metrics
        episode_rewards = []
        episode_lengths = []
        frequency_deviations = []
        power_imbalances = []

        for _ in range(n_eval_episodes):
            if self.eval_env is not None:
                obs = self.eval_env.reset()
                episode_reward = 0
                episode_length = 0
                episode_freq_devs = []
                episode_power_imbs = []

                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=deterministic)  # type: ignore[arg-type]
                    obs, reward, done, info = self.eval_env.step(action)  # type: ignore[misc,assignment]

                    episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                    episode_length += 1

                    # Extract grid metrics
                    if hasattr(info, "__len__") and len(info) > 0:
                        grid_info = info[0] if isinstance(info, list) else info
                        if isinstance(grid_info, dict) and "grid_state" in grid_info:
                            grid_state = grid_info["grid_state"]
                            episode_freq_devs.append(abs(grid_state["frequency_hz"] - 60.0))
                            episode_power_imbs.append(abs(grid_state["power_balance_mw"]))

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                frequency_deviations.extend(episode_freq_devs)
                power_imbalances.extend(episode_power_imbs)

        # Calculate metrics
        results = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "mean_frequency_deviation": float(np.mean(frequency_deviations)) if frequency_deviations else 0.0,
            "mean_power_imbalance": float(np.mean(power_imbalances)) if power_imbalances else 0.0,
            "max_frequency_deviation": float(np.max(frequency_deviations)) if frequency_deviations else 0.0,
            "max_power_imbalance": float(np.max(power_imbalances)) if power_imbalances else 0.0,
        }

        # Save evaluation results
        eval_results_path = self.log_dir / "evaluation_results.pkl"
        with open(eval_results_path, "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Evaluation completed. Results: {results}")
        return results

    def save_config(self) -> None:
        """Save training configuration."""
        config_data = {
            "rl_config": self.config.model_dump(),
            "simulation_config": self.simulation_config.model_dump(),
            "grid_config": self.grid_config.model_dump(),
            "n_envs": self.n_envs,
            "seed": self.seed,
        }

        config_path = self.log_dir / "training_config.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(config_data, f)

        logger.info(f"Training configuration saved to {config_path}")

    def load_config(self, config_path: str) -> None:
        """Load training configuration.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, "rb") as f:
            config_data = pickle.load(f)

        self.config = RLConfig(**config_data["rl_config"])
        self.simulation_config = SimulationConfig(**config_data["simulation_config"])
        self.grid_config = GridConfig(**config_data["grid_config"])
        self.n_envs = config_data.get("n_envs", 4)
        self.seed = config_data.get("seed")

        logger.info(f"Training configuration loaded from {config_path}")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.env:
            self.env.close()
        if self.eval_env:
            self.eval_env.close()

        logger.info("PPOTrainer resources cleaned up")

    def __enter__(self) -> "PPOTrainer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Context manager exit."""
        self.cleanup()


def train_ppo_agent(
    rl_config: RLConfig | None = None,
    simulation_config: SimulationConfig | None = None,
    grid_config: GridConfig | None = None,
    log_dir: str = "logs/ppo_training",
    n_envs: int = 4,
    total_timesteps: int | None = None,
    seed: int | None = None,
    resume: bool = False,
    model_path: str | None = None,
) -> str:
    """Train a PPO agent on GridEnv.

    Convenience function for training a PPO agent with default settings.

    Args:
        rl_config: RL configuration parameters
        simulation_config: Simulation configuration
        grid_config: Grid configuration
        log_dir: Directory for logs and checkpoints
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        seed: Random seed
        resume: Whether to resume training
        model_path: Path to existing model for resuming

    Returns:
        Path to trained model
    """
    with PPOTrainer(
        rl_config=rl_config,
        simulation_config=simulation_config,
        grid_config=grid_config,
        log_dir=log_dir,
        n_envs=n_envs,
        seed=seed,
    ) as trainer:
        # Save configuration
        trainer.save_config()

        # Train model
        trainer.train(total_timesteps=total_timesteps, resume=resume, model_path=model_path)

        # Evaluate model
        trainer.evaluate()

        # Return path to final model
        return str(trainer.log_dir / "final_model.zip")


# Export classes when dependencies are available
if _SB3_AVAILABLE:
    __all__ = ["PPOTrainer", "GridMetricsCallback", "train_ppo_agent"]
else:
    __all__ = []
