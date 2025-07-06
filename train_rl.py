#!/usr/bin/env python3
"""Standalone PPO training script for PSIREG renewable energy grid optimization.

This script provides a command-line interface for training PPO agents on GridEnv
using the existing PPOTrainer and GridPredictor components. It serves as the
primary entry point for training renewable energy grid control policies.

Usage:
    python train_rl.py [options]
    
Examples:
    # Basic training with default parameters
    python train_rl.py
    
    # Custom training configuration
    python train_rl.py --episodes 2000 --learning-rate 0.0001 --n-envs 8
    
    # Resume training from existing model
    python train_rl.py --resume --model-path models/checkpoint.zip
    
    # Use configuration file
    python train_rl.py --config-file config.yaml
    
Primary Output:
    - Trained predictor model ready for inference
    - Training logs and metrics
    - Model checkpoints and evaluation results
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from psireg.config.schema import RLConfig, SimulationConfig, GridConfig
from psireg.utils.logger import logger

# Import RL components with dependency checking
try:
    from psireg.rl.train import PPOTrainer
    from psireg.rl.infer import GridPredictor
    _RL_AVAILABLE = True
except ImportError as e:
    logger.error(f"RL dependencies not available: {e}")
    _RL_AVAILABLE = False


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Train PPO agent for renewable energy grid optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Basic training
  %(prog)s --episodes 2000 --n-envs 8        # Custom parameters
  %(prog)s --resume --model-path model.zip   # Resume training
  %(prog)s --config-file config.yaml         # Use config file
        """
    )
    
    # Training configuration
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument(
        '--episodes', type=int, default=1000,
        help='Number of training episodes (default: 1000)'
    )
    training_group.add_argument(
        '--timesteps', type=int, default=None,
        help='Total training timesteps (overrides episodes if set)'
    )
    training_group.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Learning rate for PPO (default: 0.001)'
    )
    training_group.add_argument(
        '--gamma', type=float, default=0.95,
        help='Discount factor (default: 0.95)'
    )
    training_group.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for training (default: 32)'
    )
    
    # Environment configuration
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument(
        '--n-envs', type=int, default=4,
        help='Number of parallel environments (default: 4)'
    )
    env_group.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    
    # Input/Output configuration
    io_group = parser.add_argument_group('Input/Output Configuration')
    io_group.add_argument(
        '--log-dir', type=str, default='logs/ppo_training',
        help='Directory for training logs (default: logs/ppo_training)'
    )
    io_group.add_argument(
        '--config-file', type=str, default=None,
        help='Configuration file (YAML format)'
    )
    io_group.add_argument(
        '--model-path', type=str, default=None,
        help='Path to existing model for resuming training'
    )
    io_group.add_argument(
        '--output-path', type=str, default=None,
        help='Path to save trained model (default: log_dir/final_model.zip)'
    )
    
    # Training control
    control_group = parser.add_argument_group('Training Control')
    control_group.add_argument(
        '--resume', action='store_true',
        help='Resume training from existing model'
    )
    control_group.add_argument(
        '--no-eval', action='store_true',
        help='Skip evaluation after training'
    )
    control_group.add_argument(
        '--eval-episodes', type=int, default=10,
        help='Number of evaluation episodes (default: 10)'
    )
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    advanced_group.add_argument(
        '--dry-run', action='store_true',
        help='Show configuration without training'
    )
    
    return parser


def load_config_from_file(config_file: str) -> Tuple[RLConfig, SimulationConfig, GridConfig]:
    """Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Tuple of (rl_config, sim_config, grid_config)
    """
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    rl_config = RLConfig(**config_data.get('rl', {}))
    sim_config = SimulationConfig(**config_data.get('simulation', {}))
    grid_config = GridConfig(**config_data.get('grid', {}))
    
    return rl_config, sim_config, grid_config


def create_configs_from_args(args: argparse.Namespace) -> Tuple[RLConfig, SimulationConfig, GridConfig]:
    """Create configuration objects from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of (rl_config, sim_config, grid_config)
    """
    if args.config_file:
        # Load from file and override with command-line args
        rl_config, sim_config, grid_config = load_config_from_file(args.config_file)
        
        # Override with command-line arguments
        if hasattr(args, 'learning_rate') and args.learning_rate != 0.001:
            rl_config.learning_rate = args.learning_rate
        if hasattr(args, 'gamma') and args.gamma != 0.95:
            rl_config.gamma = args.gamma
        if hasattr(args, 'episodes') and args.episodes != 1000:
            rl_config.training_episodes = args.episodes
        if hasattr(args, 'batch_size') and args.batch_size != 32:
            rl_config.batch_size = args.batch_size
            
    else:
        # Create from command-line arguments
        rl_config = RLConfig(
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            training_episodes=args.episodes,
            batch_size=getattr(args, 'batch_size', 32)
        )
        sim_config = SimulationConfig()
        grid_config = GridConfig()
    
    return rl_config, sim_config, grid_config


def validate_config(config: RLConfig) -> None:
    """Validate configuration parameters.
    
    Args:
        config: RL configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if config.gamma < 0 or config.gamma > 1:
        raise ValueError("Gamma must be between 0 and 1")
    if config.training_episodes <= 0:
        raise ValueError("Training episodes must be positive")
    if config.batch_size <= 0:
        raise ValueError("Batch size must be positive")


def setup_trainer(
    rl_config: RLConfig,
    sim_config: SimulationConfig,
    grid_config: GridConfig,
    log_dir: str,
    n_envs: int,
    seed: Optional[int] = None
) -> PPOTrainer:
    """Set up PPOTrainer with configurations.
    
    Args:
        rl_config: RL configuration
        sim_config: Simulation configuration
        grid_config: Grid configuration
        log_dir: Log directory
        n_envs: Number of environments
        seed: Random seed
        
    Returns:
        Configured PPOTrainer instance
    """
    trainer = PPOTrainer(
        rl_config=rl_config,
        simulation_config=sim_config,
        grid_config=grid_config,
        log_dir=log_dir,
        n_envs=n_envs,
        seed=seed
    )
    
    return trainer


def setup_predictor(model_path: str, config_path: str) -> GridPredictor:
    """Set up GridPredictor with trained model.
    
    Args:
        model_path: Path to trained model
        config_path: Path to training configuration
        
    Returns:
        Configured GridPredictor instance
    """
    predictor = GridPredictor(
        model_path=model_path,
        config_path=config_path
    )
    
    return predictor


def validate_trained_model(model_path: str) -> None:
    """Validate that trained model file exists.
    
    Args:
        model_path: Path to model file
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")


def create_predictor_model(model_path: str, config_path: str) -> Dict[str, Any]:
    """Create predictor model information.
    
    Args:
        model_path: Path to trained model
        config_path: Path to training configuration
        
    Returns:
        Dictionary with predictor model information
    """
    validate_trained_model(model_path)
    
    predictor_info = {
        "model_path": model_path,
        "config_path": config_path,
        "creation_time": datetime.now().isoformat(),
        "predictor_ready": True
    }
    
    return predictor_info


def generate_training_summary(training_results: Dict[str, Any]) -> str:
    """Generate human-readable training summary.
    
    Args:
        training_results: Training results dictionary
        
    Returns:
        Formatted training summary string
    """
    eval_results = training_results.get("evaluation_results", {})
    
    # Handle total_timesteps formatting
    total_timesteps = training_results.get('total_timesteps', 0)
    timesteps_str = f"{total_timesteps:,}" if isinstance(total_timesteps, (int, float)) else str(total_timesteps)
    
    summary = f"""
Training completed successfully!

Results:
  Model saved to: {training_results.get('model_path', 'Unknown')}
  Total timesteps: {timesteps_str}
  Training time: {training_results.get('training_time', 0) / 3600:.1f} hours

Evaluation Metrics:
  Mean reward: {eval_results.get('mean_reward', 0):.1f}
  Std reward: {eval_results.get('std_reward', 0):.1f}
  Mean episode length: {eval_results.get('mean_episode_length', 0):.1f}
  Mean frequency deviation: {eval_results.get('mean_frequency_deviation', 0):.3f} Hz
  Mean power imbalance: {eval_results.get('mean_power_imbalance', 0):.2f} MW

Primary Output:
  ✅ Predictor model ready for inference
  ✅ Training logs and metrics saved
  ✅ Model checkpoints available
"""
    
    return summary.strip()


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the training process.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Training results dictionary
    """
    # Validate dependencies
    if not _RL_AVAILABLE:
        raise ImportError("RL dependencies not available. Please install: pip install gymnasium stable-baselines3 torch")
    
    # Create configurations
    rl_config, sim_config, grid_config = create_configs_from_args(args)
    
    # Validate configuration
    validate_config(rl_config)
    
    # Check for resume training
    if args.resume:
        if not args.model_path:
            raise ValueError("Model path must be specified for resume training")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Set up trainer
    trainer = setup_trainer(
        rl_config=rl_config,
        sim_config=sim_config,
        grid_config=grid_config,
        log_dir=args.log_dir,
        n_envs=args.n_envs,
        seed=args.seed
    )
    
    # Calculate total timesteps
    total_timesteps = args.timesteps
    if total_timesteps is None:
        # Calculate from episodes (24 hours * 4 steps per hour)
        total_timesteps = args.episodes * 96
    
    # Start training
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting PPO training with {total_timesteps:,} timesteps")
        
        # Train the model
        trainer.train(
            total_timesteps=total_timesteps,
            resume=args.resume,
            model_path=args.model_path
        )
        
        # Get final model path
        final_model_path = args.output_path or str(Path(args.log_dir) / "final_model.zip")
        
        # Evaluate model (unless disabled)
        evaluation_results = {}
        if not getattr(args, 'no_eval', False):
            logger.info("Evaluating trained model...")
            evaluation_results = trainer.evaluate(n_eval_episodes=getattr(args, 'eval_episodes', 10))
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare results
        results = {
            "model_path": final_model_path,
            "evaluation_results": evaluation_results,
            "training_time": training_time,
            "total_timesteps": total_timesteps,
            "config_path": str(Path(args.log_dir) / "training_config.pkl")
        }
        
        # Create predictor model info
        predictor_info = create_predictor_model(
            final_model_path, 
            results["config_path"]
        )
        results["predictor_info"] = predictor_info
        
        logger.info("Training completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Clean up trainer resources
        trainer.cleanup()


def main():
    """Main entry point for the training script."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Show configuration and exit if dry run
    if args.dry_run:
        print("Configuration:")
        print(f"  Episodes: {args.episodes}")
        print(f"  Timesteps: {args.timesteps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Gamma: {args.gamma}")
        print(f"  N environments: {args.n_envs}")
        print(f"  Log directory: {args.log_dir}")
        print(f"  Seed: {args.seed}")
        print(f"  Resume: {args.resume}")
        print(f"  Model path: {args.model_path}")
        print(f"  Config file: {args.config_file}")
        return
    
    try:
        # Run training
        results = run_training(args)
        
        # Generate and display summary
        summary = generate_training_summary(results)
        print("\n" + "="*60)
        print(summary)
        print("="*60)
        
        # Display predictor information
        predictor_info = results.get("predictor_info", {})
        if predictor_info.get("predictor_ready"):
            print("\n🎯 PRIMARY OUTPUT: Predictor model ready!")
            print(f"   Model: {predictor_info['model_path']}")
            print(f"   Config: {predictor_info['config_path']}")
            print(f"   Created: {predictor_info['creation_time']}")
            print("\n💡 Usage:")
            print("   from psireg.rl.infer import GridPredictor")
            print(f"   predictor = GridPredictor('{predictor_info['model_path']}')")
            print("   action, info = predictor.predict_action(observation)")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 