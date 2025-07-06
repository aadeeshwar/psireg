#!/usr/bin/env python3
"""Demonstration of RL Environment Wrapper for PPO.

This script demonstrates the complete RL module functionality:
- GridEnv setup and configuration
- PPO training (if dependencies available)
- Model inference and prediction
- End-to-end renewable energy grid optimization

Run this script to see the RL environment in action.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path so we can import psireg
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import psireg
from psireg.config.schema import SimulationConfig, GridConfig, RLConfig
from psireg.sim.engine import GridEngine, NetworkNode
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.assets.wind import WindTurbine
from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.utils.enums import AssetStatus


def setup_demo_grid() -> GridEngine:
    """Set up a demonstration grid for RL training."""
    print("Setting up demonstration grid...")
    
    # Create configurations
    sim_config = SimulationConfig(timestep_minutes=15)
    grid_config = GridConfig(frequency_hz=60.0)
    
    # Create grid engine
    engine = GridEngine(sim_config, grid_config)
    
    # Add network nodes
    nodes = [
        NetworkNode(node_id="renewable_hub", name="Renewable Generation Hub", voltage_kv=230.0),
        NetworkNode(node_id="storage_hub", name="Energy Storage Hub", voltage_kv=138.0),
        NetworkNode(node_id="load_center", name="Load Center", voltage_kv=138.0),
    ]
    
    for node in nodes:
        engine.add_node(node)
    
    # Add renewable generation assets
    solar_farm = SolarPanel(
        asset_id="solar_farm_1",
        name="Main Solar Farm",
        node_id="renewable_hub",
        capacity_mw=200.0,
        panel_efficiency=0.22,
        panel_area_m2=100000.0
    )
    
    wind_farm = WindTurbine(
        asset_id="wind_farm_1",
        name="Main Wind Farm",
        node_id="renewable_hub",
        capacity_mw=150.0,
        rotor_diameter_m=90.0,
        hub_height_m=80.0
    )
    
    # Add storage assets (controllable by RL)
    grid_battery = Battery(
        asset_id="grid_battery_1",
        name="Grid-Scale Battery",
        node_id="storage_hub",
        capacity_mw=100.0,
        energy_capacity_mwh=400.0
    )
    
    # Add load assets (controllable by RL via demand response)
    city_load = Load(
        asset_id="city_load_1",
        name="City Load",
        node_id="load_center",
        capacity_mw=300.0,
        baseline_demand_mw=220.0
    )
    
    industrial_load = Load(
        asset_id="industrial_load_1",
        name="Industrial Load",
        node_id="load_center",
        capacity_mw=150.0,
        baseline_demand_mw=120.0
    )
    
    # Add all assets to grid
    assets = [solar_farm, wind_farm, grid_battery, city_load, industrial_load]
    for asset in assets:
        engine.add_asset(asset)
        asset.set_status(AssetStatus.ONLINE)
    
    print(f"Grid setup complete with {len(assets)} assets across {len(nodes)} nodes")
    return engine


def demonstrate_grid_environment():
    """Demonstrate GridEnv functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING GRIDENVIRONMENT WRAPPER")
    print("="*60)
    
    try:
        from psireg.rl.env import GridEnv, _GYM_AVAILABLE
        
        if not _GYM_AVAILABLE:
            print("⚠️  Gymnasium not available - showing concept demonstration")
            demonstrate_rl_concepts()
            return
        
        print("✅ Gymnasium available - creating GridEnv")
        
        # Create environment
        env = GridEnv(
            simulation_config=SimulationConfig(timestep_minutes=15),
            grid_config=GridConfig(frequency_hz=60.0),
            episode_length_hours=24  # 24-hour episodes
        )
        
        # Set up grid in environment
        demo_grid = setup_demo_grid()
        env.grid_engine = demo_grid
        
        # Update controllable assets
        env.controllable_assets.clear()
        for asset in demo_grid.assets.values():
            if env._is_controllable(asset):
                env.controllable_assets[asset.asset_id] = asset
        
        # Update spaces
        env._update_spaces()
        
        print(f"Environment created with:")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Controllable assets: {len(env.controllable_assets)}")
        
        # Run demonstration episode
        print("\nRunning demonstration episode...")
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial reward info: {info}")
        
        episode_rewards = []
        episode_actions = []
        
        for step in range(10):  # Run for 10 steps (2.5 hours)
            # Sample random action (in real training, PPO would provide this)
            action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_actions.append(action.copy())
            
            print(f"Step {step+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                break
        
        print(f"\nEpisode completed:")
        print(f"  - Total steps: {len(episode_rewards)}")
        print(f"  - Total reward: {sum(episode_rewards):.3f}")
        print(f"  - Average reward: {np.mean(episode_rewards):.3f}")
        
        # Show final grid state
        final_state = env.grid_engine.get_state()
        print(f"  - Final frequency: {final_state.frequency_hz:.2f} Hz")
        print(f"  - Final power balance: {final_state.power_balance_mw:.2f} MW")
        
        env.close()
        print("✅ GridEnv demonstration completed successfully")
        
    except ImportError as e:
        print(f"⚠️  RL dependencies not available: {e}")
        demonstrate_rl_concepts()


def demonstrate_rl_concepts():
    """Demonstrate RL concepts without requiring dependencies."""
    print("\n📋 RL CONCEPT DEMONSTRATION (No dependencies required)")
    print("-" * 50)
    
    # Set up grid
    engine = setup_demo_grid()
    
    # Simulate RL-like control loop
    print("\nSimulating RL control loop...")
    
    controllable_assets = []
    for asset in engine.assets.values():
        if isinstance(asset, (Battery, Load)):
            controllable_assets.append(asset)
    
    print(f"Found {len(controllable_assets)} controllable assets")
    
    for step in range(5):
        print(f"\nStep {step + 1}:")
        
        # Get current state (would be observation in RL)
        grid_state = engine.get_state()
        
        # Construct observation-like vector
        observation = [
            grid_state.frequency_hz / 60.0,  # Normalized frequency
            grid_state.total_generation_mw / 1000.0,  # Normalized generation
            grid_state.total_load_mw / 1000.0,  # Normalized load
            grid_state.power_balance_mw / 1000.0,  # Normalized balance
        ]
        
        # Add asset states
        for asset in controllable_assets:
            if isinstance(asset, Battery):
                observation.extend([
                    asset.current_output_mw / asset.capacity_mw,
                    asset.current_soc_percent / 100.0,
                ])
            elif isinstance(asset, Load):
                observation.extend([
                    asset.current_demand_mw / asset.capacity_mw,
                    1.0  # Load factor placeholder
                ])
        
        obs_array = np.array(observation, dtype=np.float32)
        print(f"  Observation: shape={obs_array.shape}, values={obs_array[:4]}")
        
        # Simulate PPO action prediction
        num_controllable = len(controllable_assets)
        action = np.random.uniform(-1, 1, size=num_controllable).astype(np.float32)
        print(f"  Action: {action}")
        
        # Apply actions to assets
        for i, asset in enumerate(controllable_assets):
            action_value = action[i]
            
            if isinstance(asset, Battery):
                max_charge = asset.get_max_charge_power()
                max_discharge = asset.get_max_discharge_power()
                
                if action_value >= 0:
                    power_setpoint = action_value * max_charge
                    action_desc = f"Charge at {action_value*100:.1f}%"
                else:
                    power_setpoint = action_value * max_discharge
                    action_desc = f"Discharge at {abs(action_value)*100:.1f}%"
                
                asset.set_power_setpoint(power_setpoint)
                
            elif isinstance(asset, Load):
                baseline_demand = asset.baseline_demand_mw
                dr_capability = baseline_demand * 0.2  # 20% DR capability
                dr_signal = action_value * dr_capability
                
                asset.set_demand_response_signal(dr_signal)
                
                if action_value >= 0:
                    action_desc = f"Increase demand by {action_value*100:.1f}%"
                else:
                    action_desc = f"Reduce demand by {abs(action_value)*100:.1f}%"
            
            print(f"    {asset.name}: {action_desc}")
        
        # Step simulation
        engine.step(timedelta(minutes=15))
        
        # Calculate simple reward (frequency stability + power balance)
        new_state = engine.get_state()
        frequency_penalty = abs(new_state.frequency_hz - 60.0) * 10
        balance_penalty = abs(new_state.power_balance_mw) * 0.1
        reward = -(frequency_penalty + balance_penalty)
        
        print(f"  Reward: {reward:.3f} (freq_dev: {abs(new_state.frequency_hz - 60.0):.3f}, balance: {abs(new_state.power_balance_mw):.1f})")
    
    print("\n✅ RL concept demonstration completed")


def demonstrate_ppo_training():
    """Demonstrate PPO training if dependencies are available."""
    print("\n" + "="*60)
    print("DEMONSTRATING PPO TRAINING")
    print("="*60)
    
    try:
        from psireg.rl.train import PPOTrainer, _SB3_AVAILABLE
        
        if not _SB3_AVAILABLE:
            print("⚠️  Stable-baselines3 not available - skipping training demonstration")
            return
        
        print("✅ Stable-baselines3 available - creating PPOTrainer")
        
        # Create minimal RL config for demonstration
        rl_config = RLConfig(
            learning_rate=0.001,
            gamma=0.95,
            batch_size=32,
            training_episodes=5,  # Very small for demo
        )
        
        sim_config = SimulationConfig(timestep_minutes=15)
        grid_config = GridConfig(frequency_hz=60.0)
        
        # Create trainer
        log_dir = "logs/rl_demo"
        trainer = PPOTrainer(
            rl_config=rl_config,
            simulation_config=sim_config,
            grid_config=grid_config,
            log_dir=log_dir,
            n_envs=1,  # Single environment for demo
            seed=42
        )
        
        print(f"Trainer created with config:")
        print(f"  - Learning rate: {rl_config.learning_rate}")
        print(f"  - Episodes: {rl_config.training_episodes}")
        print(f"  - Log directory: {log_dir}")
        
        # Save configuration
        trainer.save_config()
        print("✅ Configuration saved")
        
        # Note: We don't actually run training here as it would take too long
        # In a real scenario, you would call:
        # trainer.train(total_timesteps=1000)
        
        print("📝 Training would run here (skipped for demo)")
        print("    Use trainer.train(total_timesteps=10000) for actual training")
        
        # Cleanup
        trainer.cleanup()
        print("✅ PPO training demonstration completed")
        
    except ImportError as e:
        print(f"⚠️  PPO training dependencies not available: {e}")


def demonstrate_grid_predictor():
    """Demonstrate GridPredictor if dependencies are available."""
    print("\n" + "="*60)
    print("DEMONSTRATING GRID PREDICTOR")
    print("="*60)
    
    try:
        from psireg.rl.infer import GridPredictor, _SB3_AVAILABLE
        
        if not _SB3_AVAILABLE:
            print("⚠️  Stable-baselines3 not available - showing predictor concepts")
            demonstrate_predictor_concepts()
            return
        
        print("✅ Stable-baselines3 available")
        print("📝 GridPredictor requires trained model file (not available in demo)")
        print("    In real usage:")
        print("    predictor = GridPredictor('path/to/trained_model.zip')")
        print("    action, info = predictor.predict_action(observation)")
        
        demonstrate_predictor_concepts()
        
    except ImportError as e:
        print(f"⚠️  GridPredictor dependencies not available: {e}")
        demonstrate_predictor_concepts()


def demonstrate_predictor_concepts():
    """Demonstrate predictor concepts without requiring dependencies."""
    print("\n📋 PREDICTOR CONCEPT DEMONSTRATION")
    print("-" * 40)
    
    # Mock observation and action
    observation = np.array([
        1.0,    # frequency (normalized)
        0.15,   # generation (normalized)
        0.12,   # load (normalized)
        0.02,   # storage (normalized)
        0.003,  # losses (normalized)
        0.03,   # balance (normalized)
        0.5,    # battery soc
        0.8,    # load factor
        0.5,    # hour of day
        0.3     # day of week
    ], dtype=np.float32)
    
    # Mock predicted action
    action = np.array([0.3, -0.2], dtype=np.float32)  # Battery charge, load reduce
    
    print(f"Mock observation: {observation}")
    print(f"Mock action: {action}")
    
    # Interpret action
    battery_action = action[0]
    load_action = action[1]
    
    if battery_action > 0:
        battery_interpretation = f"Charge battery at {battery_action*100:.1f}% rate"
    else:
        battery_interpretation = f"Discharge battery at {abs(battery_action)*100:.1f}% rate"
    
    if load_action > 0:
        load_interpretation = f"Increase demand by {load_action*100:.1f}%"
    else:
        load_interpretation = f"Reduce demand by {abs(load_action)*100:.1f}%"
    
    print(f"\nAction interpretation:")
    print(f"  - Battery: {battery_interpretation}")
    print(f"  - Load: {load_interpretation}")
    
    # Mock performance metrics
    performance = {
        "prediction_time_ms": 2.5,
        "confidence_score": 0.85,
        "total_predictions": 1000,
    }
    
    print(f"\nMock performance metrics:")
    for key, value in performance.items():
        print(f"  - {key}: {value}")
    
    print("✅ Predictor concept demonstration completed")


def main():
    """Main demonstration function."""
    print("🚀 PSIREG RL ENVIRONMENT WRAPPER DEMONSTRATION")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"PSIREG version: {psireg.__version__}")
    print(f"RL module available: {psireg._rl_available}")
    
    # Check dependencies
    dependencies = {
        "gymnasium": False,
        "stable-baselines3": False,
        "torch": False,
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            pass
    
    print("\nDependency Status:")
    for dep, available in dependencies.items():
        status = "✅" if available else "❌"
        print(f"  {dep}: {status}")
    
    # Run demonstrations
    demonstrate_grid_environment()
    demonstrate_ppo_training()
    demonstrate_grid_predictor()
    
    print("\n" + "="*60)
    print("🎉 DEMONSTRATION COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("1. Install RL dependencies: pip install gymnasium stable-baselines3 torch")
    print("2. Create and train your own PPO agent")
    print("3. Use trained models for grid optimization")
    print("4. Integrate with real grid data for production deployment")
    print("\nFor more information, see the PSIREG documentation.")


if __name__ == "__main__":
    main() 