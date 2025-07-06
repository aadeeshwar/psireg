#!/usr/bin/env python3
"""Demo script for PredictiveLayer - PPO Inference Service.

This script demonstrates the PredictiveLayer functionality for renewable
energy grid optimization, including:

1. Loading trained models with graceful fallback
2. Making single predictions with predict(obs) 
3. Batch predictions for multiple scenarios
4. Performance monitoring and metrics
5. Integration with grid simulation

The PredictiveLayer serves as the primary output interface for the PPO
Inference Service, providing clean API access to trained reinforcement
learning models for real-time grid control.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psireg.config.schema import GridConfig, SimulationConfig
from psireg.rl.predictive_layer import PredictiveLayer, load_predictor, predict_action
from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.engine import GridEngine, NetworkNode
from psireg.utils.enums import AssetStatus
from psireg.utils.logger import logger


def demo_basic_prediction():
    """Demonstrate basic prediction functionality."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Prediction Functionality")
    print("=" * 60)

    # Load a model (will use fallback since no real model exists)
    predictor = PredictiveLayer.load_model("demo_model.zip")

    print(f"✓ Predictor loaded: {predictor.is_model_loaded()}")
    print(f"✓ Model info: {predictor.get_model_info()}")

    # Create sample observation (typical grid state observation)
    observation = np.array(
        [
            1.0,  # frequency (normalized to 60Hz)
            0.75,  # total generation (normalized)
            0.70,  # total load (normalized)
            0.05,  # storage charging (normalized)
            0.02,  # grid losses (normalized)
            0.03,  # power balance (normalized)
            0.65,  # battery SoC
            0.80,  # load factor
            0.58,  # hour of day (normalized)
            0.29,  # day of week (normalized)
        ],
        dtype=np.float32,
    )

    print(f"✓ Observation shape: {observation.shape}")
    print(f"✓ Observation: {observation}")

    # Make prediction
    action = predictor.predict(observation, deterministic=True)

    print(f"✓ Action shape: {action.shape}")
    print(f"✓ Predicted action: {action}")
    print("✓ Action interpretation:")
    print(f"  - Battery control: {action[0]:.3f} (charge if >0, discharge if <0)")
    if len(action) > 1:
        print(f"  - Load control: {action[1]:.3f} (increase if >0, reduce if <0)")

    return predictor


def demo_batch_prediction(predictor):
    """Demonstrate batch prediction functionality."""
    print("\n" + "=" * 60)
    print("DEMO 2: Batch Prediction Functionality")
    print("=" * 60)

    # Create multiple observations representing different grid scenarios
    scenarios = {
        "morning_peak": np.array([1.0, 0.6, 0.9, 0.0, 0.03, 0.27, 0.45, 0.95, 0.33, 0.14], dtype=np.float32),
        "midday_solar": np.array([1.0, 0.95, 0.7, -0.1, 0.02, 0.13, 0.85, 0.75, 0.5, 0.14], dtype=np.float32),
        "evening_peak": np.array([1.0, 0.7, 0.95, 0.1, 0.04, 0.11, 0.55, 0.98, 0.75, 0.14], dtype=np.float32),
        "night_low": np.array([1.0, 0.4, 0.3, -0.05, 0.01, 0.05, 0.75, 0.35, 0.08, 0.14], dtype=np.float32),
    }

    observations = list(scenarios.values())
    scenario_names = list(scenarios.keys())

    print(f"✓ Created {len(observations)} scenarios: {scenario_names}")

    # Make batch prediction
    actions = predictor.predict_batch(observations, deterministic=True)

    print("✓ Batch prediction completed")
    print("✓ Actions for each scenario:")

    for name, action in zip(scenario_names, actions, strict=False):
        battery_action = action[0]
        load_action = action[1] if len(action) > 1 else 0.0

        battery_desc = "Charge" if battery_action > 0 else "Discharge" if battery_action < 0 else "Hold"
        load_desc = "Increase" if load_action > 0 else "Reduce" if load_action < 0 else "Maintain"

        print(
            f"  - {name:12}: Battery: {battery_desc:9} ({battery_action:+.3f}), "
            f"Load: {load_desc:8} ({load_action:+.3f})"
        )


def demo_performance_monitoring(predictor):
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "=" * 60)
    print("DEMO 3: Performance Monitoring")
    print("=" * 60)

    # Make several predictions to generate performance data
    observation = np.array([1.0, 0.75, 0.70, 0.05, 0.02, 0.03, 0.65, 0.80, 0.58, 0.29], dtype=np.float32)

    print("✓ Making multiple predictions for performance analysis...")

    start_time = datetime.now()
    for i in range(10):
        # Slightly vary the observation
        varied_obs = observation + np.random.normal(0, 0.01, observation.shape).astype(np.float32)
        action = predictor.predict(varied_obs)
    end_time = datetime.now()

    total_time = (end_time - start_time).total_seconds()
    avg_time_ms = (total_time / 10) * 1000

    print(f"✓ Completed 10 predictions in {total_time:.3f}s")
    print(f"✓ Average prediction time: {avg_time_ms:.2f}ms")

    # Get performance metrics
    metrics = predictor.get_performance_metrics()

    print("✓ Performance metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.6f}")
        else:
            print(f"  - {key}: {value}")


def demo_grid_integration():
    """Demonstrate integration with GridEngine simulation."""
    print("\n" + "=" * 60)
    print("DEMO 4: Grid Integration")
    print("=" * 60)

    # Create grid simulation
    sim_config = SimulationConfig(timestep_minutes=15)
    grid_config = GridConfig(frequency_hz=60.0)
    engine = GridEngine(sim_config, grid_config)

    # Add network node
    node = NetworkNode(node_id="main_bus", name="Main Bus", voltage_kv=138.0)
    engine.add_node(node)

    # Add assets
    solar = SolarPanel(
        asset_id="solar_1",
        name="Solar Farm 1",
        node_id="main_bus",
        capacity_mw=100.0,
        panel_efficiency=0.2,
        panel_area_m2=50000.0,
    )

    battery = Battery(
        asset_id="battery_1",
        name="Grid Battery 1",
        node_id="main_bus",
        capacity_mw=50.0,
        energy_capacity_mwh=200.0,
    )

    load = Load(
        asset_id="load_1",
        name="City Load 1",
        node_id="main_bus",
        capacity_mw=120.0,
        baseline_demand_mw=90.0,
    )

    # Add assets to grid
    engine.add_asset(solar)
    engine.add_asset(battery)
    engine.add_asset(load)

    # Set assets online
    solar.set_status(AssetStatus.ONLINE)
    battery.set_status(AssetStatus.ONLINE)
    load.set_status(AssetStatus.ONLINE)

    print(f"✓ Grid created with {len(engine.assets)} assets")
    print(f"✓ Assets: {list(engine.assets.keys())}")

    # Load predictor for grid control
    predictor = PredictiveLayer.load_model("grid_control_model.zip")

    print("✓ Control predictor loaded")

    # Simulate grid operation with RL control
    print("✓ Running grid simulation with RL control...")

    for step in range(5):
        # Get current grid state
        grid_state = engine.get_state()

        # Create observation from grid state
        observation = np.array(
            [
                grid_state.frequency_hz / 60.0,  # Normalized frequency
                grid_state.total_generation_mw / 1000.0,  # Normalized generation
                grid_state.total_load_mw / 1000.0,  # Normalized load
                grid_state.total_storage_mw / 1000.0,  # Normalized storage
                grid_state.grid_losses_mw / 1000.0,  # Normalized losses
                grid_state.power_balance_mw / 1000.0,  # Normalized balance
                battery.current_soc_percent / 100.0,  # Battery SoC
                load.current_output_mw / load.capacity_mw,  # Load factor
                grid_state.timestamp.hour / 24.0,  # Hour of day
                grid_state.timestamp.weekday() / 7.0,  # Day of week
            ],
            dtype=np.float32,
        )

        # Predict optimal action
        action = predictor.predict(observation, deterministic=True)

        # Apply actions to assets
        battery_action = action[0]
        load_action = action[1] if len(action) > 1 else 0.0

        # Apply battery control
        if battery_action >= 0:
            power_setpoint = battery_action * battery.get_max_charge_power()
        else:
            power_setpoint = battery_action * battery.get_max_discharge_power()
        battery.set_power_setpoint(power_setpoint)

        # Apply load control (demand response)
        dr_signal = load_action * load.baseline_demand_mw * 0.1  # 10% DR capability
        load.set_demand_response_signal(dr_signal)

        # Step simulation
        engine.step(timedelta(minutes=15))

        # Log results
        new_state = engine.get_state()
        print(
            f"  Step {step+1}: Freq={new_state.frequency_hz:.2f}Hz, "
            f"Gen={new_state.total_generation_mw:.1f}MW, "
            f"Load={new_state.total_load_mw:.1f}MW, "
            f"Battery_SoC={battery.current_soc_percent:.1f}%"
        )


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n" + "=" * 60)
    print("DEMO 5: Convenience Functions")
    print("=" * 60)

    # Quick model loading
    predictor = load_predictor("convenience_model.zip", device="cpu")
    print(f"✓ Quick model load: {predictor.is_model_loaded()}")

    # One-shot prediction
    observation = np.array([1.0, 0.8, 0.7, 0.0, 0.02, 0.08, 0.6, 0.85, 0.45, 0.3], dtype=np.float32)
    action = predict_action("oneshot_model.zip", observation, deterministic=True)

    print(f"✓ One-shot prediction: {action}")
    print("✓ Convenience functions provide simple API for common use cases")


def main():
    """Run all demos."""
    print("🔋 PredictiveLayer Demo - PPO Inference Service")
    print("🌞 Renewable Energy Grid Optimization")
    print("⚡ Intelligent Grid Control with Reinforcement Learning")

    try:
        # Run demos
        predictor = demo_basic_prediction()
        demo_batch_prediction(predictor)
        demo_performance_monitoring(predictor)
        demo_grid_integration()
        demo_convenience_functions()

        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print("\n📊 Summary:")
        print("• PredictiveLayer provides clean predict(obs) interface")
        print("• Graceful fallback when RL dependencies unavailable")
        print("• Batch prediction for multiple scenarios")
        print("• Performance monitoring and metrics")
        print("• Seamless integration with GridEngine simulation")
        print("• Convenience functions for quick usage")
        print("\n🎯 The PredictiveLayer serves as the primary output")
        print("   interface for the PPO Inference Service!")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("This is expected if RL dependencies are not installed.")
        print("The fallback predictor should still work for basic demonstration.")


if __name__ == "__main__":
    main()
