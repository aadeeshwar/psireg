"""Comprehensive Pheromone Field and Swarm Bus Demonstration.

This demo showcases the complete swarm intelligence infrastructure including:
- PheromoneField with decay and diffusion
- SwarmBus for agent coordination
- Multi-agent coordination between battery and demand agents
- Real-time pheromone field visualization
- Emergency grid response scenarios
"""

import time
import numpy as np
from typing import Dict, List, Any

from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.swarm import (
    SwarmBus,
    PheromoneField,
    PheromoneType,
    GridPosition,
    BatteryAgent,
    DemandAgent,
)
from psireg.utils.enums import AssetStatus


def create_grid_assets() -> Dict[str, Any]:
    """Create a set of grid assets for demonstration."""
    print("Creating grid assets...")
    
    assets = {}
    
    # Create battery systems
    batteries = []
    for i in range(3):
        battery = Battery(
            asset_id=f"battery_{i+1:03d}",
            name=f"Battery Storage {i+1}",
            node_id=f"node_battery_{i+1}",
            capacity_mw=5.0 + i * 2.0,  # 5, 7, 9 MW
            energy_capacity_mwh=20.0 + i * 8.0,  # 20, 28, 36 MWh
            initial_soc_percent=50.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
        )
        battery.set_status(AssetStatus.ONLINE)
        batteries.append(battery)
    
    # Create load centers
    loads = []
    load_configs = [
        {"name": "Residential District", "baseline": 25.0, "dr_capability": 8.0, "elasticity": -0.3},
        {"name": "Commercial Center", "baseline": 40.0, "dr_capability": 15.0, "elasticity": -0.2},
        {"name": "Industrial Complex", "baseline": 60.0, "dr_capability": 20.0, "elasticity": -0.4},
        {"name": "Data Center", "baseline": 35.0, "dr_capability": 12.0, "elasticity": -0.15},
    ]
    
    for i, config in enumerate(load_configs):
        load = Load(
            asset_id=f"load_{i+1:03d}",
            name=config["name"],
            node_id=f"node_load_{i+1}",
            capacity_mw=config["baseline"] * 1.5,
            baseline_demand_mw=config["baseline"],
            dr_capability_mw=config["dr_capability"],
            price_elasticity=config["elasticity"],
        )
        load.set_status(AssetStatus.ONLINE)
        loads.append(load)
    
    assets["batteries"] = batteries
    assets["loads"] = loads
    
    print(f"Created {len(batteries)} batteries and {len(loads)} loads")
    return assets


def create_swarm_agents(assets: Dict[str, Any], swarm_bus: SwarmBus) -> Dict[str, Any]:
    """Create and register swarm agents with the bus."""
    print("\nCreating and registering swarm agents...")
    
    agents = {"battery": [], "demand": []}
    
    # Create battery agents
    for i, battery in enumerate(assets["batteries"]):
        agent = BatteryAgent(
            battery=battery,
            communication_range=4.0,
            response_time_s=1.0,
            coordination_weight=0.3,
        )
        
        # Position batteries in a line
        position = GridPosition(x=2 + i * 3, y=5)
        success = swarm_bus.register_agent(agent, position)
        
        if success:
            agents["battery"].append(agent)
            print(f"  Registered {agent.agent_id} at ({position.x}, {position.y})")
        else:
            print(f"  Failed to register {agent.agent_id}")
    
    # Create demand agents
    for i, load in enumerate(assets["loads"]):
        agent = DemandAgent(
            load=load,
            communication_range=3.5,
            response_time_s=2.0,
            coordination_weight=0.25,
        )
        
        # Position loads in a grid pattern
        position = GridPosition(x=1 + (i % 2) * 8, y=2 + (i // 2) * 6)
        success = swarm_bus.register_agent(agent, position)
        
        if success:
            agents["demand"].append(agent)
            print(f"  Registered {agent.agent_id} at ({position.x}, {position.y})")
        else:
            print(f"  Failed to register {agent.agent_id}")
    
    total_agents = len(agents["battery"]) + len(agents["demand"])
    print(f"\nSuccessfully registered {total_agents} agents")
    
    return agents


def simulate_grid_conditions(scenario: str) -> Dict[str, float]:
    """Simulate different grid conditions."""
    scenarios = {
        "normal": {
            "frequency_hz": 60.0,
            "voltage_kv": 230.0,
            "grid_stress": 0.2,
            "electricity_price": 50.0,
            "renewable_output": 0.7,
        },
        "peak_demand": {
            "frequency_hz": 59.9,
            "voltage_kv": 228.0,
            "grid_stress": 0.6,
            "electricity_price": 80.0,
            "renewable_output": 0.4,
        },
        "emergency": {
            "frequency_hz": 59.7,
            "voltage_kv": 225.0,
            "grid_stress": 0.9,
            "electricity_price": 150.0,
            "renewable_output": 0.3,
        },
        "surplus": {
            "frequency_hz": 60.1,
            "voltage_kv": 232.0,
            "grid_stress": 0.1,
            "electricity_price": 20.0,
            "renewable_output": 1.2,
        }
    }
    
    return scenarios.get(scenario, scenarios["normal"])


def update_agent_conditions(agents: Dict[str, Any], conditions: Dict[str, float]):
    """Update all agents with current grid conditions."""
    total_load = sum(load.baseline_demand_mw for load in [agent.load for agent in agents["demand"]])
    total_generation = total_load * conditions["renewable_output"]
    
    # Update battery agents
    for agent in agents["battery"]:
        agent.local_grid_stress = conditions["grid_stress"]
        agent.electricity_price = conditions["electricity_price"]
    
    # Update demand agents  
    for agent in agents["demand"]:
        agent.local_grid_stress = conditions["grid_stress"]
        agent.load.set_electricity_price(conditions["electricity_price"])


def demonstrate_pheromone_field():
    """Demonstrate basic pheromone field functionality."""
    print("\n" + "=" * 80)
    print("PHEROMONE FIELD DEMONSTRATION")
    print("=" * 80)
    
    # Create a small pheromone field for demonstration
    field = PheromoneField(
        grid_width=8,
        grid_height=8,
        decay_rate=0.9,
        diffusion_rate=0.15,
        time_step_s=1.0,
    )
    
    print(f"Created {field.grid_width}x{field.grid_height} pheromone field")
    print(f"Decay rate: {field.decay_rate}, Diffusion rate: {field.diffusion_rate}")
    
    # Deposit initial pheromones
    center = GridPosition(x=4, y=4)
    field.deposit_pheromone(center, PheromoneType.DEMAND_REDUCTION, 1.0)
    field.deposit_pheromone(GridPosition(x=2, y=2), PheromoneType.FREQUENCY_SUPPORT, 0.8)
    field.deposit_pheromone(GridPosition(x=6, y=6), PheromoneType.ECONOMIC_SIGNAL, 0.6)
    
    print(f"\nInitial pheromone deposits:")
    print(f"  DEMAND_REDUCTION at (4,4): 1.0")
    print(f"  FREQUENCY_SUPPORT at (2,2): 0.8")
    print(f"  ECONOMIC_SIGNAL at (6,6): 0.6")
    
    # Simulate time steps and show evolution
    print(f"\nPheromone field evolution:")
    
    for step in range(5):
        field.update_time_step()
        
        # Get totals for each pheromone type
        demand_total = field.get_total_pheromone(PheromoneType.DEMAND_REDUCTION)
        freq_total = field.get_total_pheromone(PheromoneType.FREQUENCY_SUPPORT)
        econ_total = field.get_total_pheromone(PheromoneType.ECONOMIC_SIGNAL)
        
        print(f"  Step {step+1}: DEMAND={demand_total:.3f}, FREQ={freq_total:.3f}, ECON={econ_total:.3f}")
        
        # Show neighborhood around center
        if step == 2:  # Show detailed view at step 3
            neighborhood = field.get_neighborhood_pheromones(center, PheromoneType.DEMAND_REDUCTION, radius=1)
            print(f"    Neighborhood around (4,4):")
            for pos, strength in neighborhood:
                if strength > 0.01:
                    print(f"      ({pos.x},{pos.y}): {strength:.3f}")


def demonstrate_swarm_coordination():
    """Demonstrate full swarm coordination."""
    print("\n" + "=" * 80)
    print("SWARM COORDINATION DEMONSTRATION")
    print("=" * 80)
    
    # Create swarm bus
    swarm_bus = SwarmBus(
        grid_width=12,
        grid_height=12,
        pheromone_decay=0.95,
        pheromone_diffusion=0.1,
        time_step_s=5.0,  # 5-second time steps
        communication_range=4.0,
    )
    
    print(f"Created SwarmBus: {swarm_bus.grid_width}x{swarm_bus.grid_height}")
    
    # Create assets and agents
    assets = create_grid_assets()
    agents = create_swarm_agents(assets, swarm_bus)
    
    # Simulation scenarios
    scenarios = ["normal", "peak_demand", "emergency", "surplus"]
    
    print(f"\nSimulating {len(scenarios)} grid scenarios...")
    
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {scenario_idx + 1}: {scenario.upper()} ---")
        
        # Update grid conditions
        conditions = simulate_grid_conditions(scenario)
        update_agent_conditions(agents, conditions)
        
        print(f"Grid conditions:")
        print(f"  Frequency: {conditions['frequency_hz']:.1f} Hz")
        print(f"  Grid stress: {conditions['grid_stress']:.1f}")
        print(f"  Price: ${conditions['electricity_price']:.0f}/MWh")
        print(f"  Renewable output: {conditions['renewable_output']:.1%}")
        
        # Coordinate agents
        coordination_results = swarm_bus.coordinate_agents(update_pheromones=True)
        
        print(f"\nAgent coordination results:")
        total_neighbors = 0
        for agent_id, neighbors in coordination_results.items():
            print(f"  {agent_id}: {len(neighbors)} neighbors")
            total_neighbors += len(neighbors)
        
        avg_neighbors = total_neighbors / len(coordination_results) if coordination_results else 0
        print(f"  Average neighbors per agent: {avg_neighbors:.1f}")
        
        # Calculate agent responses
        print(f"\nAgent responses:")
        
        # Battery agent responses
        for agent in agents["battery"]:
            forecast_load = [100.0] * 4
            forecast_generation = [80.0] * 4
            forecast_prices = [conditions["electricity_price"]] * 4
            
            optimal_power = agent.calculate_optimal_power(
                forecast_load, forecast_generation, forecast_prices
            )
            
            soc = agent.battery.current_soc_percent
            action_type = "charging" if optimal_power > 0 else "discharging" if optimal_power < 0 else "idle"
            
            print(f"  {agent.agent_id} (SoC: {soc:.1f}%): {action_type} at {abs(optimal_power):.1f} MW")
            
            # Execute control action
            agent.execute_control_action(optimal_power)
        
        # Demand agent responses
        for agent in agents["demand"]:
            forecast_prices = [conditions["electricity_price"]] * 4
            forecast_generation = [80.0] * 4
            forecast_stress = [conditions["grid_stress"]] * 4
            
            optimal_demand = agent.calculate_optimal_demand(
                forecast_prices, forecast_generation, forecast_stress
            )
            
            baseline = agent.load.baseline_demand_mw
            adjustment = optimal_demand - baseline
            adjustment_pct = (adjustment / baseline) * 100 if baseline > 0 else 0
            
            print(f"  {agent.agent_id}: {adjustment:+.1f} MW ({adjustment_pct:+.1f}%)")
            
            # Execute control action
            agent.execute_control_action(optimal_demand)
        
        # Update swarm bus time step
        swarm_bus.update_time_step()
        
        # Get system statistics
        stats = swarm_bus.get_system_stats()
        pheromone_totals = stats["pheromone_totals"]
        
        print(f"\nPheromone field status:")
        for pheromone_type, total in pheromone_totals.items():
            if total > 0.01:
                print(f"  {pheromone_type}: {total:.3f}")
        
        print(f"System time: {stats['current_time']:.1f}s")
        
        # Short pause between scenarios
        if scenario_idx < len(scenarios) - 1:
            time.sleep(1)


def demonstrate_emergency_response():
    """Demonstrate coordinated emergency response."""
    print("\n" + "=" * 80)
    print("EMERGENCY RESPONSE DEMONSTRATION")
    print("=" * 80)
    
    # Create a focused emergency scenario
    swarm_bus = SwarmBus(
        grid_width=8,
        grid_height=8,
        pheromone_decay=0.85,  # Faster decay for emergency
        pheromone_diffusion=0.2,  # Higher diffusion for rapid coordination
        time_step_s=1.0,  # 1-second time steps
        communication_range=6.0,  # Extended range for emergency
    )
    
    print("Simulating grid emergency scenario...")
    
    # Create emergency assets
    emergency_battery = Battery(
        asset_id="emergency_battery",
        name="Emergency Response Battery",
        node_id="emergency_node",
        capacity_mw=10.0,
        energy_capacity_mwh=40.0,
        initial_soc_percent=90.0,  # Full for emergency
    )
    emergency_battery.set_status(AssetStatus.ONLINE)
    
    critical_load = Load(
        asset_id="critical_load",
        name="Critical Infrastructure",
        node_id="critical_node",
        capacity_mw=50.0,
        baseline_demand_mw=30.0,
        dr_capability_mw=25.0,  # High DR capability
        price_elasticity=-0.5,  # Very responsive
    )
    critical_load.set_status(AssetStatus.ONLINE)
    
    # Create agents
    battery_agent = BatteryAgent(emergency_battery, coordination_weight=0.5)
    demand_agent = DemandAgent(critical_load, coordination_weight=0.4)
    
    # Register agents close together for emergency coordination
    swarm_bus.register_agent(battery_agent, GridPosition(x=3, y=4))
    swarm_bus.register_agent(demand_agent, GridPosition(x=4, y=4))
    
    print(f"Emergency agents registered")
    print(f"  Battery: {battery_agent.agent_id} (SoC: {emergency_battery.current_soc_percent:.0f}%)")
    print(f"  Load: {demand_agent.agent_id} (Demand: {critical_load.baseline_demand_mw:.0f} MW)")
    
    # Simulate emergency conditions escalation
    emergency_steps = [
        {"stress": 0.3, "price": 60.0, "desc": "Grid stress detected"},
        {"stress": 0.6, "price": 100.0, "desc": "High stress - frequency deviation"},
        {"stress": 0.9, "price": 200.0, "desc": "EMERGENCY - Grid instability"},
        {"stress": 0.95, "price": 300.0, "desc": "CRITICAL - Emergency response"},
        {"stress": 0.7, "price": 150.0, "desc": "Stabilizing"},
        {"stress": 0.4, "price": 80.0, "desc": "Recovery phase"},
    ]
    
    print(f"\nEmergency response sequence:")
    
    for step_idx, step_conditions in enumerate(emergency_steps):
        print(f"\nStep {step_idx + 1}: {step_conditions['desc']}")
        
        # Update agent conditions
        battery_agent.local_grid_stress = step_conditions["stress"]
        battery_agent.electricity_price = step_conditions["price"]
        demand_agent.local_grid_stress = step_conditions["stress"]
        demand_agent.load.set_electricity_price(step_conditions["price"])
        
        # Deposit emergency pheromone based on stress level
        if step_conditions["stress"] > 0.8:
            emergency_strength = min(step_conditions["stress"], 1.0)
            swarm_bus.deposit_pheromone(
                battery_agent.agent_id,
                PheromoneType.EMERGENCY_RESPONSE,
                emergency_strength
            )
            swarm_bus.deposit_pheromone(
                demand_agent.agent_id,
                PheromoneType.EMERGENCY_RESPONSE,
                emergency_strength * 0.8
            )
        
        # Coordinate agents
        coordination_results = swarm_bus.coordinate_agents(update_pheromones=True)
        
        # Calculate responses
        forecast_prices = [step_conditions["price"]] * 4
        forecast_stress = [step_conditions["stress"]] * 4
        
        # Battery response
        battery_power = battery_agent.calculate_optimal_power(
            [50.0] * 4, [30.0] * 4, forecast_prices
        )
        battery_agent.execute_control_action(battery_power)
        
        # Demand response
        demand_optimal = demand_agent.calculate_optimal_demand(
            forecast_prices, [30.0] * 4, forecast_stress
        )
        demand_agent.execute_control_action(demand_optimal)
        
        # Report actions
        battery_action = "discharging" if battery_power < 0 else "charging" if battery_power > 0 else "idle"
        demand_change = demand_optimal - critical_load.baseline_demand_mw
        
        print(f"  Grid stress: {step_conditions['stress']:.1%}, Price: ${step_conditions['price']:.0f}/MWh")
        print(f"  Battery action: {battery_action} ({abs(battery_power):.1f} MW)")
        print(f"  Demand adjustment: {demand_change:+.1f} MW")
        
        # Update time step
        swarm_bus.update_time_step()
        
        # Get emergency pheromone levels
        emergency_total = swarm_bus.pheromone_field.get_total_pheromone(PheromoneType.EMERGENCY_RESPONSE)
        if emergency_total > 0.01:
            print(f"  Emergency pheromone level: {emergency_total:.3f}")
        
        time.sleep(0.5)  # Brief pause for visualization


def demonstrate_performance_metrics():
    """Demonstrate system performance and metrics."""
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS DEMONSTRATION")
    print("=" * 80)
    
    # Create large-scale system
    swarm_bus = SwarmBus(
        grid_width=20,
        grid_height=20,
        max_agents=50,
    )
    
    print(f"Testing performance with large-scale system ({swarm_bus.grid_width}x{swarm_bus.grid_height})")
    
    # Create many mock agents for performance testing
    from unittest.mock import Mock
    
    agents = []
    for i in range(30):
        agent = Mock()
        agent.agent_id = f"performance_agent_{i:03d}"
        agent.communication_range = 3.0
        agent.get_coordination_signal.return_value = 0.1
        agent.get_pheromone_strength.return_value = 0.05
        
        # Distribute agents across grid
        x = i % swarm_bus.grid_width
        y = i // swarm_bus.grid_width
        position = GridPosition(x=x, y=y)
        
        success = swarm_bus.register_agent(agent, position)
        if success:
            agents.append(agent)
    
    print(f"Registered {len(agents)} performance test agents")
    
    # Test coordination performance
    start_time = time.time()
    
    for iteration in range(10):
        coordination_results = swarm_bus.coordinate_agents(update_pheromones=True)
        swarm_bus.update_time_step()
    
    end_time = time.time()
    coordination_time = end_time - start_time
    
    print(f"\nPerformance results:")
    print(f"  10 coordination cycles: {coordination_time:.3f} seconds")
    print(f"  Average per cycle: {coordination_time/10:.3f} seconds")
    print(f"  Throughput: {len(agents)*10/coordination_time:.1f} agent-operations/second")
    
    # Get final system statistics
    stats = swarm_bus.get_system_stats()
    print(f"\nFinal system statistics:")
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Grid size: {stats['grid_size']}")
    print(f"  System time: {stats['current_time']:.1f}s")
    
    # Memory usage estimation
    field_size = swarm_bus.pheromone_field.pheromone_grid.nbytes
    print(f"  Pheromone field memory: {field_size/1024:.1f} KB")


def main():
    """Run all demonstrations."""
    print("PSIREG Swarm Intelligence Demonstration")
    print("Pheromone Field Infrastructure & Swarm Bus Coordination")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        demonstrate_pheromone_field()
        demonstrate_swarm_coordination()
        demonstrate_emergency_response()
        demonstrate_performance_metrics()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("Successfully demonstrated:")
        print("✓ Pheromone field with decay and diffusion")
        print("✓ SwarmBus agent registration and coordination")
        print("✓ Multi-agent grid coordination scenarios")
        print("✓ Emergency response coordination")
        print("✓ Performance metrics and scalability")
        print("\nThe swarm intelligence infrastructure is ready for production use!")
        
    except Exception as e:
        print(f"\nERROR during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 