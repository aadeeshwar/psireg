#!/usr/bin/env python3
"""Real PSIREG Simulation Demo with Visualization Data Generation.

This script demonstrates using actual PSIREG simulation components to generate
real metrics data that would be visualized with the visualization module.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psireg import GridEngine, NetworkNode, TransmissionLine
from psireg.sim.assets import SolarPanel, WindTurbine, Battery, Load
from psireg.swarm.agents import BatteryAgent, DemandAgent
from psireg.config.schema import SimulationConfig, GridConfig
from psireg.utils.enums import AssetStatus


def create_renewable_grid_simulation():
    """Create a comprehensive renewable grid simulation with all asset types."""
    
    # Create configuration objects
    simulation_config = SimulationConfig(
        timestep_minutes=15,
        horizon_hours=48,
        max_assets=100
    )
    
    grid_config = GridConfig(
        frequency_hz=60.0,
        voltage_kv=230.0,
        stability_threshold=0.1,
        max_power_mw=1000.0
    )
    
    # Initialize grid engine
    engine = GridEngine(
        simulation_config=simulation_config,
        grid_config=grid_config,
        start_time=datetime(2023, 6, 15, 0, 0)  # Summer day
    )
    
    # Create network nodes
    generation_node = NetworkNode(
        node_id="generation_hub",
        name="Generation Hub",
        voltage_kv=138.0,
        x_coordinate=0.0,
        y_coordinate=0.0
    )
    
    load_node = NetworkNode(
        node_id="load_center",
        name="Load Center",
        voltage_kv=138.0,
        x_coordinate=5.0,
        y_coordinate=0.0
    )
    
    storage_node = NetworkNode(
        node_id="storage_facility",
        name="Storage Facility",
        voltage_kv=138.0,
        x_coordinate=2.5,
        y_coordinate=0.0
    )
    
    # Add nodes to engine
    engine.add_node(generation_node)
    engine.add_node(load_node)
    engine.add_node(storage_node)
    
    # Create transmission lines
    gen_to_load_line = TransmissionLine(
        line_id="gen_to_load",
        name="Generation to Load",
        from_node="generation_hub",
        to_node="load_center",
        length_km=5.0,
        resistance=0.1,
        reactance=0.4,
        capacity_mw=300.0
    )
    
    gen_to_storage_line = TransmissionLine(
        line_id="gen_to_storage",
        name="Generation to Storage",
        from_node="generation_hub",
        to_node="storage_facility",
        length_km=2.5,
        resistance=0.05,
        reactance=0.2,
        capacity_mw=150.0
    )
    
    storage_to_load_line = TransmissionLine(
        line_id="storage_to_load",
        name="Storage to Load",
        from_node="storage_facility",
        to_node="load_center",
        length_km=2.5,
        resistance=0.05,
        reactance=0.2,
        capacity_mw=150.0
    )
    
    # Add transmission lines to engine
    engine.add_transmission_line(gen_to_load_line)
    engine.add_transmission_line(gen_to_storage_line)
    engine.add_transmission_line(storage_to_load_line)
    
    # Add renewable generation assets
    solar_farm = SolarPanel(
        asset_id="solar_farm_1",
        name="Solar Farm 1",
        node_id="generation_hub",
        capacity_mw=150.0,
        efficiency=0.22,
        tilt_angle=35.0,
        azimuth_angle=180.0,  # South-facing
        temperature_coefficient=-0.004
    )
    engine.add_asset(solar_farm)
    solar_farm.set_status(AssetStatus.ONLINE)  # Activate the asset
    
    wind_farm = WindTurbine(
        asset_id="wind_farm_1",
        name="Wind Farm 1",
        node_id="generation_hub",
        capacity_mw=100.0,
        cut_in_speed=3.0,
        rated_speed=12.0,
        cut_out_speed=25.0,
        hub_height=80.0,
        rotor_diameter=80.0
    )
    engine.add_asset(wind_farm)
    wind_farm.set_status(AssetStatus.ONLINE)  # Activate the asset
    
    # Add battery storage with intelligent agent
    battery = Battery(
        asset_id="grid_battery_1",
        name="Grid Battery 1",
        node_id="storage_facility",
        capacity_mw=50.0,
        energy_capacity_mwh=200.0,
        max_charge_rate_mw=50.0,
        max_discharge_rate_mw=50.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        initial_soc=0.5
    )
    engine.add_asset(battery)
    battery.set_status(AssetStatus.ONLINE)  # Activate the asset
    
    battery_agent = BatteryAgent(
        battery=battery,
        agent_id="battery_agent_1",
        communication_range=5.0,
        response_time_s=1.0,
        coordination_weight=0.3
    )
    
    # Add demand loads with intelligent demand response
    residential_load = Load(
        asset_id="residential_load_1",
        name="Residential Load 1",
        node_id="load_center",
        capacity_mw=120.0,
        baseline_demand_mw=80.0,
        peak_demand_mw=120.0,
        profile_type="time_of_use",
        price_elasticity=0.15
    )
    engine.add_asset(residential_load)
    residential_load.set_status(AssetStatus.ONLINE)  # Activate the asset
    
    commercial_load = Load(
        asset_id="commercial_load_1",
        name="Commercial Load 1",
        node_id="load_center",
        capacity_mw=100.0,
        baseline_demand_mw=60.0,
        peak_demand_mw=100.0,
        profile_type="stochastic",
        price_elasticity=0.25
    )
    engine.add_asset(commercial_load)
    commercial_load.set_status(AssetStatus.ONLINE)  # Activate the asset
    
    # Add demand agents for intelligent coordination
    residential_agent = DemandAgent(
        load=residential_load,
        agent_id="residential_agent_1",
        communication_range=5.0,
        response_time_s=10.0,
        coordination_weight=0.25
    )
    
    commercial_agent = DemandAgent(
        load=commercial_load,
        agent_id="commercial_agent_1",
        communication_range=5.0,
        response_time_s=10.0,
        coordination_weight=0.25
    )
    
    return engine, {
        'battery_agent': battery_agent,
        'residential_agent': residential_agent, 
        'commercial_agent': commercial_agent
    }


def simulate_grid_operations(engine, agents, duration_hours=48):
    """Run a comprehensive grid simulation and collect metrics data."""
    
    print(f"Starting {duration_hours}-hour grid simulation...")
    print(f"Assets: {len(engine.get_all_assets())} active")
    print(f"Nodes: {len(engine.nodes)} grid nodes")
    print(f"Transmission Lines: {len(engine.transmission_lines)}")
    
    # Prepare metrics data structure
    metrics_data = {
        'timestamp': [],
        'solar_output_mw': [],
        'wind_output_mw': [],
        'battery_charge_mw': [],
        'battery_discharge_mw': [],
        'battery_soc': [],
        'demand_mw': [],
        'net_balance_mw': [],
        'curtailed_energy_mw': [],
        'unmet_demand_mw': []
    }
    
    # Simulation parameters
    time_step_seconds = engine.simulation_config.timestep_minutes * 60
    total_steps = (duration_hours * 3600) // time_step_seconds
    
    for step in range(total_steps):
        current_time = engine.current_time
        
        # Update environmental conditions (simulate realistic patterns)
        hour_of_day = current_time.hour + current_time.minute / 60.0
        
        # Solar irradiance pattern (bell curve, peak at noon)
        import math
        solar_irradiance = max(0, 1000 * math.exp(-0.5 * ((hour_of_day - 12) / 4) ** 2))
        # Add some cloud variability
        solar_irradiance *= (0.8 + 0.4 * math.sin(step * 0.1))
        solar_irradiance = max(0, min(1000, solar_irradiance))
        
        # Wind speed pattern (more variable, average higher at night)
        wind_speed = 8 + 4 * math.sin(2 * math.pi * hour_of_day / 24) + 2 * math.sin(step * 0.05)
        wind_speed = max(2, min(20, wind_speed))
        
        # Temperature pattern (cooler at night)
        temperature = 25 + 10 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)
        
        # Update asset environmental conditions
        for asset in engine.get_all_assets():
            if hasattr(asset, 'irradiance_w_m2'):
                asset.irradiance_w_m2 = solar_irradiance
            if hasattr(asset, 'wind_speed_ms'):
                asset.wind_speed_ms = wind_speed
            if hasattr(asset, 'ambient_temperature_c'):
                asset.ambient_temperature_c = temperature
        
        # Execute intelligent agent control
        grid_state = engine.get_state()
        
        # Battery agent optimization
        agents['battery_agent'].update_grid_conditions(
            frequency_hz=grid_state.frequency_hz,
            voltage_kv=138.0,  # Use nominal voltage
            local_load_mw=grid_state.total_load_mw,
            local_generation_mw=grid_state.total_generation_mw,
            electricity_price=50 + 25 * math.sin(2 * math.pi * hour_of_day / 24)  # Time-of-use pricing
        )
        
        # Calculate optimal power and execute
        optimal_power = agents['battery_agent'].calculate_optimal_power(
            forecast_load=[grid_state.total_load_mw] * 24,
            forecast_generation=[grid_state.total_generation_mw] * 24,
            forecast_prices=[50 + 25 * math.sin(2 * math.pi * h / 24) for h in range(24)]
        )
        agents['battery_agent'].execute_control_action(optimal_power)
        
        # Demand agent coordination
        for agent_name in ['residential_agent', 'commercial_agent']:
            agents[agent_name].update_grid_conditions(
                frequency_hz=grid_state.frequency_hz,
                voltage_kv=138.0,  # Use nominal voltage
                local_load_mw=grid_state.total_load_mw,
                local_generation_mw=grid_state.total_generation_mw,
                electricity_price=50 + 25 * math.sin(2 * math.pi * hour_of_day / 24)
            )
            
            # Calculate optimal demand and execute
            optimal_demand = agents[agent_name].calculate_optimal_demand(
                forecast_prices=[50 + 25 * math.sin(2 * math.pi * h / 24) for h in range(24)],
                forecast_generation=[grid_state.total_generation_mw] * 24,
                forecast_grid_stress=[0.2] * 24
            )
            agents[agent_name].execute_control_action(optimal_demand)
        
        # Step the simulation
        engine.step(timedelta(seconds=time_step_seconds))
        
        # Collect metrics
        state = engine.get_state()
        
        # Calculate individual asset outputs
        solar_output = 0.0
        wind_output = 0.0
        battery_charge = 0.0
        battery_discharge = 0.0
        battery_soc = 0.0
        total_demand = 0.0
        
        for asset in engine.get_all_assets():
            power = asset.current_output_mw
            
            if asset.asset_type.value == "solar":
                solar_output += max(0, power)
            elif asset.asset_type.value == "wind":
                wind_output += max(0, power)
            elif asset.asset_type.value == "battery":
                if power > 0:
                    battery_discharge += power
                else:
                    battery_charge += abs(power)
                battery_soc = getattr(asset, 'state_of_charge', 0.0)
            elif asset.asset_type.value == "load":
                total_demand += abs(power)  # Demand is typically negative power
        
        # Calculate grid balance
        total_generation = solar_output + wind_output + battery_discharge - battery_charge
        net_balance = total_generation - total_demand
        
        # Calculate curtailment and unmet demand
        curtailed_energy = max(0, net_balance) if net_balance > 0 else 0.0
        unmet_demand = max(0, -net_balance) if net_balance < 0 else 0.0
        
        # Store metrics
        metrics_data['timestamp'].append(current_time)
        metrics_data['solar_output_mw'].append(solar_output)
        metrics_data['wind_output_mw'].append(wind_output)
        metrics_data['battery_charge_mw'].append(battery_charge)
        metrics_data['battery_discharge_mw'].append(battery_discharge)
        metrics_data['battery_soc'].append(battery_soc)
        metrics_data['demand_mw'].append(total_demand)
        metrics_data['net_balance_mw'].append(net_balance)
        metrics_data['curtailed_energy_mw'].append(curtailed_energy)
        metrics_data['unmet_demand_mw'].append(unmet_demand)
        
        # Progress indicator
        if step % 20 == 0:
            progress = (step / total_steps) * 100
            print(f"  Progress: {progress:.1f}% - Time: {current_time.strftime('%Y-%m-%d %H:%M')} - "
                  f"Gen: {total_generation:.1f}MW, Demand: {total_demand:.1f}MW, "
                  f"Battery SoC: {battery_soc*100:.1f}%")
    
    return metrics_data


def analyze_simulation_results(metrics_data):
    """Analyze the simulation results and provide insights."""
    
    print("\n" + "="*60)
    print("SIMULATION RESULTS ANALYSIS")
    print("="*60)
    
    # Calculate summary statistics
    total_solar = sum(metrics_data['solar_output_mw']) * 0.25  # MWh (15min intervals)
    total_wind = sum(metrics_data['wind_output_mw']) * 0.25
    total_demand = sum(metrics_data['demand_mw']) * 0.25
    total_curtailed = sum(metrics_data['curtailed_energy_mw']) * 0.25
    total_unmet = sum(metrics_data['unmet_demand_mw']) * 0.25
    
    avg_battery_soc = sum(metrics_data['battery_soc']) / len(metrics_data['battery_soc']) * 100
    min_battery_soc = min(metrics_data['battery_soc']) * 100
    max_battery_soc = max(metrics_data['battery_soc']) * 100
    
    renewable_penetration = ((total_solar + total_wind) / total_demand) * 100 if total_demand > 0 else 0
    curtailment_rate = (total_curtailed / (total_solar + total_wind)) * 100 if (total_solar + total_wind) > 0 else 0
    demand_satisfaction = ((total_demand - total_unmet) / total_demand) * 100 if total_demand > 0 else 0
    
    print(f"Simulation Duration: {len(metrics_data['timestamp']) * 0.25:.1f} hours")
    print(f"Total Data Points: {len(metrics_data['timestamp'])}")
    print()
    print("ENERGY GENERATION:")
    print(f"  Solar Generation: {total_solar:.1f} MWh")
    print(f"  Wind Generation: {total_wind:.1f} MWh") 
    print(f"  Total Renewable: {total_solar + total_wind:.1f} MWh")
    print()
    print("DEMAND & BALANCE:")
    print(f"  Total Demand: {total_demand:.1f} MWh")
    print(f"  Renewable Penetration: {renewable_penetration:.1f}%")
    print(f"  Demand Satisfaction: {demand_satisfaction:.1f}%")
    print()
    print("CURTAILMENT & SHORTFALL:")
    print(f"  Curtailed Energy: {total_curtailed:.1f} MWh")
    print(f"  Curtailment Rate: {curtailment_rate:.1f}%")
    print(f"  Unmet Demand: {total_unmet:.1f} MWh")
    print()
    print("BATTERY PERFORMANCE:")
    print(f"  Average SoC: {avg_battery_soc:.1f}%")
    print(f"  SoC Range: {min_battery_soc:.1f}% - {max_battery_soc:.1f}%")
    
    # Peak analysis
    max_demand = max(metrics_data['demand_mw'])
    max_generation = max([s + w for s, w in zip(metrics_data['solar_output_mw'], metrics_data['wind_output_mw'])])
    max_solar = max(metrics_data['solar_output_mw'])
    max_wind = max(metrics_data['wind_output_mw'])
    
    print()
    print("PEAK VALUES:")
    print(f"  Peak Demand: {max_demand:.1f} MW")
    print(f"  Peak Generation: {max_generation:.1f} MW")
    print(f"  Peak Solar: {max_solar:.1f} MW")
    print(f"  Peak Wind: {max_wind:.1f} MW")


def show_visualization_structure(metrics_data):
    """Show what the visualization would look like with the generated data."""
    
    print("\n" + "="*60)
    print("VISUALIZATION STRUCTURE PREVIEW")
    print("="*60)
    
    print("\nWith visualization dependencies installed, you would see:")
    print()
    print("1. MAIN SIMULATION DASHBOARD (plot_simulation_metrics):")
    print("   ├── Subplot 1: Power Generation (Solar + Wind)")
    print("   ├── Subplot 2: Demand and Net Balance") 
    print("   ├── Subplot 3: Battery Operations (Charge/Discharge)")
    print("   ├── Subplot 4: Battery State of Charge (%)")
    print("   ├── Subplot 5: Curtailed Energy")
    print("   └── Subplot 6: Unmet Demand")
    print()
    print("2. POWER FLOW DASHBOARD (create_power_flow_dashboard):")
    print("   ├── Stacked Areas: Solar + Wind + Battery Discharge")
    print("   └── Demand Line: Total electrical demand")
    print()
    print("3. INTERACTIVE FEATURES:")
    print("   ├── Hover tooltips with precise values")
    print("   ├── Zoom and pan capabilities")
    print("   ├── Legend toggle for individual series")
    print("   ├── Synchronized time axes across subplots")
    print("   └── Reference lines (SoC limits, zero balance)")
    print()
    print("4. HTML REPORT (create_metrics_report):")
    print("   ├── Summary statistics table")
    print("   ├── Combined interactive dashboards") 
    print("   └── Publication-ready format")
    
    # Show sample data points
    print(f"\nSAMPLE DATA POINTS (first 5 of {len(metrics_data['timestamp'])}):")
    print("Time".ljust(20) + "Solar".ljust(8) + "Wind".ljust(8) + "Demand".ljust(8) + "SoC%".ljust(8))
    print("-" * 50)
    
    for i in range(min(5, len(metrics_data['timestamp']))):
        time_str = metrics_data['timestamp'][i].strftime("%m/%d %H:%M")
        solar = metrics_data['solar_output_mw'][i]
        wind = metrics_data['wind_output_mw'][i]
        demand = metrics_data['demand_mw'][i]
        soc = metrics_data['battery_soc'][i] * 100
        
        print(f"{time_str:<20}{solar:>6.1f}{wind:>8.1f}{demand:>8.1f}{soc:>7.1f}")
    
    print("\nTo install visualization dependencies and see interactive plots:")
    print("  poetry add pandas plotly numpy")
    print("  # or")
    print("  pip install pandas plotly numpy")
    print("\nThen run: python examples/visualization_demo.py")


def main():
    """Main demonstration function."""
    
    print("PSIREG Real Simulation Demo")
    print("="*60)
    print("Creating comprehensive renewable grid simulation...")
    
    # Create simulation
    engine, agents = create_renewable_grid_simulation()
    
    # Run simulation
    metrics_data = simulate_grid_operations(engine, agents, duration_hours=24)
    
    # Analyze results
    analyze_simulation_results(metrics_data)
    
    # Show visualization structure
    show_visualization_structure(metrics_data)
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("Real simulation data generated and ready for visualization!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 