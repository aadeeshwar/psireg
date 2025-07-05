#!/usr/bin/env python3
"""Load/Demand Node Model Integration Demonstration.

This script demonstrates the comprehensive Load/Demand Node Model implementation
including stochastic and trace-driven demand profiles, time-of-use patterns,
demand response capabilities, and swarm intelligence coordination.

Features demonstrated:
- Load assets with different demand profiles
- Demand agents with swarm coordination
- Grid integration and real-time control
- Economic optimization and demand response
- Peak shaving and frequency support
"""

import random
import tempfile
import os
from datetime import datetime, timedelta
from typing import List

from psireg.sim.engine import GridEngine, NetworkNode, TransmissionLine
from psireg.config.schema import SimulationConfig, GridConfig
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.assets.wind import WindTurbine
from psireg.sim.assets.battery import Battery
from psireg.swarm.agents.demand_agent import DemandAgent
from psireg.swarm.agents.battery_agent import BatteryAgent
from psireg.utils.enums import AssetStatus, WeatherCondition


def create_sample_trace_file() -> str:
    """Create a sample trace file for trace-driven demand profile.
    
    Returns:
        Path to temporary CSV file with demand trace data
    """
    # Create temporary CSV file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    
    # Write header
    temp_file.write("timestamp,demand_mw\n")
    
    # Generate 24-hour demand profile with realistic patterns
    base_time = datetime(2023, 7, 15, 0, 0, 0)  # Summer day
    for hour in range(24):
        timestamp = base_time + timedelta(hours=hour)
        
        # Realistic demand curve: low at night, peak during day
        if 0 <= hour <= 6:  # Night
            demand = 45.0 + random.gauss(0, 3)
        elif 7 <= hour <= 9:  # Morning ramp
            demand = 55.0 + 10 * (hour - 6) / 3 + random.gauss(0, 4)
        elif 10 <= hour <= 16:  # Day peak
            demand = 80.0 + random.gauss(0, 5)
        elif 17 <= hour <= 20:  # Evening peak
            demand = 90.0 + random.gauss(0, 6)
        else:  # Evening decline
            demand = 70.0 - 10 * (hour - 20) / 3 + random.gauss(0, 4)
        
        # Ensure reasonable bounds
        demand = max(30.0, min(100.0, demand))
        
        temp_file.write(f"{timestamp.isoformat()},{demand:.2f}\n")
    
    temp_file.close()
    return temp_file.name


def demonstrate_load_profiles():
    """Demonstrate different load profile types."""
    print("=" * 80)
    print("LOAD PROFILE DEMONSTRATION")
    print("=" * 80)
    
    # 1. Stochastic Load Profile
    print("\n1. Stochastic Load Profile:")
    stochastic_load = Load(
        asset_id="stochastic_load",
        name="Residential Complex",
        node_id="residential_node",
        capacity_mw=150.0,
        baseline_demand_mw=100.0,
        peak_demand_mw=140.0,
        off_peak_demand_mw=70.0,
        peak_hours_start=8,
        peak_hours_end=20,
        demand_volatility=0.15,
        profile_type="stochastic",
    )
    stochastic_load.set_status(AssetStatus.ONLINE)
    
    # Generate 6-hour profile
    profile = stochastic_load.generate_stochastic_profile(hours=6, timestep_minutes=60)
    print(f"   6-hour demand profile: {[f'{d:.1f}' for d in profile]} MW")
    print(f"   Average demand: {sum(profile)/len(profile):.1f} MW")
    print(f"   Peak demand: {max(profile):.1f} MW")
    print(f"   Min demand: {min(profile):.1f} MW")
    
    # 2. Time-of-Use Load Profile
    print("\n2. Time-of-Use Load Profile:")
    tou_load = Load(
        asset_id="tou_load",
        name="Commercial Building",
        node_id="commercial_node",
        capacity_mw=200.0,
        baseline_demand_mw=120.0,
        peak_demand_mw=180.0,
        off_peak_demand_mw=80.0,
        peak_hours_start=9,
        peak_hours_end=18,
        profile_type="time_of_use",
        seasonal_factor_summer=1.3,  # High AC load
    )
    tou_load.set_status(AssetStatus.ONLINE)
    
    # Test different times
    test_times = [
        (datetime(2023, 7, 15, 2, 0), "Night"),
        (datetime(2023, 7, 15, 10, 0), "Peak"),
        (datetime(2023, 7, 15, 14, 0), "Peak"),
        (datetime(2023, 7, 15, 22, 0), "Off-peak"),
    ]
    
    for test_time, period in test_times:
        tou_load.set_current_time(test_time)
        demand = tou_load.calculate_time_of_use_demand()
        print(f"   {period} ({test_time.hour:02d}:00): {demand:.1f} MW")
    
    # 3. Trace-Driven Load Profile
    print("\n3. Trace-Driven Load Profile:")
    trace_file = create_sample_trace_file()
    
    try:
        trace_load = Load(
            asset_id="trace_load",
            name="Industrial Plant",
            node_id="industrial_node",
            capacity_mw=300.0,
            baseline_demand_mw=200.0,
            profile_type="trace_driven",
            trace_file_path=trace_file,
        )
        trace_load.set_status(AssetStatus.ONLINE)
        trace_load.load_trace_data()
        
        print(f"   Loaded {len(trace_load.trace_data)} data points from trace file")
        
        # Test interpolation at different times
        test_times = [
            datetime(2023, 7, 15, 0, 30),  # Between data points
            datetime(2023, 7, 15, 6, 0),   # Exact data point
            datetime(2023, 7, 15, 12, 45), # Between data points
            datetime(2023, 7, 15, 18, 0),  # Exact data point
        ]
        
        for test_time in test_times:
            trace_load.set_current_time(test_time)
            demand = trace_load.calculate_demand_from_trace()
            print(f"   {test_time.strftime('%H:%M')}: {demand:.1f} MW (interpolated)")
    
    finally:
        os.unlink(trace_file)  # Clean up temporary file


def demonstrate_demand_response():
    """Demonstrate demand response capabilities."""
    print("\n" + "=" * 80)
    print("DEMAND RESPONSE DEMONSTRATION")
    print("=" * 80)
    
    # Create load with demand response capability
    dr_load = Load(
        asset_id="dr_load",
        name="Smart Building Complex",
        node_id="smart_node",
        capacity_mw=250.0,
        baseline_demand_mw=180.0,
        dr_capability_mw=50.0,
        price_elasticity=-0.3,
        baseline_price=60.0,
    )
    dr_load.set_status(AssetStatus.ONLINE)
    
    # Test price response
    print("\n1. Price Elasticity Response:")
    test_prices = [30.0, 60.0, 90.0, 120.0]  # Low to high prices
    
    for price in test_prices:
        responsive_demand = dr_load.calculate_price_response_demand(price)
        price_change = (price - dr_load.baseline_price) / dr_load.baseline_price * 100
        demand_change = (responsive_demand - dr_load.baseline_demand_mw) / dr_load.baseline_demand_mw * 100
        print(f"   Price: ${price:.0f}/MWh ({price_change:+.0f}%) → Demand: {responsive_demand:.1f} MW ({demand_change:+.1f}%)")
    
    # Test demand response signals
    print("\n2. Demand Response Signals:")
    dr_signals = [-30.0, -15.0, 0.0, 10.0, 25.0]  # MW signals
    
    for signal in dr_signals:
        dr_load.set_demand_response_signal(signal)
        response = dr_load.calculate_demand_response()
        final_demand = dr_load.calculate_final_demand()
        print(f"   DR Signal: {signal:+.0f} MW → Response: {response:+.1f} MW → Final: {final_demand:.1f} MW")


def demonstrate_swarm_coordination():
    """Demonstrate swarm intelligence coordination between demand agents."""
    print("\n" + "=" * 80)
    print("SWARM COORDINATION DEMONSTRATION")
    print("=" * 80)
    
    # Create multiple loads with different characteristics
    loads = []
    agents = []
    
    load_configs = [
        {
            "id": "residential_001",
            "name": "Residential District A",
            "baseline": 120.0,
            "dr_capability": 30.0,
            "price_elasticity": -0.25,
            "coordination_weight": 0.3,
        },
        {
            "id": "commercial_001", 
            "name": "Commercial Center B",
            "baseline": 200.0,
            "dr_capability": 60.0,
            "price_elasticity": -0.15,
            "coordination_weight": 0.4,
        },
        {
            "id": "industrial_001",
            "name": "Manufacturing Plant C",
            "baseline": 350.0,
            "dr_capability": 80.0,
            "price_elasticity": -0.35,
            "coordination_weight": 0.2,
        },
    ]
    
    # Create loads and agents
    for i, config in enumerate(load_configs):
        load = Load(
            asset_id=config["id"],
            name=config["name"],
            node_id=f"node_{i+1}",
            capacity_mw=config["baseline"] * 1.5,
            baseline_demand_mw=config["baseline"],
            dr_capability_mw=config["dr_capability"],
            price_elasticity=config["price_elasticity"],
        )
        load.set_status(AssetStatus.ONLINE)
        loads.append(load)
        
        agent = DemandAgent(
            load=load,
            coordination_weight=config["coordination_weight"],
        )
        agents.append(agent)
    
    print(f"\nCreated {len(agents)} demand agents:")
    for i, agent in enumerate(agents):
        load = agent.load
        print(f"   {load.name}: {load.baseline_demand_mw:.0f} MW baseline, "
              f"{load.dr_capability_mw:.0f} MW DR capability")
    
    # Simulate grid stress event
    print("\n1. Grid Stress Event Simulation:")
    print("   Simulating high frequency deviation (grid overload)...")
    
    for agent in agents:
        agent.update_grid_conditions(
            frequency_hz=59.85,  # Low frequency indicates generation shortage
            voltage_kv=230.0,
            local_load_mw=500.0,
            local_generation_mw=450.0,  # Generation deficit
            electricity_price=100.0,   # High price
        )
    
    # Calculate coordination signals
    for i, agent in enumerate(agents):
        neighbor_signals = [
            agents[j].get_coordination_signal() 
            for j in range(len(agents)) 
            if j != i
        ]
        agent.update_swarm_signals(neighbor_signals)
    
    # Calculate optimal demands
    print("\n   Agent responses to grid stress:")
    total_reduction = 0.0
    
    for agent in agents:
        load = agent.load
        forecast_prices = [100.0, 95.0, 90.0, 85.0]  # Declining prices
        forecast_generation = [450.0, 460.0, 470.0, 480.0]  # Improving generation
        forecast_grid_stress = [0.8, 0.6, 0.4, 0.2]  # Declining stress
        
        optimal_demand = agent.calculate_optimal_demand(
            forecast_prices=forecast_prices,
            forecast_generation=forecast_generation,
            forecast_grid_stress=forecast_grid_stress,
        )
        
        baseline = load.baseline_demand_mw
        reduction = baseline - optimal_demand
        total_reduction += reduction
        
        print(f"   {load.name}:")
        print(f"     Baseline: {baseline:.0f} MW → Optimal: {optimal_demand:.0f} MW")
        print(f"     Reduction: {reduction:.0f} MW ({reduction/baseline*100:.1f}%)")
        
        # Execute control action
        agent.execute_control_action(optimal_demand)
    
    print(f"\n   Total demand reduction: {total_reduction:.0f} MW")
    print(f"   Coordination effectiveness: {total_reduction/(sum(load.dr_capability_mw for load in loads))*100:.1f}% of total DR capability")
    
    # Demonstrate load shifting
    print("\n2. Load Shifting Coordination:")
    peak_hour = 18  # 6 PM peak
    off_peak_hour = 23  # 11 PM off-peak
    
    for agent in agents:
        load = agent.load
        shift_amount = load.dr_capability_mw * 0.4  # Shift 40% of DR capability
        
        success = agent.schedule_demand_shift(
            shift_mw=shift_amount,
            from_hour=peak_hour,
            to_hour=off_peak_hour,
            duration_hours=2,
        )
        
        if success:
            print(f"   {load.name}: Shifted {shift_amount:.0f} MW from {peak_hour}:00 to {off_peak_hour}:00")
        else:
            print(f"   {load.name}: Load shift failed (constraints violated)")


def demonstrate_grid_integration():
    """Demonstrate load integration with full grid simulation."""
    print("\n" + "=" * 80)
    print("GRID INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create grid engine
    engine = GridEngine(
        simulation_config=SimulationConfig(timestep_minutes=15),
        grid_config=GridConfig(frequency_hz=60.0, max_power_mw=2000.0),
    )
    
    # Create network topology
    nodes = [
        NetworkNode(node_id="gen_node", name="Generation Hub", voltage_kv=500.0),
        NetworkNode(node_id="load_node_1", name="Urban Center", voltage_kv=230.0),
        NetworkNode(node_id="load_node_2", name="Industrial Zone", voltage_kv=230.0),
        NetworkNode(node_id="storage_node", name="Storage Hub", voltage_kv=230.0),
    ]
    
    for node in nodes:
        engine.add_node(node)
    
    # Add transmission lines
    lines = [
        TransmissionLine(
            line_id="gen_to_urban",
            name="Gen-Urban Line",
            from_node="gen_node",
            to_node="load_node_1",
            capacity_mw=800.0,
            length_km=50.0,
            resistance=0.02,
        ),
        TransmissionLine(
            line_id="gen_to_industrial",
            name="Gen-Industrial Line", 
            from_node="gen_node",
            to_node="load_node_2",
            capacity_mw=600.0,
            length_km=75.0,
            resistance=0.03,
        ),
        TransmissionLine(
            line_id="gen_to_storage",
            name="Gen-Storage Line",
            from_node="gen_node", 
            to_node="storage_node",
            capacity_mw=400.0,
            length_km=25.0,
            resistance=0.01,
        ),
    ]
    
    for line in lines:
        engine.add_transmission_line(line)
    
    # Add generation assets
    solar_farm = SolarPanel(
        asset_id="solar_farm_1",
        name="Solar Farm Alpha",
        node_id="gen_node",
        capacity_mw=300.0,
        panel_efficiency=0.22,
        panel_area_m2=136364.0,  # ~300 MW at 1000 W/m2
        current_irradiance_w_m2=800.0,  # Good sunlight
        current_temperature_c=25.0,
    )
    solar_farm.set_status(AssetStatus.ONLINE)
    engine.add_asset(solar_farm)
    
    wind_farm = WindTurbine(
        asset_id="wind_farm_1",
        name="Wind Farm Beta",
        node_id="gen_node",
        capacity_mw=400.0,
        rotor_diameter_m=120.0,
        hub_height_m=100.0,
        current_wind_speed_ms=8.0,  # Good wind
    )
    wind_farm.set_status(AssetStatus.ONLINE)
    engine.add_asset(wind_farm)
    
    # Add storage
    battery_storage = Battery(
        asset_id="grid_battery",
        name="Grid-Scale Battery",
        node_id="storage_node",
        capacity_mw=200.0,
        energy_capacity_mwh=800.0,
        initial_soc_percent=60.0,
    )
    battery_storage.set_status(AssetStatus.ONLINE)
    engine.add_asset(battery_storage)
    
    # Add loads with different profiles
    urban_load = Load(
        asset_id="urban_load",
        name="Urban Residential Load",
        node_id="load_node_1",
        capacity_mw=400.0,
        baseline_demand_mw=320.0,
        peak_demand_mw=380.0,
        off_peak_demand_mw=250.0,
        dr_capability_mw=60.0,
        price_elasticity=-0.2,
        profile_type="stochastic",
        demand_volatility=0.1,
    )
    urban_load.set_status(AssetStatus.ONLINE)
    engine.add_asset(urban_load)
    
    industrial_load = Load(
        asset_id="industrial_load", 
        name="Industrial Manufacturing Load",
        node_id="load_node_2",
        capacity_mw=500.0,
        baseline_demand_mw=450.0,
        peak_demand_mw=480.0,
        off_peak_demand_mw=400.0,
        dr_capability_mw=80.0,
        price_elasticity=-0.3,
        profile_type="time_of_use",
        peak_hours_start=6,
        peak_hours_end=22,
    )
    industrial_load.set_status(AssetStatus.ONLINE)
    engine.add_asset(industrial_load)
    
    # Create demand agents
    urban_agent = DemandAgent(load=urban_load, coordination_weight=0.3)
    industrial_agent = DemandAgent(load=industrial_load, coordination_weight=0.4)
    
    # Create battery agent
    battery_agent = BatteryAgent(battery=battery_storage, coordination_weight=0.3)
    
    print(f"\nCreated grid with {len(engine.assets)} assets:")
    print(f"   Generation: {solar_farm.capacity_mw + wind_farm.capacity_mw:.0f} MW")
    print(f"   Load: {urban_load.capacity_mw + industrial_load.capacity_mw:.0f} MW")
    print(f"   Storage: {battery_storage.capacity_mw:.0f} MW")
    
    # Run simulation for 2 hours
    print("\n3. Grid Simulation (2 hours, 15-minute intervals):")
    
    simulation_data = []
    
    for step in range(8):  # 8 * 15 minutes = 2 hours
        # Update environmental conditions
        if step >= 4:  # Second hour: reduce solar, increase load
            solar_farm.set_irradiance(400.0)  # Clouds
            urban_load.current_demand_mw = 340.0  # Evening peak
            industrial_load.current_demand_mw = 470.0
        
        # Update agents with grid conditions
        state = engine.get_state()
        
        for agent in [urban_agent, industrial_agent]:
            agent.update_grid_conditions(
                frequency_hz=state.frequency_hz,
                voltage_kv=230.0,
                local_load_mw=state.total_load_mw,
                local_generation_mw=state.total_generation_mw,
                electricity_price=80.0 + step * 5.0,  # Rising prices
            )
        
        battery_agent.update_grid_conditions(
            frequency_hz=state.frequency_hz,
            voltage_kv=230.0,
            local_load_mw=state.total_load_mw,
            local_generation_mw=state.total_generation_mw,
            electricity_price=80.0 + step * 5.0,
        )
        
        # Agent coordination
        neighbor_signals = [urban_agent.get_coordination_signal(), industrial_agent.get_coordination_signal()]
        urban_agent.update_swarm_signals([neighbor_signals[1]])
        industrial_agent.update_swarm_signals([neighbor_signals[0]])
        
        # Calculate optimal setpoints
        forecast_prices = [85.0 + step * 5.0] * 4
        forecast_generation = [650.0] * 4
        forecast_grid_stress = [0.3] * 4
        
        # Demand agent actions
        urban_optimal = urban_agent.calculate_optimal_demand(
            forecast_prices=forecast_prices,
            forecast_generation=forecast_generation,
            forecast_grid_stress=forecast_grid_stress,
        )
        urban_agent.execute_control_action(urban_optimal)
        
        industrial_optimal = industrial_agent.calculate_optimal_demand(
            forecast_prices=forecast_prices,
            forecast_generation=forecast_generation,
            forecast_grid_stress=forecast_grid_stress,
        )
        industrial_agent.execute_control_action(industrial_optimal)
        
        # Battery agent actions
        battery_optimal = battery_agent.calculate_optimal_power(
            forecast_load=[state.total_load_mw] * 4,
            forecast_generation=[state.total_generation_mw] * 4,
            forecast_prices=forecast_prices,
        )
        battery_agent.execute_control_action(battery_optimal)
        
        # Step simulation
        engine.step(timedelta(minutes=15))
        
        # Record state
        state = engine.get_state()
        simulation_data.append({
            'step': step,
            'time': f"{step * 15:02d}min",
            'generation': state.total_generation_mw,
            'load': state.total_load_mw,
            'storage': state.total_storage_mw,
            'frequency': state.frequency_hz,
            'balance': state.power_balance_mw,
            'battery_soc': battery_storage.current_soc_percent,
        })
        
        print(f"   Step {step+1} ({step*15:02d}min): "
              f"Gen={state.total_generation_mw:.0f}MW, "
              f"Load={state.total_load_mw:.0f}MW, "
              f"Storage={state.total_storage_mw:+.0f}MW, "
              f"Freq={state.frequency_hz:.2f}Hz, "
              f"SOC={battery_storage.current_soc_percent:.1f}%")
    
    # Summary
    print(f"\n4. Simulation Summary:")
    final_state = simulation_data[-1]
    initial_state = simulation_data[0]
    
    print(f"   Initial frequency: {initial_state['frequency']:.3f} Hz")
    print(f"   Final frequency: {final_state['frequency']:.3f} Hz")
    print(f"   Frequency stability: {abs(final_state['frequency'] - 60.0):.3f} Hz deviation")
    
    print(f"   Initial battery SoC: {initial_state['battery_soc']:.1f}%")
    print(f"   Final battery SoC: {final_state['battery_soc']:.1f}%")
    
    avg_balance = sum(abs(data['balance']) for data in simulation_data) / len(simulation_data)
    print(f"   Average power balance: {avg_balance:.1f} MW")
    
    total_load_response = sum(abs(agent.load.dr_signal_mw) for agent in [urban_agent, industrial_agent])
    print(f"   Total demand response: {total_load_response:.1f} MW")


def main():
    """Run the comprehensive load/demand demonstration."""
    print("PSIREG Load/Demand Node Model Demonstration")
    print("Comprehensive Consumer Nodes with Swarm Intelligence")
    print(f"Simulation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Run demonstrations
    demonstrate_load_profiles()
    demonstrate_demand_response()
    demonstrate_swarm_coordination()
    demonstrate_grid_integration()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey achievements demonstrated:")
    print("✓ Stochastic and trace-driven demand profiles")
    print("✓ Time-of-use patterns with seasonal variations")
    print("✓ Price elasticity and demand response signals")
    print("✓ Swarm intelligence coordination between consumer nodes")
    print("✓ Grid frequency support through demand modulation")
    print("✓ Economic optimization and peak shaving")
    print("✓ Full integration with GridEngine simulation")
    print("✓ Multi-objective optimization (frequency, economic, coordination, comfort)")
    print("✓ Load scheduling and shifting capabilities")
    print("\nPrimary output achieved: Consumer nodes with intelligent swarm coordination")


if __name__ == "__main__":
    main() 