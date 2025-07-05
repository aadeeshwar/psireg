#!/usr/bin/env python3
"""Demonstration of PSIREG visualization capabilities.

This script shows how to use the visualization module to create
interactive plots and dashboards for simulation metrics analysis.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from psireg.viz import plot_simulation_metrics, create_power_flow_dashboard, create_metrics_report
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required visualization dependencies:")
    print("  poetry install --with viz")
    print("  or")
    print("  pip install pandas plotly numpy")
    sys.exit(1)


def generate_sample_data(duration_hours: int = 24, timestep_minutes: int = 15) -> pd.DataFrame:
    """Generate realistic sample simulation data for demonstration.
    
    Args:
        duration_hours: Duration of simulation in hours
        timestep_minutes: Time step interval in minutes
        
    Returns:
        DataFrame with sample metrics data
    """
    # Calculate number of time steps
    num_steps = (duration_hours * 60) // timestep_minutes
    
    # Generate time series
    start_time = datetime(2023, 6, 15, 0, 0)  # Summer day for good solar
    timestamps = [start_time + timedelta(minutes=i * timestep_minutes) for i in range(num_steps)]
    
    # Generate realistic patterns
    hours = np.array([t.hour + t.minute/60 for t in timestamps])
    
    # Solar generation (bell curve, peak at noon)
    solar_base = 150  # MW peak capacity
    solar_output = np.maximum(0, solar_base * np.exp(-0.5 * ((hours - 12) / 4) ** 2))
    # Add some variability for clouds
    solar_noise = np.random.normal(0, 0.1, num_steps)
    solar_output = np.maximum(0, solar_output * (1 + solar_noise))
    
    # Wind generation (more variable, higher at night)
    wind_base = 100  # MW capacity
    wind_pattern = 0.4 + 0.3 * np.sin(2 * np.pi * hours / 24) + 0.3 * np.sin(4 * np.pi * hours / 24)
    wind_noise = np.random.normal(0, 0.2, num_steps)
    wind_output = np.maximum(0, wind_base * wind_pattern * (1 + wind_noise))
    
    # Demand pattern (higher during day, peak in evening)
    demand_base = 200  # MW base load
    demand_daily = 0.7 + 0.2 * np.sin(2 * np.pi * (hours - 6) / 24) + 0.1 * np.sin(4 * np.pi * hours / 24)
    demand_noise = np.random.normal(0, 0.05, num_steps)
    demand = demand_base * demand_daily * (1 + demand_noise)
    
    # Battery operations (charge when excess, discharge when deficit)
    total_generation = solar_output + wind_output
    net_balance = total_generation - demand
    
    # Simple battery strategy
    battery_capacity = 200  # MWh
    battery_power_limit = 50  # MW
    
    battery_charge = np.zeros(num_steps)
    battery_discharge = np.zeros(num_steps)
    battery_energy = np.zeros(num_steps)  # MWh stored
    battery_soc = np.zeros(num_steps)
    
    # Initialize battery at 50% SoC
    battery_energy[0] = battery_capacity * 0.5
    battery_soc[0] = 0.5
    
    for i in range(1, num_steps):
        dt_hours = timestep_minutes / 60
        
        if net_balance[i] > 0 and battery_soc[i-1] < 0.95:
            # Excess generation, charge battery
            charge_power = min(net_balance[i], battery_power_limit)
            charge_energy = charge_power * dt_hours
            available_capacity = battery_capacity * (0.95 - battery_soc[i-1])
            charge_energy = min(charge_energy, available_capacity)
            
            battery_charge[i] = charge_energy / dt_hours
            battery_energy[i] = battery_energy[i-1] + charge_energy
            
        elif net_balance[i] < 0 and battery_soc[i-1] > 0.05:
            # Deficit, discharge battery
            discharge_power = min(-net_balance[i], battery_power_limit)
            discharge_energy = discharge_power * dt_hours
            available_energy = battery_capacity * (battery_soc[i-1] - 0.05)
            discharge_energy = min(discharge_energy, available_energy)
            
            battery_discharge[i] = discharge_energy / dt_hours
            battery_energy[i] = battery_energy[i-1] - discharge_energy
            
        else:
            # No battery operation
            battery_energy[i] = battery_energy[i-1]
        
        battery_soc[i] = battery_energy[i] / battery_capacity
    
    # Recalculate net balance with battery
    net_balance_with_battery = total_generation + battery_discharge - battery_charge - demand
    
    # Calculate curtailed energy and unmet demand
    curtailed_energy = np.maximum(0, net_balance_with_battery)
    unmet_demand = np.maximum(0, -net_balance_with_battery)
    
    # Create DataFrame
    data = {
        'timestamp': timestamps,
        'solar_output_mw': solar_output,
        'wind_output_mw': wind_output,
        'battery_charge_mw': battery_charge,
        'battery_discharge_mw': battery_discharge,
        'battery_soc': battery_soc,
        'demand_mw': demand,
        'net_balance_mw': net_balance_with_battery,
        'curtailed_energy_mw': curtailed_energy,
        'unmet_demand_mw': unmet_demand
    }
    
    return pd.DataFrame(data)


def main():
    """Run the visualization demonstration."""
    print("PSIREG Visualization Demo")
    print("=" * 40)
    
    # Generate sample data
    print("Generating sample simulation data...")
    df = generate_sample_data(duration_hours=48, timestep_minutes=15)  # 2 days of data
    print(f"Generated {len(df)} data points over {len(df) * 0.25:.1f} hours")
    
    # Create main dashboard
    print("\nCreating main simulation dashboard...")
    main_fig = plot_simulation_metrics(
        df, 
        title="PSIREG Demo: 48-Hour Renewable Grid Simulation",
        height=1400
    )
    
    # Create power flow chart
    print("Creating power flow analysis...")
    flow_fig = create_power_flow_dashboard(
        df,
        title="Power Supply vs Demand Analysis"
    )
    
    # Generate comprehensive report
    print("Generating comprehensive report...")
    dashboard, flow_chart, stats = create_metrics_report(
        df,
        save_html="psireg_demo_report.html"
    )
    
    # Display summary statistics
    print("\nSimulation Summary:")
    print("-" * 30)
    print(f"Duration: {stats['simulation_duration_hours']:.1f} hours")
    print(f"Total Solar Generation: {stats['total_solar_generation_mwh']:.1f} MWh")
    print(f"Total Wind Generation: {stats['total_wind_generation_mwh']:.1f} MWh")
    print(f"Total Demand: {stats['total_demand_mwh']:.1f} MWh")
    print(f"Renewable Penetration: {stats['renewable_penetration_pct']:.1f}%")
    print(f"Curtailment Rate: {stats['curtailment_rate_pct']:.1f}%")
    print(f"Demand Satisfaction: {stats['demand_satisfaction_rate_pct']:.1f}%")
    print(f"Average Battery SoC: {stats['avg_battery_soc']:.1f}%")
    print(f"Battery SoC Range: {stats['min_battery_soc']:.1f}% - {stats['max_battery_soc']:.1f}%")
    
    # Show interactive plots
    print(f"\nDisplaying interactive visualizations...")
    print("Note: Plots will open in your default web browser")
    
    try:
        # Show the main dashboard
        main_fig.show()
        
        # Show the power flow chart
        flow_fig.show()
        
        print(f"\nHTML report saved to: psireg_demo_report.html")
        print("You can open this file in any web browser for a complete report.")
        
    except Exception as e:
        print(f"Error displaying plots: {e}")
        print("Saving plots as HTML files instead...")
        
        main_fig.write_html("psireg_main_dashboard.html")
        flow_fig.write_html("psireg_power_flow.html")
        
        print("Files saved:")
        print("  - psireg_main_dashboard.html")
        print("  - psireg_power_flow.html")
        print("  - psireg_demo_report.html")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 