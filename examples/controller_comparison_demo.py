#!/usr/bin/env python3
"""Demo script showing how to use the controller comparison visualization module."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from psireg.viz.viz_comparison import visualize_controller_comparison

def create_demo_data():
    """Create demonstration data showing PSI controller superiority."""
    np.random.seed(42)
    
    # Time range (24 hours, 15-minute intervals)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    time_points = [start_time + timedelta(minutes=15*i) for i in range(96)]
    
    controllers = ["rule", "ml", "swarm", "psi"]
    scenarios = ["routine_day", "storm_day", "demand_spike"]
    
    all_data = []
    
    for scenario in scenarios:
        for controller in controllers:
            for i, timestamp in enumerate(time_points):
                hour = timestamp.hour
                
                # Solar generation (peak at noon)
                solar_factor = max(0, np.sin(np.pi * (hour - 6) / 12))
                base_solar = 50 * solar_factor
                
                # Wind generation (more variable)
                base_wind = 30 + 20 * np.random.random()
                
                # Demand (varies by time of day)
                demand_factor = 0.7 + 0.3 * np.sin(np.pi * (hour - 6) / 12)
                base_demand = 80 * demand_factor
                if scenario == "demand_spike":
                    base_demand *= 1.3
                elif scenario == "storm_day":
                    base_demand *= 1.1
                
                # PSI controller performs better across all metrics
                efficiency_factor = 0.95 if controller == "psi" else np.random.uniform(0.8, 0.9)
                
                solar_mw = base_solar * efficiency_factor * np.random.uniform(0.95, 1.05)
                wind_mw = base_wind * efficiency_factor * np.random.uniform(0.95, 1.05)
                demand_mw = base_demand * np.random.uniform(0.95, 1.05)
                
                # Battery operations
                net_generation = solar_mw + wind_mw - demand_mw
                if net_generation > 0:
                    battery_charge_mw = min(20, net_generation * 0.5)
                    battery_discharge_mw = 0
                else:
                    battery_charge_mw = 0
                    battery_discharge_mw = min(25, abs(net_generation) * 0.6)
                
                # Battery SOC
                battery_soc = 0.5 + 0.3 * np.sin(np.pi * i / 48)
                
                # Net balance after battery
                net_balance_mw = solar_mw + wind_mw + battery_discharge_mw - battery_charge_mw - demand_mw
                
                # PSI controller reduces curtailment and fossil backup
                curtailment_factor = 0.03 if controller == "psi" else np.random.uniform(0.08, 0.12)
                fossil_factor = 0.4 if controller == "psi" else np.random.uniform(0.6, 0.8)
                
                curtailed_mw = max(0, net_balance_mw * curtailment_factor)
                fossil_backup_mw = max(0, -net_balance_mw * fossil_factor)
                
                # Grid stability (PSI controller better)
                freq_deviation = 0.02 if controller == "psi" else np.random.uniform(0.05, 0.1)
                voltage_deviation = 1.0 if controller == "psi" else np.random.uniform(3.0, 7.0)
                
                frequency_hz = 50.0 + np.random.normal(0, freq_deviation)
                voltage_v = 230.0 + np.random.normal(0, voltage_deviation)
                
                # Create record
                record = {
                    "timestamp": timestamp,
                    "controller": controller,
                    "scenario": scenario,
                    "solar_mw": solar_mw,
                    "wind_mw": wind_mw,
                    "battery_charge_mw": battery_charge_mw,
                    "battery_discharge_mw": battery_discharge_mw,
                    "battery_soc": battery_soc,
                    "demand_mw": demand_mw,
                    "curtailed_mw": curtailed_mw,
                    "fossil_backup_mw": fossil_backup_mw,
                    "frequency_hz": frequency_hz,
                    "voltage_v": voltage_v,
                    "net_balance_mw": net_balance_mw,
                }
                
                all_data.append(record)
    
    return pd.DataFrame(all_data)

def main():
    """Run the controller comparison demo."""
    print("🚀 PSI Controller Comparison Demo")
    print("=" * 40)
    
    # Create demo data
    print("📊 Creating demonstration data...")
    df = create_demo_data()
    
    # Save to output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save as Parquet (more efficient for large datasets)
    data_file = output_dir / "controller_comparison_demo.parquet"
    df.to_parquet(data_file, index=False)
    print(f"✅ Data saved to: {data_file}")
    
    # Create visualizations
    print("\n📈 Creating controller comparison visualizations...")
    print("This will demonstrate PSI controller superiority across all metrics:")
    print("  • Lower curtailment (better renewable utilization)")
    print("  • Reduced fossil backup (greener operation)")
    print("  • Better grid stability (frequency & voltage)")
    print("  • Superior demand satisfaction")
    
    # Run visualization
    visualize_controller_comparison(str(data_file), scenario="routine_day")
    
    print("\n🎉 Demo completed!")
    print("📋 The interactive HTML report shows:")
    print("  A. Time-series comparison of all controllers")
    print("  B. Supply vs demand analysis")
    print("  C. Performance metrics with % improvements")
    print("  D. Radar chart showing overall performance")
    print("  E. Statistical significance tests")
    
    print(f"\n💡 To run with your own data:")
    print(f"   python -m psireg.viz.viz_comparison 'your_data/*.parquet' --scenario your_scenario")

if __name__ == "__main__":
    main() 