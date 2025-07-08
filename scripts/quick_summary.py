#!/usr/bin/env python3
"""Quick Summary Visualization for PSI Controller Results.

This script provides a quick overview of PSI controller performance
with essential charts and statistics.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Color scheme for controllers
CONTROLLER_COLORS = {
    "rule": "#999999",  # Gray
    "ml": "#1f77b4",    # Blue  
    "swarm": "#d62728",  # Red
    "psi": "#2ca02c",   # Green (highlight PSI)
}

CONTROLLER_NAMES = {
    "rule": "Rule-based",
    "ml": "ML-only", 
    "swarm": "Swarm-only",
    "psi": "PSI (Proposed)"
}


class QuickSummary:
    """Quick summary visualization for PSI controller results."""

    def __init__(self, data_file: str):
        """Initialize quick summary.
        
        Args:
            data_file: Path to simulation results parquet file
        """
        self.data_file = data_file
        
        # Load data
        print(f"📊 Loading data from: {data_file}")
        self.df = pd.read_parquet(data_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Calculate derived metrics
        self.df['renewable_generation_mw'] = self.df['solar_mw'] + self.df['wind_mw']
        self.df['frequency_deviation_hz'] = abs(self.df['frequency_hz'] - 60.0)
        self.df['voltage_deviation_v'] = abs(self.df['voltage_v'] - 230.0)
        
        print(f"✅ Loaded {len(self.df)} records")
        print(f"Controllers: {list(self.df['controller'].unique())}")
        print(f"Scenarios: {list(self.df['scenario'].unique())}")

    def create_summary_dashboard(self) -> str:
        """Create a comprehensive summary dashboard."""
        print("📈 Creating summary dashboard...")
        
        # Set up the figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('PSI Controller Performance Summary Dashboard', fontsize=16, fontweight='bold')
        
        # Calculate KPIs
        kpis = self._calculate_kpis()
        
        # 1. Fossil backup comparison
        fossil_data = kpis.groupby('controller')['total_fossil_backup_mwh'].mean()
        controllers = fossil_data.index
        colors = [CONTROLLER_COLORS[c] for c in controllers]
        
        bars = axes[0, 0].bar([CONTROLLER_NAMES[c] for c in controllers], fossil_data.values, color=colors)
        axes[0, 0].set_title('Average Fossil Backup Energy', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Energy (MWh)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, fossil_data.values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                          f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Grid stability scores
        stability_data = kpis.groupby('controller')['grid_stability_score'].mean()
        bars = axes[0, 1].bar([CONTROLLER_NAMES[c] for c in controllers], stability_data.values, color=colors)
        axes[0, 1].set_title('Grid Stability Score', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Score (0-100)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, stability_data.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Renewable utilization
        renewable_data = kpis.groupby('controller')['renewable_utilization_pct'].mean()
        bars = axes[0, 2].bar([CONTROLLER_NAMES[c] for c in controllers], renewable_data.values, color=colors)
        axes[0, 2].set_title('Renewable Utilization Rate', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('Utilization (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, renewable_data.values):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Scenario comparison (fossil backup)
        scenarios = kpis['scenario'].unique()
        x = np.arange(len(scenarios))
        width = 0.2
        
        for i, controller in enumerate(controllers):
            controller_data = kpis[kpis['controller'] == controller]
            values = [controller_data[controller_data['scenario'] == s]['total_fossil_backup_mwh'].iloc[0] 
                     for s in scenarios]
            bars = axes[1, 0].bar(x + i * width, values, width, label=CONTROLLER_NAMES[controller], 
                                color=CONTROLLER_COLORS[controller])
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                              f'{value:.0f}', ha='center', va='bottom', fontsize=8)
        
        axes[1, 0].set_title('Fossil Backup by Scenario', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Energy (MWh)')
        axes[1, 0].set_xlabel('Scenario')
        axes[1, 0].set_xticks(x + width * 1.5)
        axes[1, 0].set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
        axes[1, 0].legend()
        
        # 5. PSI improvements
        psi_improvements = self._calculate_psi_improvements(kpis)
        if psi_improvements:
            baselines = list(psi_improvements.keys())
            improvements = list(psi_improvements.values())
            
            bars = axes[1, 1].bar(baselines, improvements, color=['#ff7f0e', '#d62728', '#9467bd'])
            axes[1, 1].set_title('PSI Improvements vs Baselines', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Improvement (%)')
            axes[1, 1].set_xlabel('Baseline Controller')
            
            # Add improvement percentages on bars
            for bar, value in zip(bars, improvements):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                              f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Performance radar chart
        metrics = ['Efficiency', 'Stability', 'Renewable Use', 'Frequency Control', 'Voltage Control']
        
        # Normalize metrics for radar chart
        max_fossil = fossil_data.max()
        max_freq_dev = kpis.groupby('controller')['avg_frequency_deviation_hz'].mean().max()
        max_voltage_dev = kpis.groupby('controller')['avg_voltage_deviation_v'].mean().max()
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax_radar = plt.subplot(2, 3, 6, projection='polar')
        
        for controller in controllers:
            # Only calculate mean for numeric columns
            numeric_cols = ['total_fossil_backup_mwh', 'avg_frequency_deviation_hz', 
                           'avg_voltage_deviation_v', 'renewable_utilization_pct', 'grid_stability_score']
            controller_kpis = kpis[kpis['controller'] == controller][numeric_cols].mean()
            
            # Calculate normalized scores (0-100)
            efficiency = 100 * (1 - controller_kpis['total_fossil_backup_mwh'] / max_fossil)
            stability = controller_kpis['grid_stability_score']
            renewable = controller_kpis['renewable_utilization_pct']
            frequency = 100 * (1 - controller_kpis['avg_frequency_deviation_hz'] / max_freq_dev)
            voltage = 100 * (1 - controller_kpis['avg_voltage_deviation_v'] / max_voltage_dev)
            
            values = [efficiency, stability, renewable, frequency, voltage]
            values += values[:1]  # Complete the circle
            
            ax_radar.plot(angles, values, linewidth=2, linestyle='solid', 
                         label=CONTROLLER_NAMES[controller], color=CONTROLLER_COLORS[controller])
            ax_radar.fill(angles, values, alpha=0.25, color=CONTROLLER_COLORS[controller])
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_ylim(0, 100)
        ax_radar.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        
        # Save the dashboard
        output_file = "psi_summary_dashboard.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Dashboard saved to: {output_file}")
        return output_file

    def print_key_statistics(self):
        """Print key statistics to console."""
        print("\n📊 KEY PERFORMANCE STATISTICS")
        print("=" * 50)
        
        kpis = self._calculate_kpis()
        
        # Overall performance
        print("\n🏆 OVERALL PERFORMANCE (Average across all scenarios)")
        print("-" * 30)
        
        fossil_data = kpis.groupby('controller')['total_fossil_backup_mwh'].mean()
        stability_data = kpis.groupby('controller')['grid_stability_score'].mean()
        renewable_data = kpis.groupby('controller')['renewable_utilization_pct'].mean()
        
        for controller in fossil_data.index:
            print(f"{CONTROLLER_NAMES[controller]:>12}: "
                  f"Fossil: {fossil_data[controller]:6.1f} MWh  |  "
                  f"Stability: {stability_data[controller]:5.1f}/100  |  "
                  f"Renewable: {renewable_data[controller]:5.1f}%")
        
        # PSI improvements
        print("\n🚀 PSI IMPROVEMENTS vs BASELINES")
        print("-" * 30)
        
        psi_improvements = self._calculate_psi_improvements(kpis)
        if psi_improvements:
            for baseline, improvement in psi_improvements.items():
                print(f"vs {baseline:>6}: {improvement:+6.1f}% fossil backup reduction")
        
        # Scenario performance
        print("\n🎬 SCENARIO-SPECIFIC PERFORMANCE")
        print("-" * 30)
        
        for scenario in kpis['scenario'].unique():
            scenario_data = kpis[kpis['scenario'] == scenario]
            print(f"\n{scenario.replace('_', ' ').title()}:")
            
            scenario_fossil = scenario_data.groupby('controller')['total_fossil_backup_mwh'].mean()
            best_controller = scenario_fossil.idxmin()
            best_value = scenario_fossil.min()
            
            print(f"  Best performer: {CONTROLLER_NAMES[best_controller]} ({best_value:.1f} MWh)")
            
            if 'psi' in scenario_fossil.index:
                psi_value = scenario_fossil['psi']
                for controller in scenario_fossil.index:
                    if controller != 'psi':
                        improvement = ((scenario_fossil[controller] - psi_value) / scenario_fossil[controller]) * 100
                        print(f"  PSI vs {controller.upper()}: {improvement:+.1f}% improvement")

    def _calculate_kpis(self) -> pd.DataFrame:
        """Calculate key performance indicators."""
        kpis = []
        time_step_hours = 15 / 60  # 15 minutes
        
        for controller in self.df['controller'].unique():
            for scenario in self.df['scenario'].unique():
                subset = self.df[(self.df['controller'] == controller) & 
                               (self.df['scenario'] == scenario)]
                
                if len(subset) == 0:
                    continue
                
                # Calculate KPIs
                total_fossil_backup_mwh = subset['fossil_backup_mw'].sum() * time_step_hours
                avg_freq_deviation_hz = subset['frequency_deviation_hz'].mean()
                avg_voltage_deviation_v = subset['voltage_deviation_v'].mean()
                
                # Renewable utilization
                total_renewable = subset['renewable_generation_mw'].sum()
                renewable_used = total_renewable - subset['curtailed_mw'].sum()
                renewable_utilization = (renewable_used / total_renewable * 100) if total_renewable > 0 else 0
                
                # Grid stability score
                freq_stability_score = max(0, 100 - (avg_freq_deviation_hz / 0.1 * 100))
                voltage_stability_score = max(0, 100 - (avg_voltage_deviation_v / 10.0 * 100))
                grid_stability_score = (freq_stability_score + voltage_stability_score) / 2
                
                kpis.append({
                    'controller': controller,
                    'scenario': scenario,
                    'total_fossil_backup_mwh': total_fossil_backup_mwh,
                    'avg_frequency_deviation_hz': avg_freq_deviation_hz,
                    'avg_voltage_deviation_v': avg_voltage_deviation_v,
                    'renewable_utilization_pct': renewable_utilization,
                    'grid_stability_score': grid_stability_score
                })
        
        return pd.DataFrame(kpis)

    def _calculate_psi_improvements(self, kpis: pd.DataFrame) -> Dict[str, float]:
        """Calculate PSI improvements over baselines."""
        # Only calculate mean for numeric columns
        numeric_cols = ['total_fossil_backup_mwh', 'avg_frequency_deviation_hz', 
                       'avg_voltage_deviation_v', 'renewable_utilization_pct', 'grid_stability_score']
        overall_kpis = kpis.groupby('controller')[numeric_cols].mean()
        
        if 'psi' not in overall_kpis.index:
            return {}
        
        psi_fossil = overall_kpis.loc['psi', 'total_fossil_backup_mwh']
        improvements = {}
        
        for baseline in ['rule', 'ml', 'swarm']:
            if baseline in overall_kpis.index:
                baseline_fossil = overall_kpis.loc[baseline, 'total_fossil_backup_mwh']
                improvement = ((baseline_fossil - psi_fossil) / baseline_fossil * 100)
                improvements[baseline.upper()] = improvement
        
        return improvements

    def create_quick_summary(self) -> str:
        """Create complete quick summary."""
        print("⚡ Creating quick summary...")
        
        # Create dashboard
        dashboard_file = self.create_summary_dashboard()
        
        # Print key statistics
        self.print_key_statistics()
        
        return dashboard_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate quick PSI controller summary")
    parser.add_argument("data_file", help="Path to simulation results parquet file")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data_file).exists():
        print(f"❌ Data file not found: {args.data_file}")
        print("💡 Run the PSI evaluation first: python run_psi_evaluation.py")
        sys.exit(1)
    
    # Create quick summary
    summary = QuickSummary(args.data_file)
    dashboard_file = summary.create_quick_summary()
    
    print(f"\n✅ Quick summary complete!")
    print(f"📊 Dashboard saved to: {dashboard_file}")
    print(f"🔍 Open the image file to view the summary dashboard")


if __name__ == "__main__":
    main() 