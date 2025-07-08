#!/usr/bin/env python3
"""Comprehensive PSI Controller Results Visualization Suite.

This script creates detailed visualizations for PSI controller evaluation results,
suitable for research presentations and thesis documentation.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import seaborn as sns
from plotly.subplots import make_subplots

# Set style for matplotlib
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


class PSIVisualizationSuite:
    """Comprehensive visualization suite for PSI controller evaluation."""

    def __init__(self, data_file: str, output_dir: str = "visualizations"):
        """Initialize visualization suite.
        
        Args:
            data_file: Path to simulation results parquet file
            output_dir: Directory to save visualizations
        """
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        print(f"📊 Loading simulation data from: {data_file}")
        self.df = pd.read_parquet(data_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"✅ Loaded {len(self.df)} records with {self.df['controller'].nunique()} controllers")
        print(f"📅 Scenarios: {list(self.df['scenario'].unique())}")
        print(f"🎛️ Controllers: {list(self.df['controller'].unique())}")

    def create_performance_comparison_chart(self) -> str:
        """Create comprehensive performance comparison chart."""
        print("📈 Creating performance comparison chart...")
        
        # Calculate KPIs
        kpis = self._calculate_kpis()
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Fossil Backup Power (MW)', 'Grid Stability Score',
                'Frequency Deviation (Hz)', 'Voltage Deviation (V)',
                'Renewable Utilization (%)', 'Overall Performance Score'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        controllers = kpis['controller'].unique()
        
        # Get grouped data with consistent ordering
        fossil_backup_data = kpis.groupby('controller')['avg_fossil_backup_mw'].mean()
        
        # Fossil backup
        fig.add_trace(
            go.Bar(
                x=[CONTROLLER_NAMES[c] for c in controllers],
                y=[fossil_backup_data[c] for c in controllers],
                name='Fossil Backup',
                marker_color=[CONTROLLER_COLORS[c] for c in controllers]
            ),
            row=1, col=1
        )
        
        # Get all grouped data with consistent ordering
        grid_stability_data = kpis.groupby('controller')['grid_stability_score'].mean()
        frequency_deviation_data = kpis.groupby('controller')['avg_frequency_deviation_hz'].mean()
        voltage_deviation_data = kpis.groupby('controller')['avg_voltage_deviation_v'].mean()
        renewable_utilization_data = kpis.groupby('controller')['renewable_utilization_pct'].mean()
        
        # Grid stability
        fig.add_trace(
            go.Bar(
                x=[CONTROLLER_NAMES[c] for c in controllers],
                y=[grid_stability_data[c] for c in controllers],
                name='Grid Stability',
                marker_color=[CONTROLLER_COLORS[c] for c in controllers],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Frequency deviation
        fig.add_trace(
            go.Bar(
                x=[CONTROLLER_NAMES[c] for c in controllers],
                y=[frequency_deviation_data[c] for c in controllers],
                name='Frequency Dev',
                marker_color=[CONTROLLER_COLORS[c] for c in controllers],
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Voltage deviation
        fig.add_trace(
            go.Bar(
                x=[CONTROLLER_NAMES[c] for c in controllers],
                y=[voltage_deviation_data[c] for c in controllers],
                name='Voltage Dev',
                marker_color=[CONTROLLER_COLORS[c] for c in controllers],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Renewable utilization
        fig.add_trace(
            go.Bar(
                x=[CONTROLLER_NAMES[c] for c in controllers],
                y=[renewable_utilization_data[c] for c in controllers],
                name='Renewable Util',
                marker_color=[CONTROLLER_COLORS[c] for c in controllers],
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Overall performance score (composite)
        overall_scores = self._calculate_overall_scores(kpis)
        fig.add_trace(
            go.Bar(
                x=[CONTROLLER_NAMES[c] for c in controllers],
                y=[overall_scores[c] for c in controllers],
                name='Overall Score',
                marker_color=[CONTROLLER_COLORS[c] for c in controllers],
                showlegend=False
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="PSI Controller Performance Comparison Across Key Metrics",
            height=800,
            showlegend=False
        )
        
        # Save plot
        output_file = self.output_dir / "performance_comparison.html"
        fig.write_html(str(output_file))
        print(f"   ✅ Saved to: {output_file}")
        
        return str(output_file)

    def create_scenario_analysis_chart(self) -> str:
        """Create scenario-specific analysis chart."""
        print("🎬 Creating scenario analysis chart...")
        
        kpis = self._calculate_kpis()
        
        # Create subplot for each scenario
        scenarios = kpis['scenario'].unique()
        fig = make_subplots(
            rows=1, cols=len(scenarios),
            subplot_titles=[s.replace('_', ' ').title() for s in scenarios]
        )
        
        for i, scenario in enumerate(scenarios, 1):
            scenario_data = kpis[kpis['scenario'] == scenario]
            
            # Fossil backup for this scenario
            fig.add_trace(
                go.Bar(
                    x=[CONTROLLER_NAMES[c] for c in scenario_data['controller']],
                    y=scenario_data['avg_fossil_backup_mw'],
                    name=f'Fossil Backup - {scenario}',
                    marker_color=[CONTROLLER_COLORS[c] for c in scenario_data['controller']],
                    showlegend=(i == 1)
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            title="Fossil Backup Power by Scenario (Lower is Better)",
            height=500,
            yaxis_title="Fossil Backup (MW)"
        )
        
        output_file = self.output_dir / "scenario_analysis.html"
        fig.write_html(str(output_file))
        print(f"   ✅ Saved to: {output_file}")
        
        return str(output_file)

    def create_time_series_chart(self) -> str:
        """Create time series analysis chart."""
        print("📈 Creating time series analysis chart...")
        
        # Focus on routine_day for time series
        routine_data = self.df[self.df['scenario'] == 'routine_day'].copy()
        routine_data['hour'] = routine_data['timestamp'].dt.hour
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Renewable Generation', 'Fossil Backup',
                'Grid Frequency', 'Battery Operations'
            ]
        )
        
        for controller in routine_data['controller'].unique():
            controller_data = routine_data[routine_data['controller'] == controller]
            # Only calculate mean for numeric columns
            numeric_cols = ['solar_mw', 'wind_mw', 'battery_charge_mw', 'battery_discharge_mw', 
                           'battery_soc', 'demand_mw', 'curtailed_mw', 'fossil_backup_mw', 
                           'frequency_hz', 'voltage_v', 'net_balance_mw']
            hourly_avg = controller_data.groupby('hour')[numeric_cols].mean()
            
            # Renewable generation
            renewable_gen = hourly_avg['solar_mw'] + hourly_avg['wind_mw']
            fig.add_trace(
                go.Scatter(
                    x=hourly_avg.index,
                    y=renewable_gen,
                    mode='lines+markers',
                    name=f'{CONTROLLER_NAMES[controller]} - Renewable',
                    line_color=CONTROLLER_COLORS[controller]
                ),
                row=1, col=1
            )
            
            # Fossil backup
            fig.add_trace(
                go.Scatter(
                    x=hourly_avg.index,
                    y=hourly_avg['fossil_backup_mw'],
                    mode='lines+markers',
                    name=f'{CONTROLLER_NAMES[controller]} - Fossil',
                    line_color=CONTROLLER_COLORS[controller],
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Grid frequency
            fig.add_trace(
                go.Scatter(
                    x=hourly_avg.index,
                    y=hourly_avg['frequency_hz'],
                    mode='lines+markers',
                    name=f'{CONTROLLER_NAMES[controller]} - Frequency',
                    line_color=CONTROLLER_COLORS[controller],
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Battery SOC
            fig.add_trace(
                go.Scatter(
                    x=hourly_avg.index,
                    y=hourly_avg['battery_soc'] * 100,
                    mode='lines+markers',
                    name=f'{CONTROLLER_NAMES[controller]} - Battery SOC',
                    line_color=CONTROLLER_COLORS[controller],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Time Series Analysis - Routine Day Scenario",
            height=700
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Hour of Day")
        fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
        fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="SOC (%)", row=2, col=2)
        
        output_file = self.output_dir / "time_series_analysis.html"
        fig.write_html(str(output_file))
        print(f"   ✅ Saved to: {output_file}")
        
        return str(output_file)

    def create_improvement_heatmap(self) -> str:
        """Create PSI improvement heatmap."""
        print("🔥 Creating PSI improvement heatmap...")
        
        kpis = self._calculate_kpis()
        
        # Calculate PSI improvements vs each baseline
        metrics = ['avg_fossil_backup_mw', 'avg_frequency_deviation_hz', 
                  'avg_voltage_deviation_v', 'grid_stability_score']
        
        improvements = []
        
        for scenario in kpis['scenario'].unique():
            scenario_data = kpis[kpis['scenario'] == scenario]
            psi_data = scenario_data[scenario_data['controller'] == 'psi'].iloc[0]
            
            for baseline in ['rule', 'ml', 'swarm']:
                if baseline in scenario_data['controller'].values:
                    baseline_data = scenario_data[scenario_data['controller'] == baseline].iloc[0]
                    
                    for metric in metrics:
                        if metric == 'grid_stability_score':
                            # Higher is better
                            improvement = ((psi_data[metric] - baseline_data[metric]) / 
                                         baseline_data[metric] * 100)
                        else:
                            # Lower is better
                            improvement = ((baseline_data[metric] - psi_data[metric]) / 
                                         baseline_data[metric] * 100)
                        
                        improvements.append({
                            'scenario': scenario.replace('_', ' ').title(),
                            'baseline': CONTROLLER_NAMES[baseline],
                            'metric': metric.replace('_', ' ').title(),
                            'improvement_pct': improvement
                        })
        
        improvement_df = pd.DataFrame(improvements)
        
        # Create heatmap using plotly
        pivot_data = improvement_df.pivot_table(
            index=['baseline', 'metric'], 
            columns='scenario', 
            values='improvement_pct'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=[f"{row[0]} - {row[1]}" for row in pivot_data.index],
            colorscale='RdYlGn',
            text=np.round(pivot_data.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 10},
            colorbar=dict(title="Improvement (%)")
        ))
        
        fig.update_layout(
            title="PSI Controller Improvements vs Baselines (%)",
            xaxis_title="Scenario",
            yaxis_title="Baseline Controller - Metric",
            height=600
        )
        
        output_file = self.output_dir / "improvement_heatmap.html"
        fig.write_html(str(output_file))
        print(f"   ✅ Saved to: {output_file}")
        
        return str(output_file)

    def create_radar_chart(self) -> str:
        """Create radar chart comparing controllers."""
        print("🎯 Creating radar chart comparison...")
        
        kpis = self._calculate_kpis()
        # Only calculate mean for numeric columns
        numeric_cols = ['avg_fossil_backup_mw', 'avg_frequency_deviation_hz', 
                       'avg_voltage_deviation_v', 'renewable_utilization_pct', 'grid_stability_score']
        overall_kpis = kpis.groupby('controller')[numeric_cols].mean()
        
        # Normalize metrics for radar chart (0-100 scale)
        metrics = {
            'Efficiency': 100 - (overall_kpis['avg_fossil_backup_mw'] / 
                                overall_kpis['avg_fossil_backup_mw'].max() * 100),
            'Frequency Stability': 100 - (overall_kpis['avg_frequency_deviation_hz'] / 
                                         overall_kpis['avg_frequency_deviation_hz'].max() * 100),
            'Voltage Stability': 100 - (overall_kpis['avg_voltage_deviation_v'] / 
                                       overall_kpis['avg_voltage_deviation_v'].max() * 100),
            'Grid Stability': overall_kpis['grid_stability_score'],
            'Renewable Utilization': overall_kpis['renewable_utilization_pct']
        }
        
        fig = go.Figure()
        
        for controller in overall_kpis.index:
            values = [metrics[metric][controller] for metric in metrics.keys()]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=list(metrics.keys()) + [list(metrics.keys())[0]],
                fill='toself',
                name=CONTROLLER_NAMES[controller],
                line_color=CONTROLLER_COLORS[controller]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title="Controller Performance Radar Chart (0-100 Scale)",
            height=600
        )
        
        output_file = self.output_dir / "radar_comparison.html"
        fig.write_html(str(output_file))
        print(f"   ✅ Saved to: {output_file}")
        
        return str(output_file)

    def create_statistical_summary(self) -> str:
        """Create statistical summary charts."""
        print("📊 Creating statistical summary...")
        
        # Box plots for key metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Fossil Backup Distribution', 'Frequency Deviation Distribution',
                'Voltage Deviation Distribution', 'Grid Stability Distribution'
            ]
        )
        
        controllers = self.df['controller'].unique()
        
        # Fossil backup box plot
        for controller in controllers:
            controller_data = self.df[self.df['controller'] == controller]
            fig.add_trace(
                go.Box(
                    y=controller_data['fossil_backup_mw'],
                    name=CONTROLLER_NAMES[controller],
                    marker_color=CONTROLLER_COLORS[controller]
                ),
                row=1, col=1
            )
        
        # Frequency deviation box plot
        for controller in controllers:
            controller_data = self.df[self.df['controller'] == controller]
            if 'frequency_deviation_hz' in controller_data.columns:
                freq_dev = controller_data['frequency_deviation_hz']
            else:
                freq_dev = abs(controller_data['frequency_hz'] - 60.0)  # Use 60 Hz nominal
            fig.add_trace(
                go.Box(
                    y=freq_dev,
                    name=CONTROLLER_NAMES[controller],
                    marker_color=CONTROLLER_COLORS[controller],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Voltage deviation box plot
        for controller in controllers:
            controller_data = self.df[self.df['controller'] == controller]
            if 'voltage_deviation_v' in controller_data.columns:
                voltage_dev = controller_data['voltage_deviation_v']
            else:
                voltage_dev = abs(controller_data['voltage_v'] - 230.0)
            fig.add_trace(
                go.Box(
                    y=voltage_dev,
                    name=CONTROLLER_NAMES[controller],
                    marker_color=CONTROLLER_COLORS[controller],
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Grid stability (calculated per scenario)
        kpis = self._calculate_kpis()
        for controller in controllers:
            controller_kpis = kpis[kpis['controller'] == controller]
            fig.add_trace(
                go.Box(
                    y=controller_kpis['grid_stability_score'],
                    name=CONTROLLER_NAMES[controller],
                    marker_color=CONTROLLER_COLORS[controller],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Statistical Distribution Analysis",
            height=700
        )
        
        output_file = self.output_dir / "statistical_summary.html"
        fig.write_html(str(output_file))
        print(f"   ✅ Saved to: {output_file}")
        
        return str(output_file)

    def create_matplotlib_summary(self) -> str:
        """Create matplotlib summary for publication."""
        print("📄 Creating publication-ready matplotlib summary...")
        
        # Set up the figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PSI Controller Performance Evaluation Summary', fontsize=16, fontweight='bold')
        
        kpis = self._calculate_kpis()
        # Only calculate mean for numeric columns
        numeric_cols = ['avg_fossil_backup_mw', 'avg_frequency_deviation_hz', 
                       'avg_voltage_deviation_v', 'renewable_utilization_pct', 'grid_stability_score']
        overall_kpis = kpis.groupby('controller')[numeric_cols].mean()
        
        controllers = list(overall_kpis.index)
        controller_labels = [CONTROLLER_NAMES[c] for c in controllers]
        colors = [CONTROLLER_COLORS[c] for c in controllers]
        
        # 1. Fossil backup comparison
        axes[0, 0].bar(controller_labels, overall_kpis['avg_fossil_backup_mw'], color=colors)
        axes[0, 0].set_title('Average Fossil Backup Power')
        axes[0, 0].set_ylabel('Power (MW)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Grid stability scores
        axes[0, 1].bar(controller_labels, overall_kpis['grid_stability_score'], color=colors)
        axes[0, 1].set_title('Grid Stability Score')
        axes[0, 1].set_ylabel('Score (0-100)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Frequency deviation
        axes[0, 2].bar(controller_labels, overall_kpis['avg_frequency_deviation_hz'], color=colors)
        axes[0, 2].set_title('Average Frequency Deviation')
        axes[0, 2].set_ylabel('Deviation (Hz)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Scenario comparison (fossil backup)
        scenarios = kpis['scenario'].unique()
        x = np.arange(len(scenarios))
        width = 0.2
        
        for i, controller in enumerate(controllers):
            controller_data = kpis[kpis['controller'] == controller]
            values = [controller_data[controller_data['scenario'] == s]['avg_fossil_backup_mw'].iloc[0] 
                     for s in scenarios]
            axes[1, 0].bar(x + i * width, values, width, label=CONTROLLER_NAMES[controller], 
                          color=CONTROLLER_COLORS[controller])
        
        axes[1, 0].set_title('Fossil Backup by Scenario')
        axes[1, 0].set_ylabel('Power (MW)')
        axes[1, 0].set_xlabel('Scenario')
        axes[1, 0].set_xticks(x + width * 1.5)
        axes[1, 0].set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
        axes[1, 0].legend()
        
        # 5. PSI improvements
        psi_improvements = self._calculate_psi_improvements(kpis)
        baselines = list(psi_improvements.keys())
        improvements = list(psi_improvements.values())
        
        axes[1, 1].bar(baselines, improvements, color=['#ff7f0e', '#d62728', '#9467bd'])
        axes[1, 1].set_title('PSI Fossil Backup Reduction vs Baselines')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_xlabel('Baseline Controller')
        
        # Add improvement percentages on bars
        for i, v in enumerate(improvements):
            axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Time series (routine day)
        routine_data = self.df[self.df['scenario'] == 'routine_day'].copy()
        routine_data['hour'] = routine_data['timestamp'].dt.hour
        
        for controller in controllers:
            controller_data = routine_data[routine_data['controller'] == controller]
            hourly_avg = controller_data.groupby('hour')['fossil_backup_mw'].mean()
            axes[1, 2].plot(hourly_avg.index, hourly_avg.values, 
                           label=CONTROLLER_NAMES[controller], 
                           color=CONTROLLER_COLORS[controller], 
                           marker='o', markersize=4)
        
        axes[1, 2].set_title('Fossil Backup - Routine Day (Hourly)')
        axes[1, 2].set_ylabel('Power (MW)')
        axes[1, 2].set_xlabel('Hour of Day')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "psi_summary_matplotlib.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to: {output_file}")
        return str(output_file)

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
                
                # Calculate KPIs to match statistical analysis methodology
                total_curtailed_mwh = subset['curtailed_mw'].sum() * time_step_hours
                
                # FIX 1: Use average MW instead of total MWh for fossil backup 
                # to match statistical summary which shows fossil_backup_mw_mean
                avg_fossil_backup_mw = subset['fossil_backup_mw'].mean()
                
                # FIX 2: Use the actual frequency deviation values from data
                # Statistical summary uses 'frequency_deviation_hz' which are already deviations
                if 'frequency_deviation_hz' in subset.columns:
                    avg_freq_deviation_hz = subset['frequency_deviation_hz'].mean()
                else:
                    # Fallback: calculate deviation from nominal frequency (60 Hz for US grid)
                    avg_freq_deviation_hz = abs(subset['frequency_hz'] - 60.0).mean()
                
                # FIX 3: Use the actual voltage deviation values from data  
                # Statistical summary uses 'voltage_deviation_v' which are already deviations
                if 'voltage_deviation_v' in subset.columns:
                    avg_voltage_deviation_v = subset['voltage_deviation_v'].mean()
                else:
                    # Fallback: calculate deviation from nominal voltage (230V)
                    avg_voltage_deviation_v = abs(subset['voltage_v'] - 230.0).mean()
                
                # Renewable utilization
                total_renewable = subset['solar_mw'].sum() + subset['wind_mw'].sum()
                renewable_used = total_renewable - subset['curtailed_mw'].sum()
                renewable_utilization = (renewable_used / total_renewable * 100) if total_renewable > 0 else 0
                
                # Grid stability score - using the correct deviation values
                freq_stability_score = max(0, 100 - (avg_freq_deviation_hz / 0.1 * 100))
                voltage_stability_score = max(0, 100 - (avg_voltage_deviation_v / 10.0 * 100))
                grid_stability_score = (freq_stability_score + voltage_stability_score) / 2
                
                kpis.append({
                    'controller': controller,
                    'scenario': scenario,
                    'total_curtailed_mwh': total_curtailed_mwh,
                    'avg_fossil_backup_mw': avg_fossil_backup_mw,  # Changed from total_fossil_backup_mwh
                    'avg_frequency_deviation_hz': avg_freq_deviation_hz,
                    'avg_voltage_deviation_v': avg_voltage_deviation_v,
                    'renewable_utilization_pct': renewable_utilization,
                    'grid_stability_score': grid_stability_score
                })
        
        return pd.DataFrame(kpis)

    def _calculate_overall_scores(self, kpis: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall performance scores."""
        # Only calculate mean for numeric columns
        numeric_cols = ['avg_fossil_backup_mw', 'avg_frequency_deviation_hz', 
                       'avg_voltage_deviation_v', 'renewable_utilization_pct', 'grid_stability_score']
        overall_kpis = kpis.groupby('controller')[numeric_cols].mean()
        
        scores = {}
        for controller in overall_kpis.index:
            # Composite score (0-100)
            efficiency_score = 100 - (overall_kpis.loc[controller, 'avg_fossil_backup_mw'] / 
                                    overall_kpis['avg_fossil_backup_mw'].max() * 100)
            stability_score = overall_kpis.loc[controller, 'grid_stability_score']
            renewable_score = overall_kpis.loc[controller, 'renewable_utilization_pct']
            
            overall_score = (efficiency_score * 0.4 + stability_score * 0.4 + renewable_score * 0.2)
            scores[controller] = overall_score
        
        return scores

    def _calculate_psi_improvements(self, kpis: pd.DataFrame) -> Dict[str, float]:
        """Calculate PSI improvements over baselines."""
        # Only calculate mean for numeric columns
        numeric_cols = ['avg_fossil_backup_mw', 'avg_frequency_deviation_hz', 
                       'avg_voltage_deviation_v', 'renewable_utilization_pct', 'grid_stability_score']
        overall_kpis = kpis.groupby('controller')[numeric_cols].mean()
        
        psi_fossil = overall_kpis.loc['psi', 'avg_fossil_backup_mw']
        improvements = {}
        
        for baseline in ['rule', 'ml', 'swarm']:
            if baseline in overall_kpis.index:
                baseline_fossil = overall_kpis.loc[baseline, 'avg_fossil_backup_mw']
                improvement = ((baseline_fossil - psi_fossil) / baseline_fossil * 100)
                improvements[baseline.upper()] = improvement
        
        return improvements

    def generate_all_visualizations(self) -> List[str]:
        """Generate all visualizations."""
        print("🎨 Generating comprehensive visualization suite...")
        
        output_files = []
        
        try:
            output_files.append(self.create_performance_comparison_chart())
            output_files.append(self.create_scenario_analysis_chart())
            output_files.append(self.create_time_series_chart())
            output_files.append(self.create_improvement_heatmap())
            output_files.append(self.create_radar_chart())
            output_files.append(self.create_statistical_summary())
            output_files.append(self.create_matplotlib_summary())
            
            print(f"\n🎉 Successfully generated {len(output_files)} visualizations!")
            print(f"📁 All files saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        return output_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate PSI controller visualizations")
    parser.add_argument("data_file", help="Path to simulation results parquet file")
    parser.add_argument("--output-dir", default="visualizations", 
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data_file).exists():
        print(f"❌ Data file not found: {args.data_file}")
        print("💡 Run the PSI evaluation first: python run_psi_evaluation.py")
        sys.exit(1)
    
    # Create visualization suite
    viz_suite = PSIVisualizationSuite(args.data_file, args.output_dir)
    
    # Generate all visualizations
    output_files = viz_suite.generate_all_visualizations()
    
    print(f"\n📊 Visualization files created:")
    for file in output_files:
        print(f"   • {file}")
    
    print(f"\n🔗 Open the HTML files in your browser to view interactive charts!")


if __name__ == "__main__":
    main() 