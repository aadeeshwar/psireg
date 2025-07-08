"""Controller comparison visualization suite for PSIREG.

This module provides comprehensive visualizations to demonstrate PSI controller
superiority over baseline controllers (Rule-Based, ML-Only, Swarm-Only).
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Color palette for controller comparison
COLOR_MAP = {
    "rule": "#999999",  # Gray
    "ml": "#1f77b4",  # Blue
    "swarm": "#d62728",  # Red
    "psi": "#2ca02c",  # Green (highlight PSI)
}

# KPI definitions and preferences
KPI_DEFINITIONS = {
    "total_curtailed_mwh": {
        "name": "Total Curtailed Energy",
        "unit": "MWh",
        "description": "Wasted renewable energy",
        "better": "lower",
        "column": "curtailed_mw",
    },
    "total_fossil_backup_mwh": {
        "name": "Total Fossil Backup",
        "unit": "MWh",
        "description": "Fossil fuel reliance",
        "better": "lower",
        "column": "fossil_backup_mw",
    },
    "avg_frequency_deviation_hz": {
        "name": "Average Frequency Deviation",
        "unit": "Hz",
        "description": "Grid stability metric",
        "better": "lower",
        "column": "frequency_hz",
    },
    "avg_voltage_deviation_v": {
        "name": "Average Voltage Deviation",
        "unit": "V",
        "description": "Grid stability metric",
        "better": "lower",
        "column": "voltage_v",
    },
    "total_unmet_demand_mwh": {
        "name": "Total Unmet Demand",
        "unit": "MWh",
        "description": "Blackout events",
        "better": "lower",
        "column": "net_balance_mw",
    },
}


def load_simulation_data(file_pattern: str) -> pd.DataFrame:
    """Load simulation results from multiple files.

    Args:
        file_pattern: Glob pattern for result files (CSV/Feather/Parquet)

    Returns:
        Combined DataFrame with all simulation results
    """
    file_paths = glob.glob(file_pattern)
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {file_pattern}")

    dataframes = []
    for file_path in file_paths:
        try:
            if file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            elif file_path.endswith(".feather"):
                df = pd.read_feather(file_path)
            else:
                df = pd.read_csv(file_path)

            # Ensure timestamp is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            dataframes.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue

    if not dataframes:
        raise ValueError("No valid data files could be loaded")

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate KPIs per controller and scenario.

    Args:
        df: Raw simulation data

    Returns:
        DataFrame with KPIs per controller/scenario
    """
    # Detect time step (assuming uniform)
    if len(df) > 1:
        time_step_hours = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() / 3600
    else:
        time_step_hours = 0.25  # Default 15-minute intervals

    # Group by controller and scenario (if available)
    group_cols = ["controller"]
    if "scenario" in df.columns:
        group_cols.append("scenario")

    kpi_results = []

    for group_key, group_df in df.groupby(group_cols):
        if isinstance(group_key, str):
            controller = group_key
            scenario = "default"
        else:
            controller, scenario = group_key

        # Calculate KPIs
        kpis = {
            "controller": controller,
            "scenario": scenario,
        }

        # Total curtailed energy (MWh)
        if "curtailed_mw" in group_df.columns:
            kpis["total_curtailed_mwh"] = group_df["curtailed_mw"].sum() * time_step_hours
        else:
            kpis["total_curtailed_mwh"] = 0.0

        # Total fossil backup (MWh)
        if "fossil_backup_mw" in group_df.columns:
            kpis["total_fossil_backup_mwh"] = group_df["fossil_backup_mw"].sum() * time_step_hours
        else:
            kpis["total_fossil_backup_mwh"] = 0.0

        # Average frequency deviation
        if "frequency_hz" in group_df.columns:
            nominal_freq = 50.0  # Assuming 50Hz grid
            kpis["avg_frequency_deviation_hz"] = abs(group_df["frequency_hz"] - nominal_freq).mean()
        else:
            kpis["avg_frequency_deviation_hz"] = 0.0

        # Average voltage deviation
        if "voltage_v" in group_df.columns:
            nominal_voltage = 230.0  # Assuming 230V nominal
            kpis["avg_voltage_deviation_v"] = abs(group_df["voltage_v"] - nominal_voltage).mean()
        else:
            kpis["avg_voltage_deviation_v"] = 0.0

        # Total unmet demand (estimated from negative net balance)
        if "net_balance_mw" in group_df.columns:
            unmet_demand = group_df["net_balance_mw"].apply(lambda x: max(0, -x)).sum()
            kpis["total_unmet_demand_mwh"] = unmet_demand * time_step_hours
        else:
            kpis["total_unmet_demand_mwh"] = 0.0

        kpi_results.append(kpis)

    return pd.DataFrame(kpi_results)


def create_multi_line_timeseries(df: pd.DataFrame) -> go.Figure:
    """Create multi-line time-series visualization.

    Args:
        df: Raw simulation data

    Returns:
        Plotly figure with time-series subplots
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["Curtailed Energy (MW)", "Fossil Backup (MW)", "Net Balance (MW)"],
    )

    controllers = df["controller"].unique()

    for controller in controllers:
        controller_df = df[df["controller"] == controller]
        color = COLOR_MAP.get(controller, "#000000")

        # Curtailed energy
        if "curtailed_mw" in controller_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=controller_df["timestamp"],
                    y=controller_df["curtailed_mw"],
                    name=f"{controller.upper()} - Curtailed",
                    line={"color": color, "width": 2},
                    legendgroup=controller,
                ),
                row=1,
                col=1,
            )

        # Fossil backup
        if "fossil_backup_mw" in controller_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=controller_df["timestamp"],
                    y=controller_df["fossil_backup_mw"],
                    name=f"{controller.upper()} - Fossil",
                    line={"color": color, "width": 2},
                    legendgroup=controller,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Net balance
        if "net_balance_mw" in controller_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=controller_df["timestamp"],
                    y=controller_df["net_balance_mw"],
                    name=f"{controller.upper()} - Balance",
                    line={"color": color, "width": 2},
                    legendgroup=controller,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

    fig.update_layout(
        title="Controller Performance Comparison - Time Series",
        height=800,
        template="plotly_white",
        hovermode="x unified",
    )

    # Add zero line for net balance
    fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="gray")

    return fig


def create_stacked_area_chart(df: pd.DataFrame, scenario: str = "routine_day") -> go.Figure:
    """Create stacked area chart for supply vs demand.

    Args:
        df: Raw simulation data
        scenario: Scenario to visualize

    Returns:
        Plotly figure with stacked area chart
    """
    # Filter for specific scenario if available
    if "scenario" in df.columns:
        scenario_df = df[df["scenario"] == scenario]
    else:
        scenario_df = df

    fig = go.Figure()

    controllers = scenario_df["controller"].unique()

    for controller in controllers:
        controller_df = scenario_df[scenario_df["controller"] == controller]
        color = COLOR_MAP.get(controller, "#000000")

        # Calculate supply stack
        supply_stack = 0
        if "solar_mw" in controller_df.columns:
            supply_stack += controller_df["solar_mw"]
        if "wind_mw" in controller_df.columns:
            supply_stack += controller_df["wind_mw"]
        if "battery_discharge_mw" in controller_df.columns:
            supply_stack += controller_df["battery_discharge_mw"]

        # Add supply area
        fig.add_trace(
            go.Scatter(
                x=controller_df["timestamp"],
                y=supply_stack,
                name=f"{controller.upper()} - Supply",
                fill="tonexty" if controller != controllers[0] else "tozeroy",
                line={"color": color, "width": 0},
                fillcolor=color,
                opacity=0.7,
            )
        )

        # Add demand line
        if "demand_mw" in controller_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=controller_df["timestamp"],
                    y=controller_df["demand_mw"],
                    name=f"{controller.upper()} - Demand",
                    line={"color": color, "width": 3, "dash": "dash"},
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"Supply vs Demand Stack - {scenario.replace('_', ' ').title()}",
        xaxis_title="Time",
        yaxis_title="Power (MW)",
        height=600,
        template="plotly_white",
    )

    return fig


def create_grouped_bar_chart(kpi_df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart with improvement annotations.

    Args:
        kpi_df: KPI summary DataFrame

    Returns:
        Plotly figure with grouped bars
    """
    fig = go.Figure()

    controllers = kpi_df["controller"].unique()
    metrics = ["total_curtailed_mwh", "total_fossil_backup_mwh", "total_unmet_demand_mwh"]

    x_labels = []
    for metric in metrics:
        for controller in controllers:
            x_labels.append(
                f"{controller.upper()}<br>{metric.replace('_', ' ').replace('total ', '').replace('mwh', '').title()}"
            )

    y_values = []
    colors = []

    for metric in metrics:
        for controller in controllers:
            controller_data = kpi_df[kpi_df["controller"] == controller]
            if not controller_data.empty:
                value = controller_data[metric].iloc[0]
                y_values.append(value)
                colors.append(COLOR_MAP.get(controller, "#000000"))
            else:
                y_values.append(0)
                colors.append("#000000")

    fig.add_trace(
        go.Bar(x=x_labels, y=y_values, marker_color=colors, text=[f"{v:.1f}" for v in y_values], textposition="outside")
    )

    # Calculate and add improvement annotations
    if "psi" in controllers:
        psi_data = kpi_df[kpi_df["controller"] == "psi"]
        if not psi_data.empty:
            annotations = []
            for i, metric in enumerate(metrics):
                psi_value = psi_data[metric].iloc[0]
                for j, controller in enumerate(controllers):
                    if controller != "psi":
                        controller_data = kpi_df[kpi_df["controller"] == controller]
                        if not controller_data.empty:
                            baseline_value = controller_data[metric].iloc[0]
                            if baseline_value > 0:
                                improvement = ((baseline_value - psi_value) / baseline_value) * 100
                                x_pos = i * len(controllers) + j
                                annotations.append(
                                    {
                                        "x": x_pos,
                                        "y": y_values[x_pos] + max(y_values) * 0.05,
                                        "text": f"{improvement:.1f}%",
                                        "showarrow": False,
                                        "font": {"color": "green" if improvement > 0 else "red", "size": 12},
                                    }
                                )

            fig.update_layout(annotations=annotations)

    fig.update_layout(
        title="Controller Performance Comparison - Key Metrics",
        xaxis_title="Controller / Metric",
        yaxis_title="Value",
        height=600,
        template="plotly_white",
        showlegend=False,
    )

    return fig


def create_radar_chart(kpi_df: pd.DataFrame) -> go.Figure:
    """Create radar/spider chart for controller comparison.

    Args:
        kpi_df: KPI summary DataFrame

    Returns:
        Plotly figure with radar chart
    """
    fig = go.Figure()

    controllers = kpi_df["controller"].unique()
    metrics = list(KPI_DEFINITIONS.keys())

    # Normalize metrics to 0-1 scale (higher is better)
    normalized_data = {}
    for metric in metrics:
        if metric in kpi_df.columns:
            values = kpi_df[metric].values
            if KPI_DEFINITIONS[metric]["better"] == "lower":
                # For "lower is better" metrics, invert the scale
                max_val = max(values) if max(values) > 0 else 1
                normalized_data[metric] = [1 - (v / max_val) for v in values]
            else:
                # For "higher is better" metrics
                max_val = max(values) if max(values) > 0 else 1
                normalized_data[metric] = [v / max_val for v in values]
        else:
            normalized_data[metric] = [0.5] * len(controllers)

    # Create radar chart for each controller
    for i, controller in enumerate(controllers):
        values = []
        for metric in metrics:
            if metric in normalized_data:
                values.append(normalized_data[metric][i])
            else:
                values.append(0.5)

        # Close the radar chart
        values.append(values[0])

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=[KPI_DEFINITIONS[m]["name"] for m in metrics] + [KPI_DEFINITIONS[metrics[0]]["name"]],
                fill="toself",
                name=controller.upper(),
                line={"color": COLOR_MAP.get(controller, "#000000"), "width": 2},
                fillcolor=COLOR_MAP.get(controller, "#000000"),
                opacity=0.3,
            )
        )

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 1],
                "tickvals": [0, 0.25, 0.5, 0.75, 1],
                "ticktext": ["Poor", "Fair", "Good", "Very Good", "Excellent"],
            }
        },
        title="Controller Performance Radar Chart<br><sub>Higher values = Better performance</sub>",
        height=600,
        template="plotly_white",
    )

    return fig


def create_significance_table(df: pd.DataFrame, kpi_df: pd.DataFrame) -> str:
    """Create statistical significance table.

    Args:
        df: Raw simulation data
        kpi_df: KPI summary DataFrame

    Returns:
        HTML table with significance test results
    """
    if "psi" not in kpi_df["controller"].values:
        return "<p>PSI controller not found in data for significance testing.</p>"

    # Extract PSI data
    psi_data = {}
    baselines = [c for c in kpi_df["controller"].unique() if c != "psi"]

    # For each KPI, collect time-series data by controller
    for kpi_name, kpi_info in KPI_DEFINITIONS.items():
        if kpi_info["column"] in df.columns:
            psi_values = df[df["controller"] == "psi"][kpi_info["column"]].values
            psi_data[kpi_name] = psi_values

    # Perform t-tests
    table_rows = []
    table_rows.append("<table border='1' style='border-collapse: collapse; width: 100%;'>")
    table_rows.append(
        "<tr><th>KPI</th><th>Baseline</th><th>PSI Mean</th><th>Baseline Mean</th>"
        "<th>P-value</th><th>Significance</th></tr>"
    )

    for kpi_name, kpi_info in KPI_DEFINITIONS.items():
        if kpi_name in psi_data:
            psi_values = psi_data[kpi_name]

            for baseline in baselines:
                baseline_values = df[df["controller"] == baseline][kpi_info["column"]].values

                if len(baseline_values) > 0 and len(psi_values) > 0:
                    # Perform t-test
                    try:
                        t_stat, p_value = stats.ttest_ind(psi_values, baseline_values)

                        # Determine significance
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            significance = "ns"

                        table_rows.append(
                            f"<tr>"
                            f"<td>{kpi_info['name']}</td>"
                            f"<td>{baseline.upper()}</td>"
                            f"<td>{np.mean(psi_values):.3f}</td>"
                            f"<td>{np.mean(baseline_values):.3f}</td>"
                            f"<td>{p_value:.4f}</td>"
                            f"<td>{significance}</td>"
                            f"</tr>"
                        )
                    except Exception as e:
                        table_rows.append(
                            f"<tr>"
                            f"<td>{kpi_info['name']}</td>"
                            f"<td>{baseline.upper()}</td>"
                            f"<td colspan='4'>Error: {str(e)}</td>"
                            f"</tr>"
                        )

    table_rows.append("</table>")
    table_rows.append("<p><small>*** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant</small></p>")

    return "\n".join(table_rows)


def visualize_controller_comparison(results_path_pattern: str, scenario: str = "routine_day") -> None:
    """Main function to create controller comparison visualizations.

    Args:
        results_path_pattern: Glob pattern for result files
        scenario: Scenario to focus on for some visualizations
    """
    try:
        # Load data
        print(f"Loading data from: {results_path_pattern}")
        df = load_simulation_data(results_path_pattern)
        print(f"Loaded {len(df)} records with {len(df['controller'].unique())} controllers")

        # Compute KPIs
        print("Computing KPIs...")
        kpi_df = compute_kpis(df)
        print(f"Computed KPIs for {len(kpi_df)} controller/scenario combinations")

        # Create visualizations
        print("Creating visualizations...")

        # A. Multi-line time-series
        fig_timeseries = create_multi_line_timeseries(df)

        # B. Stacked area chart
        fig_stacked = create_stacked_area_chart(df, scenario)

        # C. Grouped bar chart
        fig_bars = create_grouped_bar_chart(kpi_df)

        # D. Radar chart
        fig_radar = create_radar_chart(kpi_df)

        # E. Significance table
        significance_html = create_significance_table(df, kpi_df)

        # Create combined HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PSI Controller Performance Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .viz-container {{ margin: 20px 0; }}
                .significance-table {{ margin: 20px 0; }}
                h1, h2 {{ color: #2ca02c; }}
                table {{ margin: 20px auto; }}
                th, td {{ padding: 8px; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>PSI Controller Performance Analysis</h1>
            <p>This report demonstrates the superior performance of the PSI controller
            compared to baseline controllers.</p>

            <div class="viz-container">
                <h2>A. Time-Series Comparison</h2>
                {fig_timeseries.to_html(include_plotlyjs='cdn', div_id='timeseries')}
            </div>

            <div class="viz-container">
                <h2>B. Supply vs Demand Analysis</h2>
                {fig_stacked.to_html(include_plotlyjs='cdn', div_id='stacked')}
            </div>

            <div class="viz-container">
                <h2>C. Performance Metrics Comparison</h2>
                {fig_bars.to_html(include_plotlyjs='cdn', div_id='bars')}
            </div>

            <div class="viz-container">
                <h2>D. Overall Performance Radar</h2>
                {fig_radar.to_html(include_plotlyjs='cdn', div_id='radar')}
            </div>

            <div class="significance-table">
                <h2>E. Statistical Significance Analysis</h2>
                {significance_html}
            </div>
        </body>
        </html>
        """

        # Save HTML report
        output_path = Path("psi_controller_analysis.html")
        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"Interactive report saved to: {output_path.absolute()}")

        # Try to open in browser
        try:
            import webbrowser

            webbrowser.open(f"file://{output_path.absolute()}")
            print("Opening report in browser...")
        except Exception:
            print("Could not open browser automatically. Please open the HTML file manually.")

    except Exception as e:
        print(f"Error: {e}")
        raise


def main() -> None:
    """Command-line interface for controller comparison visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize PSI controller performance vs baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python -m psireg.viz.viz_comparison "runs/*.parquet" --scenario storm_day
    python -m psireg.viz.viz_comparison "output/results_*.csv" --scenario routine_day
        """,
    )

    parser.add_argument("results_pattern", help="Glob pattern for result files (CSV/Feather/Parquet)")

    parser.add_argument(
        "--scenario", default="routine_day", help="Scenario to focus on for some visualizations (default: routine_day)"
    )

    args = parser.parse_args()

    visualize_controller_comparison(args.results_pattern, args.scenario)


if __name__ == "__main__":
    main()
