"""Interactive visualization functions for PSIREG simulation metrics.

This module provides comprehensive plotting capabilities for analyzing
renewable energy grid simulation results using Plotly.
"""

from typing import Any

try:
    import pandas as pd  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # type: ignore[import-not-found]
    from plotly.subplots import make_subplots  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "Visualization dependencies not available. " "Install with: poetry install or pip install pandas plotly"
    ) from e


def plot_simulation_metrics(
    metrics_df: pd.DataFrame,
    title: str = "PSIREG Simulation Metrics",
    height: int = 1200,
    show_grid: bool = True,
    theme: str = "plotly_white",
) -> go.Figure:
    """Create comprehensive time-series visualizations of simulation metrics.

    Args:
        metrics_df: DataFrame with simulation metrics containing columns:
            - timestamp: datetime
            - solar_output_mw: Solar power generation
            - wind_output_mw: Wind power generation
            - battery_charge_mw: Battery charging power
            - battery_discharge_mw: Battery discharging power
            - battery_soc: Battery state of charge (0.0-1.0)
            - demand_mw: Electrical demand
            - net_balance_mw: Generation minus demand
            - curtailed_energy_mw: Curtailed renewable energy
            - unmet_demand_mw: Unmet electrical demand
        title: Main title for the dashboard
        height: Total height of the figure in pixels
        show_grid: Whether to show grid lines
        theme: Plotly theme to use

    Returns:
        Interactive Plotly figure with multiple subplots
    """
    # Validate required columns
    required_cols = [
        "timestamp",
        "solar_output_mw",
        "wind_output_mw",
        "battery_charge_mw",
        "battery_discharge_mw",
        "battery_soc",
        "demand_mw",
        "net_balance_mw",
        "curtailed_energy_mw",
        "unmet_demand_mw",
    ]

    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if metrics_df.empty:
        raise ValueError("Empty DataFrame provided")

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Power Generation (MW)",
            "Demand and Net Balance (MW)",
            "Battery Operations (MW)",
            "Battery State of Charge (%)",
            "Curtailed Energy (MW)",
            "Unmet Demand (MW)",
        ],
        specs=[[{"secondary_y": False}] for _ in range(6)],
    )

    # Color scheme
    colors = {
        "solar": "#FFA500",  # Orange
        "wind": "#1E90FF",  # DodgerBlue
        "demand": "#DC143C",  # Crimson
        "charge": "#32CD32",  # LimeGreen
        "discharge": "#FF6347",  # Tomato
        "soc": "#9370DB",  # MediumPurple
        "balance": "#2E8B57",  # SeaGreen
        "curtailed": "#FF69B4",  # HotPink
        "unmet": "#B22222",  # FireBrick
    }

    # 1. Power Generation
    fig.add_trace(
        go.Scatter(
            x=metrics_df["timestamp"],
            y=metrics_df["solar_output_mw"],
            name="Solar Output",
            line={"color": colors["solar"], "width": 2},
            hovertemplate="<b>Solar</b><br>%{y:.1f} MW<br>%{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=metrics_df["timestamp"],
            y=metrics_df["wind_output_mw"],
            name="Wind Output",
            line={"color": colors["wind"], "width": 2},
            hovertemplate="<b>Wind</b><br>%{y:.1f} MW<br>%{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. Demand and Net Balance
    fig.add_trace(
        go.Scatter(
            x=metrics_df["timestamp"],
            y=metrics_df["demand_mw"],
            name="Demand",
            line={"color": colors["demand"], "width": 2},
            hovertemplate="<b>Demand</b><br>%{y:.1f} MW<br>%{x}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=metrics_df["timestamp"],
            y=metrics_df["net_balance_mw"],
            name="Net Balance",
            line={"color": colors["balance"], "width": 2},
            hovertemplate="<b>Net Balance</b><br>%{y:.1f} MW<br>%{x}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add zero line for net balance
    fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray", opacity=0.5)

    # 3. Battery Operations
    fig.add_trace(
        go.Scatter(
            x=metrics_df["timestamp"],
            y=metrics_df["battery_charge_mw"],
            name="Battery Charge",
            line={"color": colors["charge"], "width": 2},
            hovertemplate="<b>Charging</b><br>%{y:.1f} MW<br>%{x}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=metrics_df["timestamp"],
            y=metrics_df["battery_discharge_mw"],
            name="Battery Discharge",
            line={"color": colors["discharge"], "width": 2},
            hovertemplate="<b>Discharging</b><br>%{y:.1f} MW<br>%{x}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # 4. Battery State of Charge
    fig.add_trace(
        go.Scatter(
            x=metrics_df["timestamp"],
            y=metrics_df["battery_soc"] * 100,  # Convert to percentage
            name="Battery SoC",
            line={"color": colors["soc"], "width": 3},
            fill="tonexty" if len(fig.data) == 0 else None,
            hovertemplate="<b>SoC</b><br>%{y:.1f}%<br>%{x}<extra></extra>",
        ),
        row=4,
        col=1,
    )

    # Add SoC reference lines
    for level, name in [(20, "Low"), (80, "High")]:
        fig.add_hline(
            y=level,
            row=4,
            col=1,
            line_dash="dot",
            line_color="red" if level == 20 else "orange",
            opacity=0.7,
            annotation_text=f"{name} SoC",
        )

    # 5. Curtailed Energy
    fig.add_trace(
        go.Scatter(
            x=metrics_df["timestamp"],
            y=metrics_df["curtailed_energy_mw"],
            name="Curtailed Energy",
            line={"color": colors["curtailed"], "width": 2},
            fill="tozeroy",
            hovertemplate="<b>Curtailed</b><br>%{y:.1f} MW<br>%{x}<extra></extra>",
        ),
        row=5,
        col=1,
    )

    # 6. Unmet Demand
    fig.add_trace(
        go.Scatter(
            x=metrics_df["timestamp"],
            y=metrics_df["unmet_demand_mw"],
            name="Unmet Demand",
            line={"color": colors["unmet"], "width": 2},
            fill="tozeroy",
            hovertemplate="<b>Unmet Demand</b><br>%{y:.1f} MW<br>%{x}<extra></extra>",
        ),
        row=6,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 20, "family": "Arial"}},
        height=height,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
        template=theme,
        hovermode="x unified",
    )

    # Update x-axes
    fig.update_xaxes(title_text="Time", row=6, col=1, showgrid=show_grid, gridwidth=1, gridcolor="lightgray")

    # Update y-axes with appropriate titles
    y_titles = ["Power (MW)", "Power (MW)", "Power (MW)", "SoC (%)", "Power (MW)", "Power (MW)"]

    for i, title in enumerate(y_titles, 1):
        fig.update_yaxes(title_text=title, row=i, col=1, showgrid=show_grid, gridwidth=1, gridcolor="lightgray")

    return fig


def create_power_flow_dashboard(
    metrics_df: pd.DataFrame, title: str = "Power Flow Analysis", height: int = 800
) -> go.Figure:
    """Create a stacked area chart showing power supply vs demand.

    Args:
        metrics_df: DataFrame with simulation metrics
        title: Title for the chart
        height: Height of the figure in pixels

    Returns:
        Interactive Plotly figure with stacked areas
    """
    if metrics_df.empty:
        raise ValueError("Empty DataFrame provided")

    # Calculate total supply components
    df = metrics_df.copy()
    df["total_generation"] = df["solar_output_mw"] + df["wind_output_mw"]
    df["net_battery"] = df["battery_discharge_mw"] - df["battery_charge_mw"]
    df["total_supply"] = df["total_generation"] + df["net_battery"].clip(lower=0)

    # Create figure
    fig = go.Figure()

    # Supply stack
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["solar_output_mw"],
            stackgroup="supply",
            name="Solar",
            line={"width": 0},
            fillcolor="rgba(255, 165, 0, 0.8)",
            hovertemplate="<b>Solar</b><br>%{y:.1f} MW<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["wind_output_mw"],
            stackgroup="supply",
            name="Wind",
            line={"width": 0},
            fillcolor="rgba(30, 144, 255, 0.8)",
            hovertemplate="<b>Wind</b><br>%{y:.1f} MW<extra></extra>",
        )
    )

    # Battery discharge (positive contribution to supply)
    battery_discharge_positive = df["net_battery"].clip(lower=0)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=battery_discharge_positive,
            stackgroup="supply",
            name="Battery Discharge",
            line={"width": 0},
            fillcolor="rgba(255, 99, 71, 0.8)",
            hovertemplate="<b>Battery Discharge</b><br>%{y:.1f} MW<extra></extra>",
        )
    )

    # Demand line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["demand_mw"],
            name="Demand",
            line={"color": "crimson", "width": 3},
            hovertemplate="<b>Demand</b><br>%{y:.1f} MW<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 18}},
        xaxis_title="Time",
        yaxis_title="Power (MW)",
        height=height,
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
    )

    return fig


def generate_summary_stats(metrics_df: pd.DataFrame) -> dict[str, Any]:
    """Generate summary statistics for the simulation metrics.

    Args:
        metrics_df: DataFrame with simulation metrics

    Returns:
        Dictionary with key simulation statistics
    """
    if metrics_df.empty:
        return {}

    stats = {
        "simulation_duration_hours": len(metrics_df) * 0.25,  # Assuming 15-min intervals
        "total_solar_generation_mwh": metrics_df["solar_output_mw"].sum() * 0.25,
        "total_wind_generation_mwh": metrics_df["wind_output_mw"].sum() * 0.25,
        "total_demand_mwh": metrics_df["demand_mw"].sum() * 0.25,
        "total_curtailed_mwh": metrics_df["curtailed_energy_mw"].sum() * 0.25,
        "total_unmet_demand_mwh": metrics_df["unmet_demand_mw"].sum() * 0.25,
        "avg_battery_soc": metrics_df["battery_soc"].mean() * 100,
        "min_battery_soc": metrics_df["battery_soc"].min() * 100,
        "max_battery_soc": metrics_df["battery_soc"].max() * 100,
        "avg_net_balance_mw": metrics_df["net_balance_mw"].mean(),
        "renewable_penetration_pct": (
            (metrics_df["solar_output_mw"].sum() + metrics_df["wind_output_mw"].sum())
            / metrics_df["demand_mw"].sum()
            * 100
        ),
        "curtailment_rate_pct": (
            metrics_df["curtailed_energy_mw"].sum()
            / (metrics_df["solar_output_mw"].sum() + metrics_df["wind_output_mw"].sum())
            * 100
        ),
        "demand_satisfaction_rate_pct": (1 - metrics_df["unmet_demand_mw"].sum() / metrics_df["demand_mw"].sum()) * 100,
    }

    return stats


def create_metrics_report(
    metrics_df: pd.DataFrame, save_html: str | None = None
) -> tuple[go.Figure, go.Figure, dict[str, Any]]:
    """Create a comprehensive metrics report with multiple visualizations.

    Args:
        metrics_df: DataFrame with simulation metrics
        save_html: Optional path to save HTML report

    Returns:
        Tuple of (main_dashboard, power_flow_chart, summary_stats)
    """
    # Generate visualizations
    main_dashboard = plot_simulation_metrics(metrics_df)
    power_flow_chart = create_power_flow_dashboard(metrics_df)
    summary_stats = generate_summary_stats(metrics_df)

    # Save HTML report if requested
    if save_html:
        # Create combined HTML with both charts
        main_html = main_dashboard.to_html(include_plotlyjs=True)
        flow_html = power_flow_chart.to_html(include_plotlyjs=False)

        # Simple HTML template
        # Extract stats for shorter variable names
        stats = summary_stats
        duration = stats.get("simulation_duration_hours", 0)
        solar_gen = stats.get("total_solar_generation_mwh", 0)
        wind_gen = stats.get("total_wind_generation_mwh", 0)
        demand = stats.get("total_demand_mwh", 0)
        renewable_pct = stats.get("renewable_penetration_pct", 0)
        curtail_pct = stats.get("curtailment_rate_pct", 0)
        satisfy_pct = stats.get("demand_satisfaction_rate_pct", 0)
        avg_soc = stats.get("avg_battery_soc", 0)

        combined_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PSIREG Simulation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .stats {{ background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .stat-item {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>PSIREG Simulation Analysis Report</h1>

            <div class="stats">
                <h2>Summary Statistics</h2>
                <div class="stat-item"><b>Simulation Duration:</b> {duration:.1f} hours</div>
                <div class="stat-item"><b>Total Solar Generation:</b> {solar_gen:.1f} MWh</div>
                <div class="stat-item"><b>Total Wind Generation:</b> {wind_gen:.1f} MWh</div>
                <div class="stat-item"><b>Total Demand:</b> {demand:.1f} MWh</div>
                <div class="stat-item"><b>Renewable Penetration:</b> {renewable_pct:.1f}%</div>
                <div class="stat-item"><b>Curtailment Rate:</b> {curtail_pct:.1f}%</div>
                <div class="stat-item"><b>Demand Satisfaction:</b> {satisfy_pct:.1f}%</div>
                <div class="stat-item"><b>Average Battery SoC:</b> {avg_soc:.1f}%</div>
            </div>

            {main_html.split('<body>')[1].split('</body>')[0]}

            <h2>Power Flow Analysis</h2>
            {flow_html.split('<body>')[1].split('</body>')[0]}

        </body>
        </html>
        """

        with open(save_html, "w") as f:
            f.write(combined_html)

    return main_dashboard, power_flow_chart, summary_stats
