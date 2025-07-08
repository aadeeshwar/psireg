"""Visualization module for PSIREG simulation metrics.

This module provides interactive visualization capabilities for analyzing
simulation results including power generation, demand, battery behavior,
and grid balance metrics.
"""

try:
    from .metrics import create_metrics_report, create_power_flow_dashboard, plot_simulation_metrics
    from .viz_comparison import visualize_controller_comparison

    __all__ = [
        "plot_simulation_metrics",
        "create_power_flow_dashboard",
        "create_metrics_report",
        "visualize_controller_comparison",
    ]
except ImportError as e:
    import warnings

    warnings.warn(
        f"Visualization dependencies not available: {e}. " "Install with: poetry install or pip install pandas plotly",
        ImportWarning,
        stacklevel=2,
    )

    # Define dummy functions for graceful degradation
    def plot_simulation_metrics(*args, **kwargs):  # type: ignore[no-untyped-def, misc]
        raise ImportError("Visualization dependencies (pandas, plotly) not installed")

    def create_power_flow_dashboard(*args, **kwargs):  # type: ignore[no-untyped-def, misc]
        raise ImportError("Visualization dependencies (pandas, plotly) not installed")

    def create_metrics_report(*args, **kwargs):  # type: ignore[no-untyped-def, misc]
        raise ImportError("Visualization dependencies (pandas, plotly) not installed")

    def visualize_controller_comparison(*args, **kwargs):  # type: ignore[no-untyped-def, misc]
        raise ImportError("Visualization dependencies (pandas, plotly, scipy) not installed")

    __all__ = [
        "plot_simulation_metrics",
        "create_power_flow_dashboard",
        "create_metrics_report",
        "visualize_controller_comparison",
    ]
