# PSIREG Visualization Module

Interactive visualization capabilities for PSIREG renewable energy grid simulation results.

## Overview

The `psireg.viz` module provides comprehensive plotting and analysis tools for understanding simulation metrics from renewable energy grid simulations. It creates interactive Plotly-based dashboards that help visualize:

- Power generation patterns (solar, wind)
- Demand profiles and grid balance
- Battery charging/discharging behavior
- State of charge trends
- Energy curtailment and unmet demand
- Power flow analysis

## Installation

Install the required visualization dependencies:

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install pandas plotly
```

## Quick Start

```python
import pandas as pd
from psireg.viz import plot_simulation_metrics, create_power_flow_dashboard

# Assuming you have a DataFrame with simulation metrics
# See "Data Format" section below for required columns
metrics_df = load_your_simulation_data()

# Create comprehensive dashboard
fig = plot_simulation_metrics(metrics_df)
fig.show()

# Create power flow analysis
flow_fig = create_power_flow_dashboard(metrics_df)
flow_fig.show()
```

## Data Format

The visualization functions expect a pandas DataFrame with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Simulation timestamp |
| `solar_output_mw` | float | Solar power generation (MW) |
| `wind_output_mw` | float | Wind power generation (MW) |
| `battery_charge_mw` | float | Battery charging power (MW) |
| `battery_discharge_mw` | float | Battery discharging power (MW) |
| `battery_soc` | float | Battery state of charge (0.0-1.0) |
| `demand_mw` | float | Electrical demand (MW) |
| `net_balance_mw` | float | Generation minus demand (MW) |
| `curtailed_energy_mw` | float | Curtailed renewable energy (MW) |
| `unmet_demand_mw` | float | Unmet electrical demand (MW) |

## Functions

### `plot_simulation_metrics(metrics_df, title, height, show_grid, theme)`

Creates a comprehensive multi-subplot dashboard with:
- Power generation time series
- Demand and net balance
- Battery operations
- Battery state of charge with reference levels
- Curtailed energy
- Unmet demand

**Parameters:**
- `metrics_df`: DataFrame with simulation data
- `title`: Dashboard title (default: "PSIREG Simulation Metrics")
- `height`: Figure height in pixels (default: 1200)
- `show_grid`: Show grid lines (default: True)
- `theme`: Plotly theme (default: "plotly_white")

### `create_power_flow_dashboard(metrics_df, title, height)`

Creates a stacked area chart showing total power supply vs demand.

**Parameters:**
- `metrics_df`: DataFrame with simulation data
- `title`: Chart title (default: "Power Flow Analysis")
- `height`: Figure height in pixels (default: 800)

### `generate_summary_stats(metrics_df)`

Generates key performance indicators and statistics.

**Returns:**
- Dictionary with simulation KPIs including:
  - Total generation by source (MWh)
  - Renewable penetration (%)
  - Curtailment rate (%)
  - Demand satisfaction rate (%)
  - Battery utilization metrics

### `create_metrics_report(metrics_df, save_html)`

Generates a comprehensive report with multiple visualizations and statistics.

**Parameters:**
- `metrics_df`: DataFrame with simulation data
- `save_html`: Optional path to save HTML report

**Returns:**
- Tuple of (main_dashboard, power_flow_chart, summary_stats)

## Example Usage

See `examples/visualization_demo.py` for a complete demonstration including:
- Sample data generation
- Creating interactive dashboards
- Generating HTML reports
- Summary statistics calculation

Run the demo:

```bash
cd examples
python visualization_demo.py
```

## Features

### Interactive Elements
- Hover tooltips with detailed information
- Zoom and pan capabilities
- Legend toggling
- Time-based x-axis synchronization across subplots

### Color Scheme
- **Solar**: Orange (#FFA500)
- **Wind**: Dodger Blue (#1E90FF)
- **Demand**: Crimson (#DC143C)
- **Battery Charge**: Lime Green (#32CD32)
- **Battery Discharge**: Tomato (#FF6347)
- **Battery SoC**: Medium Purple (#9370DB)
- **Net Balance**: Sea Green (#2E8B57)
- **Curtailed Energy**: Hot Pink (#FF69B4)
- **Unmet Demand**: Fire Brick (#B22222)

### Reference Lines
- Zero line for net balance
- 20% and 80% SoC levels for battery management
- Grid lines for easy value reading

## Error Handling

The module includes comprehensive error handling:
- Missing dependency warnings
- Required column validation
- Empty DataFrame detection
- Graceful degradation when dependencies unavailable

## Export Options

Visualizations can be exported as:
- Interactive HTML files
- Static images (PNG, JPG, PDF, SVG)
- Complete HTML reports with statistics

```python
# Save individual charts
fig.write_html("simulation_dashboard.html")
fig.write_image("dashboard.png")

# Save comprehensive report
create_metrics_report(df, save_html="full_report.html")
```

## Performance Considerations

- Optimized for datasets with hundreds to thousands of time points
- Efficient rendering for 24 hours to 7 days of simulation data
- Memory-efficient processing for large datasets
- Fast interactive updates and zooming

## Customization

All functions support customization through parameters:
- Titles and labels
- Colors and themes
- Figure dimensions
- Grid and axis options
- Layout preferences

## Integration

The visualization module integrates seamlessly with:
- PSIREG simulation engine
- GridEngine output data
- Battery and renewable asset metrics
- Demand response analysis
- Grid stability studies 