"""CLI module for PSIREG renewable energy grid system.

This module provides command-line interface functionality for the
Predictive Swarm Intelligence for Renewable Energy Grids system.

The CLI enables:
- Scenario orchestration and execution
- Grid simulation control
- Configuration management
- Results visualization and analysis
"""

from .main import app, create_cli_app

__all__ = ["app", "create_cli_app"]

__version__ = "0.1.0"
