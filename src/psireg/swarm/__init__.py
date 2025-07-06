"""Swarm intelligence module for PSIREG renewable energy grid system.

This module provides swarm-based coordination algorithms for distributed grid
assets using pheromone-based communication and local optimization strategies.

The primary components include:
- PheromoneField: Grid-based pheromone storage with decay and diffusion
- SwarmBus: Central coordination system for agent management
- Agent classes: Intelligent agents for different grid asset types
"""

from .agents.battery_agent import BatteryAgent
from .agents.demand_agent import DemandAgent
from .agents.solar_agent import SolarAgent
from .agents.wind_agent import WindAgent
from .pheromone import (
    GridPosition,
    PheromoneField,
    PheromoneType,
    SwarmBus,
)

__all__ = [
    # Core pheromone infrastructure
    "PheromoneField",
    "SwarmBus",
    "PheromoneType",
    "GridPosition",
    # Agent classes
    "BatteryAgent",
    "DemandAgent",
    "SolarAgent",
    "WindAgent",
]
