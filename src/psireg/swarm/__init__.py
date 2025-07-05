"""Swarm intelligence module for PSIREG renewable energy grid system.

This module provides swarm-based coordination algorithms for distributed grid
assets using pheromone-based communication and local optimization strategies.
"""

from .agents.battery_agent import BatteryAgent

__all__ = [
    "BatteryAgent",
]
