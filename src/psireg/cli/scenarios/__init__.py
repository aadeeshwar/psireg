"""Scenarios module for PSIREG renewable energy grid system.

This module provides predefined scenarios for comprehensive renewable energy
grid simulation testing including weather conditions, emergency response,
and various operational scenarios.
"""

from .base import BaseScenario
from .storm_day import StormDayScenario

__all__ = ["BaseScenario", "StormDayScenario"]

__version__ = "0.1.0"
