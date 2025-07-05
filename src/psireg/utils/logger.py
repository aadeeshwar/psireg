"""Logging utilities for PSIREG renewable energy grid system.

This module provides centralized logging configuration for the
Predictive Swarm Intelligence for Renewable Energy Grids research platform.
"""

import logging
import sys

# Create a default logger
logger = logging.getLogger("psireg")

# Configure logging if not already configured
if not logger.handlers:
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

__all__ = ["logger"]
