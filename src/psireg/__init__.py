"""PSIREG - Predictive Swarm Intelligence for Renewable Energy Grids."""

from .sim import GridEngine, GridState, NetworkNode, TransmissionLine

# Optional visualization module (requires pandas and plotly)
try:
    from . import viz  # noqa: F401

    _viz_available = True
except ImportError:
    _viz_available = False

__version__ = "0.1.0"
__author__ = "Aadeeshwar Pathak"
__description__ = "Predictive Swarm Intelligence for Renewable Energy Grids"

__all__ = [
    "GridEngine",
    "GridState",
    "NetworkNode",
    "TransmissionLine",
]

if _viz_available:
    __all__.append("viz")
