"""Controller interfaces for PSI-based renewable energy grid control.

This module provides different control strategies for renewable energy grid management:
- Rule-based controllers using traditional power system control logic
- ML-only controllers using reinforcement learning models
- Swarm-only controllers using distributed swarm intelligence
- Comparison framework for evaluating controller efficiency
"""

from .base import BaseController

# Import other controllers only if they exist
__all__ = ["BaseController"]

try:
    from .rule import RuleBasedController  # noqa: F401

    __all__.append("RuleBasedController")
except ImportError:
    pass

try:
    from .ml import MLController  # noqa: F401

    __all__.append("MLController")
except ImportError:
    pass

try:
    from .swarm import SwarmController  # noqa: F401

    __all__.append("SwarmController")
except ImportError:
    pass

try:
    from .psi import PSIController  # noqa: F401

    __all__.append("PSIController")
except ImportError:
    pass

try:
    from .comparison import ComparisonMetrics, ControllerComparison  # noqa: F401

    __all__.extend(["ControllerComparison", "ComparisonMetrics"])
except ImportError:
    pass
