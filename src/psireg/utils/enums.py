"""Domain enumerations for PSIREG renewable energy grid system.

This module defines enumerations for various domain concepts used in the
Predictive Swarm Intelligence for Renewable Energy Grids research platform.
"""

from enum import Enum


class PowerUnit(Enum):
    """Power measurement units."""

    WATT = "W"
    KILOWATT = "kW"
    MEGAWATT = "MW"
    GIGAWATT = "GW"


class EnergyUnit(Enum):
    """Energy measurement units."""

    WATT_HOUR = "Wh"
    KILOWATT_HOUR = "kWh"
    MEGAWATT_HOUR = "MWh"
    GIGAWATT_HOUR = "GWh"


class FrequencyUnit(Enum):
    """Frequency measurement units."""

    HERTZ = "Hz"
    KILOHERTZ = "kHz"
    MEGAHERTZ = "MHz"


class VoltageUnit(Enum):
    """Voltage measurement units."""

    VOLT = "V"
    KILOVOLT = "kV"
    MEGAVOLT = "MV"


class AssetType(Enum):
    """Types of grid assets in the renewable energy system."""

    SOLAR = "Solar Photovoltaic"
    WIND = "Wind Turbine"
    BATTERY = "Battery Storage"
    LOAD = "Electrical Load"
    HYDRO = "Hydroelectric"
    NUCLEAR = "Nuclear Power"
    COAL = "Coal Power Plant"
    GAS = "Natural Gas"
    BIOMASS = "Biomass Power"
    GEOTHERMAL = "Geothermal"


class AssetStatus(Enum):
    """Operational status of grid assets."""

    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    MAINTENANCE = "MAINTENANCE"
    FAULT = "FAULT"
    STARTING = "STARTING"
    STOPPING = "STOPPING"
    DEGRADED = "DEGRADED"
    WARNING = "WARNING"


class WeatherCondition(Enum):
    """Weather conditions affecting renewable energy generation."""

    CLEAR = "CLEAR"
    CLOUDY = "CLOUDY"
    RAINY = "RAINY"
    SNOWY = "SNOWY"
    WINDY = "WINDY"
    FOGGY = "FOGGY"
    STORMY = "STORMY"
    PARTLY_CLOUDY = "PARTLY_CLOUDY"


class GridPhase(Enum):
    """Electrical grid phases for three-phase power systems."""

    PHASE_A = "PHASE_A"
    PHASE_B = "PHASE_B"
    PHASE_C = "PHASE_C"
    NEUTRAL = "NEUTRAL"
    GROUND = "GROUND"


class SimulationMode(Enum):
    """Simulation execution modes."""

    REAL_TIME = "REAL_TIME"
    HISTORICAL = "HISTORICAL"
    FORECAST = "FORECAST"
    BATCH = "BATCH"
    INTERACTIVE = "INTERACTIVE"


class OptimizationTarget(Enum):
    """Optimization objectives for the grid system."""

    COST_MINIMIZATION = "COST_MINIMIZATION"
    EFFICIENCY_MAXIMIZATION = "EFFICIENCY_MAXIMIZATION"
    EMISSION_REDUCTION = "EMISSION_REDUCTION"
    RELIABILITY_MAXIMIZATION = "RELIABILITY_MAXIMIZATION"
    LOAD_BALANCING = "LOAD_BALANCING"
    PEAK_SHAVING = "PEAK_SHAVING"
    RENEWABLE_MAXIMIZATION = "RENEWABLE_MAXIMIZATION"
