"""Type definitions and constants for PSIREG renewable energy grid system.

This module provides type aliases, domain constants, and utility functions for the
Predictive Swarm Intelligence for Renewable Energy Grids research platform.
"""

from datetime import datetime, timedelta
from decimal import Decimal, getcontext

# Set high precision for decimal calculations
getcontext().prec = 28

# =============================================================================
# Power Type Definitions and Aliases
# =============================================================================

type Power = float
type MW = float  # Megawatts
type kW = float  # Kilowatts
type W = float  # Watts

# Power Constants
MIN_POWER_MW: float = 0.0
MAX_SOLAR_OUTPUT_MW: float = 500.0  # Typical utility-scale solar farm
MAX_WIND_OUTPUT_MW: float = 800.0  # Large wind farm capacity
MAX_BATTERY_POWER_MW: float = 200.0  # Grid-scale battery system
MAX_HYDRO_OUTPUT_MW: float = 1000.0  # Large hydroelectric facility

# =============================================================================
# Energy Type Definitions and Aliases
# =============================================================================

type Energy = float
type MWh = float  # Megawatt-hours
type kWh = float  # Kilowatt-hours
type Wh = float  # Watt-hours

# Energy Constants
MIN_ENERGY_MWH: float = 0.0
MAX_BATTERY_CAPACITY_MWH: float = 800.0  # Large grid battery storage

# PSIREG Research Target Constants
ENERGY_WASTE_REDUCTION_TARGET: float = 0.20  # 20% reduction target
FOSSIL_FUEL_REDUCTION_TARGET: float = 0.15  # 15% reduction target
CALIFORNIA_CURTAILED_ENERGY_2022_MWH: float = 2_400_000.0  # Historical reference

# =============================================================================
# Electrical Parameter Type Definitions
# =============================================================================

type Voltage = float
type kV = float  # Kilovolts
type V = float  # Volts

type Frequency = float
type Hz = float  # Hertz

# Electrical Constants
GRID_FREQUENCY_HZ: float = 60.0  # North American standard
TRANSMISSION_VOLTAGE_KV: float = 230.0  # High voltage transmission
DISTRIBUTION_VOLTAGE_KV: float = 12.47  # Medium voltage distribution
LOW_VOLTAGE_V: float = 480.0  # Low voltage industrial
RESIDENTIAL_VOLTAGE_V: float = 240.0  # Residential service

# =============================================================================
# Time Type Definitions
# =============================================================================

type Timestamp = datetime
type Duration = timedelta
type TimeStep = int  # Simulation time step index

# Time Constants
SIMULATION_TIMESTEP_MINUTES: int = 15  # 15-minute intervals for PSIREG grid simulation
HOURS_PER_DAY: int = 24
MINUTES_PER_HOUR: int = 60
SECONDS_PER_MINUTE: int = 60
MILLISECONDS_PER_SECOND: int = 1000

# PSIREG-Specific Simulation Constants
RL_PREDICTION_HORIZON_HOURS: int = 24  # RL model prediction window
SWARM_RESPONSE_TIME_SECONDS: float = 1.0  # Local swarm response time
PHEROMONE_DECAY_RATE: float = 0.95  # Pheromone signal decay per timestep
GRID_STABILITY_THRESHOLD_HZ: float = 0.1  # Maximum frequency deviation tolerance

# =============================================================================
# Geographic Type Definitions
# =============================================================================

type Latitude = float  # Degrees north/south (-90 to +90)
type Longitude = float  # Degrees east/west (-180 to +180)
type Coordinates = tuple[Latitude, Longitude]

# Geographic Constants
MIN_LATITUDE: float = -90.0
MAX_LATITUDE: float = 90.0
MIN_LONGITUDE: float = -180.0
MAX_LONGITUDE: float = 180.0

# =============================================================================
# Precision Type Definitions
# =============================================================================

type PrecisionFloat = float
type PrecisionDecimal = Decimal

# =============================================================================
# Power Conversion Functions
# =============================================================================


def mw_to_kw(megawatts: MW) -> kW:
    """Convert megawatts to kilowatts."""
    return megawatts * 1000.0


def kw_to_mw(kilowatts: kW) -> MW:
    """Convert kilowatts to megawatts."""
    return kilowatts / 1000.0


def kw_to_w(kilowatts: kW) -> W:
    """Convert kilowatts to watts."""
    return kilowatts * 1000.0


def w_to_kw(watts: W) -> kW:
    """Convert watts to kilowatts."""
    return watts / 1000.0


def mw_to_w(megawatts: MW) -> W:
    """Convert megawatts to watts."""
    return megawatts * 1_000_000.0


def w_to_mw(watts: W) -> MW:
    """Convert watts to megawatts."""
    return watts / 1_000_000.0


# =============================================================================
# Energy Conversion Functions
# =============================================================================


def mwh_to_kwh(megawatt_hours: MWh) -> kWh:
    """Convert megawatt-hours to kilowatt-hours."""
    return megawatt_hours * 1000.0


def kwh_to_mwh(kilowatt_hours: kWh) -> MWh:
    """Convert kilowatt-hours to megawatt-hours."""
    return kilowatt_hours / 1000.0


def kwh_to_wh(kilowatt_hours: kWh) -> Wh:
    """Convert kilowatt-hours to watt-hours."""
    return kilowatt_hours * 1000.0


def wh_to_kwh(watt_hours: Wh) -> kWh:
    """Convert watt-hours to kilowatt-hours."""
    return watt_hours / 1000.0


def mwh_to_wh(megawatt_hours: MWh) -> Wh:
    """Convert megawatt-hours to watt-hours."""
    return megawatt_hours * 1_000_000.0


def wh_to_mwh(watt_hours: Wh) -> MWh:
    """Convert watt-hours to megawatt-hours."""
    return watt_hours / 1_000_000.0


# =============================================================================
# Voltage Conversion Functions
# =============================================================================


def kv_to_v(kilovolts: kV) -> V:
    """Convert kilovolts to volts."""
    return kilovolts * 1000.0


def v_to_kv(volts: V) -> kV:
    """Convert volts to kilovolts."""
    return volts / 1000.0


# =============================================================================
# Time Conversion Functions
# =============================================================================


def hours_to_minutes(hours: float) -> float:
    """Convert hours to minutes."""
    return hours * MINUTES_PER_HOUR


def minutes_to_seconds(minutes: float) -> float:
    """Convert minutes to seconds."""
    return minutes * SECONDS_PER_MINUTE


def hours_to_seconds(hours: float) -> float:
    """Convert hours to seconds."""
    return hours * MINUTES_PER_HOUR * SECONDS_PER_MINUTE


# =============================================================================
# Validation Functions
# =============================================================================


def is_valid_power(power_value: float) -> bool:
    """Validate that power value is non-negative."""
    return power_value >= 0.0


def is_valid_energy(energy_value: float) -> bool:
    """Validate that energy value is non-negative."""
    return energy_value >= 0.0


def is_valid_voltage(voltage_value: float) -> bool:
    """Validate that voltage value is positive."""
    return voltage_value > 0.0


def is_valid_frequency(frequency_value: float) -> bool:
    """Validate that frequency value is positive."""
    return frequency_value > 0.0


def is_valid_latitude(latitude: float) -> bool:
    """Validate latitude is within valid range."""
    return MIN_LATITUDE <= latitude <= MAX_LATITUDE


def is_valid_longitude(longitude: float) -> bool:
    """Validate longitude is within valid range."""
    return MIN_LONGITUDE <= longitude <= MAX_LONGITUDE


def is_valid_coordinates(coords: Coordinates) -> bool:
    """Validate coordinate pair."""
    lat, lon = coords
    return is_valid_latitude(lat) and is_valid_longitude(lon)


# =============================================================================
# Precision Conversion Functions
# =============================================================================


def to_precision_decimal(value: float | int | str) -> PrecisionDecimal:
    """Convert value to high-precision Decimal."""
    return Decimal(str(value))


def to_precision_float(value: Decimal | int | str) -> PrecisionFloat:
    """Convert value to float (with potential precision loss warning)."""
    return float(value)


# =============================================================================
# Range Validation Functions
# =============================================================================


def validate_power_range(power: float, min_power: float = 0.0, max_power: float = float("inf")) -> bool:
    """Validate power is within specified range."""
    return min_power <= power <= max_power


def validate_energy_range(energy: float, min_energy: float = 0.0, max_energy: float = float("inf")) -> bool:
    """Validate energy is within specified range."""
    return min_energy <= energy <= max_energy


def validate_voltage_range(voltage: float, min_voltage: float = 0.0, max_voltage: float = float("inf")) -> bool:
    """Validate voltage is within specified range."""
    return min_voltage < voltage <= max_voltage


# =============================================================================
# Unit Scaling Functions
# =============================================================================


def scale_power_to_base_unit(value: float, unit: str) -> MW:
    """Scale power value to base unit (MW)."""
    if unit == "W":
        return w_to_mw(value)
    elif unit == "kW":
        return kw_to_mw(value)
    elif unit == "MW":
        return float(value)
    elif unit == "GW":
        return float(value * 1000.0)
    else:
        raise ValueError(f"Unknown power unit: {unit}")


def scale_energy_to_base_unit(value: float, unit: str) -> MWh:
    """Scale energy value to base unit (MWh)."""
    if unit == "Wh":
        return wh_to_mwh(value)
    elif unit == "kWh":
        return kwh_to_mwh(value)
    elif unit == "MWh":
        return float(value)
    elif unit == "GWh":
        return float(value * 1000.0)
    else:
        raise ValueError(f"Unknown energy unit: {unit}")


# =============================================================================
# Mathematical Utility Functions
# =============================================================================


def calculate_efficiency(output: float, input: float) -> float:
    """Calculate efficiency as output/input ratio."""
    if input == 0.0:
        return 0.0
    return output / input


def calculate_capacity_factor(actual_output: float, rated_capacity: float, duration_hours: float) -> float:
    """Calculate capacity factor for renewable energy assets."""
    if rated_capacity == 0.0 or duration_hours == 0.0:
        return 0.0
    theoretical_max = rated_capacity * duration_hours
    return actual_output / theoretical_max


def normalize_to_per_unit(value: float, base_value: float) -> float:
    """Convert value to per-unit system."""
    if base_value == 0.0:
        return 0.0
    return value / base_value
