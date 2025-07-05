"""Tests for domain constants and typing utilities."""

from decimal import Decimal

import psireg.utils.types as types
import pytest
from psireg.utils.enums import (
    AssetStatus,
    AssetType,
    EnergyUnit,
    GridPhase,
    PowerUnit,
    WeatherCondition,
)


class TestPowerTypes:
    """Test power-related type definitions and constants."""

    def test_power_aliases_exist(self):
        """Test that power type aliases are defined."""
        assert hasattr(types, "Power")
        assert hasattr(types, "MW")
        assert hasattr(types, "kW")
        assert hasattr(types, "W")

    def test_power_conversion_functions(self):
        """Test power unit conversion functions."""
        # MW to kW
        assert types.mw_to_kw(1.0) == 1000.0
        assert types.mw_to_kw(2.5) == 2500.0

        # kW to MW
        assert types.kw_to_mw(1000.0) == 1.0
        assert types.kw_to_mw(2500.0) == 2.5

        # kW to W
        assert types.kw_to_w(1.0) == 1000.0
        assert types.kw_to_w(5.5) == 5500.0

    def test_power_validation(self):
        """Test power value validation."""
        assert types.is_valid_power(100.0) is True
        assert types.is_valid_power(0.0) is True
        assert types.is_valid_power(-10.0) is False  # Negative power invalid

    def test_power_constants(self):
        """Test predefined power constants."""
        assert types.MAX_SOLAR_OUTPUT_MW > 0
        assert types.MAX_WIND_OUTPUT_MW > 0
        assert types.MAX_BATTERY_POWER_MW > 0
        assert types.MIN_POWER_MW == 0.0


class TestEnergyTypes:
    """Test energy-related type definitions and constants."""

    def test_energy_aliases_exist(self):
        """Test that energy type aliases are defined."""
        assert hasattr(types, "Energy")
        assert hasattr(types, "MWh")
        assert hasattr(types, "kWh")
        assert hasattr(types, "Wh")

    def test_energy_conversion_functions(self):
        """Test energy unit conversion functions."""
        # MWh to kWh
        assert types.mwh_to_kwh(1.0) == 1000.0
        assert types.mwh_to_kwh(0.5) == 500.0

        # kWh to MWh
        assert types.kwh_to_mwh(1000.0) == 1.0
        assert types.kwh_to_mwh(500.0) == 0.5

        # kWh to Wh
        assert types.kwh_to_wh(1.0) == 1000.0

    def test_energy_validation(self):
        """Test energy value validation."""
        assert types.is_valid_energy(100.0) is True
        assert types.is_valid_energy(0.0) is True
        assert types.is_valid_energy(-10.0) is False


class TestElectricalTypes:
    """Test electrical parameter type definitions."""

    def test_voltage_types_exist(self):
        """Test voltage type definitions."""
        assert hasattr(types, "Voltage")
        assert hasattr(types, "kV")
        assert hasattr(types, "V")

    def test_frequency_types_exist(self):
        """Test frequency type definitions."""
        assert hasattr(types, "Frequency")
        assert hasattr(types, "Hz")

    def test_voltage_conversion(self):
        """Test voltage unit conversions."""
        assert types.kv_to_v(1.0) == 1000.0
        assert types.v_to_kv(1000.0) == 1.0

    def test_electrical_constants(self):
        """Test electrical system constants."""
        assert types.GRID_FREQUENCY_HZ == 60.0  # North American standard
        assert types.TRANSMISSION_VOLTAGE_KV > 100.0
        assert types.DISTRIBUTION_VOLTAGE_KV < 50.0

    def test_voltage_validation(self):
        """Test voltage range validation."""
        assert types.is_valid_voltage(120.0) is True
        assert types.is_valid_voltage(0.0) is False  # Zero voltage invalid
        assert types.is_valid_voltage(-120.0) is False  # Negative voltage invalid


class TestTimeTypes:
    """Test time-related type definitions."""

    def test_time_aliases_exist(self):
        """Test time type aliases are defined."""
        assert hasattr(types, "Timestamp")
        assert hasattr(types, "Duration")
        assert hasattr(types, "TimeStep")

    def test_time_constants(self):
        """Test time-related constants."""
        assert types.SIMULATION_TIMESTEP_MINUTES > 0
        assert types.HOURS_PER_DAY == 24
        assert types.MINUTES_PER_HOUR == 60
        assert types.SECONDS_PER_MINUTE == 60


class TestCoordinateTypes:
    """Test geographic coordinate types."""

    def test_coordinate_aliases_exist(self):
        """Test coordinate type aliases."""
        assert hasattr(types, "Latitude")
        assert hasattr(types, "Longitude")
        assert hasattr(types, "Coordinates")

    def test_coordinate_validation(self):
        """Test coordinate validation functions."""
        # Valid coordinates
        assert types.is_valid_latitude(45.0) is True
        assert types.is_valid_longitude(-122.0) is True

        # Invalid coordinates
        assert types.is_valid_latitude(91.0) is False  # > 90
        assert types.is_valid_latitude(-91.0) is False  # < -90
        assert types.is_valid_longitude(181.0) is False  # > 180
        assert types.is_valid_longitude(-181.0) is False  # < -180


class TestEnums:
    """Test enumeration definitions."""

    def test_power_unit_enum(self):
        """Test PowerUnit enumeration."""
        assert PowerUnit.WATT.value == "W"
        assert PowerUnit.KILOWATT.value == "kW"
        assert PowerUnit.MEGAWATT.value == "MW"

    def test_energy_unit_enum(self):
        """Test EnergyUnit enumeration."""
        assert EnergyUnit.WATT_HOUR.value == "Wh"
        assert EnergyUnit.KILOWATT_HOUR.value == "kWh"
        assert EnergyUnit.MEGAWATT_HOUR.value == "MWh"

    def test_asset_type_enum(self):
        """Test AssetType enumeration."""
        assert AssetType.SOLAR in [AssetType.SOLAR, AssetType.WIND, AssetType.BATTERY, AssetType.LOAD]
        assert len(AssetType) >= 4  # At least 4 asset types

    def test_asset_status_enum(self):
        """Test AssetStatus enumeration."""
        expected_statuses = ["ONLINE", "OFFLINE", "MAINTENANCE", "FAULT"]
        actual_values = [status.value for status in AssetStatus]
        for status in expected_statuses:
            assert status in actual_values

    def test_weather_condition_enum(self):
        """Test WeatherCondition enumeration."""
        conditions = [condition.value for condition in WeatherCondition]
        assert "CLEAR" in conditions
        assert "CLOUDY" in conditions
        assert "RAINY" in conditions

    def test_grid_phase_enum(self):
        """Test GridPhase enumeration."""
        phases = [phase.value for phase in GridPhase]
        assert "PHASE_A" in phases
        assert "PHASE_B" in phases
        assert "PHASE_C" in phases


class TestPrecisionTypes:
    """Test high-precision numeric types."""

    def test_precision_types_exist(self):
        """Test that precision types are defined."""
        assert hasattr(types, "PrecisionFloat")
        assert hasattr(types, "PrecisionDecimal")

    def test_precision_conversion(self):
        """Test precision type conversions."""
        value = 123.456789
        precise_decimal = types.to_precision_decimal(value)
        assert isinstance(precise_decimal, Decimal)
        assert float(precise_decimal) == pytest.approx(value, rel=1e-9)


class TestUnitConversions:
    """Test comprehensive unit conversion utilities."""

    def test_power_conversion_matrix(self):
        """Test all power conversions work correctly."""
        # Test conversion consistency
        original_mw = 5.0
        converted_kw = types.mw_to_kw(original_mw)
        converted_back = types.kw_to_mw(converted_kw)
        assert converted_back == pytest.approx(original_mw, rel=1e-9)

    def test_energy_conversion_matrix(self):
        """Test all energy conversions work correctly."""
        original_mwh = 2.5
        converted_kwh = types.mwh_to_kwh(original_mwh)
        converted_back = types.kwh_to_mwh(converted_kwh)
        assert converted_back == pytest.approx(original_mwh, rel=1e-9)

    def test_time_conversion_utilities(self):
        """Test time conversion utilities."""
        if hasattr(types, "hours_to_minutes"):
            assert types.hours_to_minutes(2.0) == 120.0
        if hasattr(types, "minutes_to_seconds"):
            assert types.minutes_to_seconds(5.0) == 300.0
