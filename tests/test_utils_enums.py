"""Tests for domain enumeration definitions."""

from enum import Enum

from psireg.utils.enums import (
    AssetStatus,
    AssetType,
    EnergyUnit,
    FrequencyUnit,
    GridPhase,
    OptimizationTarget,
    PowerUnit,
    SimulationMode,
    VoltageUnit,
    WeatherCondition,
)


class TestPowerUnitEnum:
    """Test PowerUnit enumeration."""

    def test_all_power_units_exist(self):
        """Test all expected power units are defined."""
        expected_units = ["W", "kW", "MW", "GW"]
        actual_values = [unit.value for unit in PowerUnit]
        for unit in expected_units:
            assert unit in actual_values

    def test_power_unit_names(self):
        """Test power unit enum names."""
        assert PowerUnit.WATT.name == "WATT"
        assert PowerUnit.KILOWATT.name == "KILOWATT"
        assert PowerUnit.MEGAWATT.name == "MEGAWATT"

    def test_power_unit_is_enum(self):
        """Test PowerUnit is proper Enum type."""
        assert issubclass(PowerUnit, Enum)
        assert len(PowerUnit) >= 3


class TestEnergyUnitEnum:
    """Test EnergyUnit enumeration."""

    def test_all_energy_units_exist(self):
        """Test all expected energy units are defined."""
        expected_units = ["Wh", "kWh", "MWh", "GWh"]
        actual_values = [unit.value for unit in EnergyUnit]
        for unit in expected_units:
            assert unit in actual_values

    def test_energy_unit_names(self):
        """Test energy unit enum names."""
        assert EnergyUnit.WATT_HOUR.name == "WATT_HOUR"
        assert EnergyUnit.KILOWATT_HOUR.name == "KILOWATT_HOUR"
        assert EnergyUnit.MEGAWATT_HOUR.name == "MEGAWATT_HOUR"


class TestFrequencyUnitEnum:
    """Test FrequencyUnit enumeration."""

    def test_frequency_units_exist(self):
        """Test frequency units are defined."""
        expected_units = ["Hz", "kHz", "MHz"]
        actual_values = [unit.value for unit in FrequencyUnit]
        for unit in expected_units:
            assert unit in actual_values

    def test_hertz_unit(self):
        """Test specific hertz unit."""
        assert FrequencyUnit.HERTZ.value == "Hz"
        assert FrequencyUnit.HERTZ.name == "HERTZ"


class TestVoltageUnitEnum:
    """Test VoltageUnit enumeration."""

    def test_voltage_units_exist(self):
        """Test voltage units are defined."""
        expected_units = ["V", "kV", "MV"]
        actual_values = [unit.value for unit in VoltageUnit]
        for unit in expected_units:
            assert unit in actual_values

    def test_volt_units(self):
        """Test specific voltage units."""
        assert VoltageUnit.VOLT.value == "V"
        assert VoltageUnit.KILOVOLT.value == "kV"
        assert VoltageUnit.MEGAVOLT.value == "MV"


class TestAssetTypeEnum:
    """Test AssetType enumeration."""

    def test_core_asset_types_exist(self):
        """Test core renewable energy asset types."""
        required_types = ["SOLAR", "WIND", "BATTERY", "LOAD"]
        actual_names = [asset.name for asset in AssetType]
        for asset_type in required_types:
            assert asset_type in actual_names

    def test_additional_asset_types(self):
        """Test additional asset types that may exist."""
        actual_names = [asset.name for asset in AssetType]
        # At least some additional types should exist beyond core 4
        assert len(actual_names) >= 4

    def test_asset_type_values(self):
        """Test asset type values are descriptive."""
        for asset_type in AssetType:
            assert isinstance(asset_type.value, str)
            assert len(asset_type.value) > 0


class TestAssetStatusEnum:
    """Test AssetStatus enumeration."""

    def test_operational_statuses_exist(self):
        """Test operational status values."""
        required_statuses = ["ONLINE", "OFFLINE", "MAINTENANCE", "FAULT"]
        actual_values = [status.value for status in AssetStatus]
        for status in required_statuses:
            assert status in actual_values

    def test_additional_statuses(self):
        """Test additional status types."""
        actual_values = [status.value for status in AssetStatus]
        # Should have at least the 4 core statuses
        assert len(actual_values) >= 4

    def test_status_enum_properties(self):
        """Test status enum properties."""
        assert AssetStatus.ONLINE.name == "ONLINE"
        assert isinstance(AssetStatus.FAULT.value, str)


class TestWeatherConditionEnum:
    """Test WeatherCondition enumeration."""

    def test_basic_weather_conditions(self):
        """Test basic weather conditions exist."""
        required_conditions = ["CLEAR", "CLOUDY", "RAINY"]
        actual_values = [condition.value for condition in WeatherCondition]
        for condition in required_conditions:
            assert condition in actual_values

    def test_extended_weather_conditions(self):
        """Test extended weather conditions."""
        actual_values = [condition.value for condition in WeatherCondition]
        # Should have at least 3 basic conditions
        assert len(actual_values) >= 3

    def test_weather_condition_names(self):
        """Test weather condition enum names."""
        for condition in WeatherCondition:
            assert condition.name.isupper()
            assert isinstance(condition.value, str)


class TestGridPhaseEnum:
    """Test GridPhase enumeration."""

    def test_three_phase_system(self):
        """Test three-phase electrical system representation."""
        required_phases = ["PHASE_A", "PHASE_B", "PHASE_C"]
        actual_values = [phase.value for phase in GridPhase]
        for phase in required_phases:
            assert phase in actual_values

    def test_neutral_and_ground(self):
        """Test neutral and ground phases if they exist."""
        actual_values = [phase.value for phase in GridPhase]
        # At minimum should have 3 phases
        assert len(actual_values) >= 3

    def test_phase_naming_convention(self):
        """Test phase naming follows convention."""
        for phase in GridPhase:
            assert "PHASE" in phase.value or phase.value in ["NEUTRAL", "GROUND"]


class TestSimulationModeEnum:
    """Test SimulationMode enumeration."""

    def test_simulation_modes_exist(self):
        """Test simulation mode options."""
        expected_modes = ["REAL_TIME", "HISTORICAL", "FORECAST"]
        actual_values = [mode.value for mode in SimulationMode]
        for mode in expected_modes:
            assert mode in actual_values

    def test_simulation_mode_properties(self):
        """Test simulation mode enum properties."""
        assert len(SimulationMode) >= 3
        for mode in SimulationMode:
            assert isinstance(mode.value, str)
            assert len(mode.value) > 0


class TestOptimizationTargetEnum:
    """Test OptimizationTarget enumeration."""

    def test_optimization_targets_exist(self):
        """Test optimization target options."""
        expected_targets = ["COST_MINIMIZATION", "EFFICIENCY_MAXIMIZATION", "EMISSION_REDUCTION"]
        actual_values = [target.value for target in OptimizationTarget]
        for target in expected_targets:
            assert target in actual_values

    def test_additional_optimization_targets(self):
        """Test additional optimization targets."""
        actual_values = [target.value for target in OptimizationTarget]
        # Should have at least 3 core targets
        assert len(actual_values) >= 3

    def test_optimization_target_names(self):
        """Test optimization target naming."""
        for target in OptimizationTarget:
            assert "_" in target.value or target.value.isupper()
            assert isinstance(target.value, str)


class TestEnumConsistency:
    """Test consistency across all enums."""

    def test_all_enums_are_proper_enums(self):
        """Test all defined classes are proper Enum subclasses."""
        enum_classes = [
            PowerUnit,
            EnergyUnit,
            FrequencyUnit,
            VoltageUnit,
            AssetType,
            AssetStatus,
            WeatherCondition,
            GridPhase,
            SimulationMode,
            OptimizationTarget,
        ]

        for enum_class in enum_classes:
            assert issubclass(enum_class, Enum)
            assert len(enum_class) > 0

    def test_enum_values_are_strings(self):
        """Test all enum values are strings."""
        enum_classes = [
            PowerUnit,
            EnergyUnit,
            FrequencyUnit,
            VoltageUnit,
            AssetType,
            AssetStatus,
            WeatherCondition,
            GridPhase,
            SimulationMode,
            OptimizationTarget,
        ]

        for enum_class in enum_classes:
            for enum_member in enum_class:
                assert isinstance(enum_member.value, str)
                assert len(enum_member.value) > 0

    def test_enum_names_follow_convention(self):
        """Test enum names follow Python naming conventions."""
        enum_classes = [
            PowerUnit,
            EnergyUnit,
            FrequencyUnit,
            VoltageUnit,
            AssetType,
            AssetStatus,
            WeatherCondition,
            GridPhase,
            SimulationMode,
            OptimizationTarget,
        ]

        for enum_class in enum_classes:
            for enum_member in enum_class:
                # Names should be uppercase with underscores
                assert enum_member.name.isupper()
                assert " " not in enum_member.name
