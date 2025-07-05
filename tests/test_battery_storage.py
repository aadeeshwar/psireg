"""Test battery storage implementation."""

import pytest
from psireg.sim.assets.battery import Battery
from psireg.utils.enums import AssetStatus, AssetType
from pydantic import ValidationError


class TestBatteryCreation:
    """Test Battery asset creation and validation."""

    def test_battery_creation_basic(self):
        """Test creation of basic battery asset."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        assert battery.asset_id == "battery_001"
        assert battery.asset_type == AssetType.BATTERY
        assert battery.name == "Test Battery"
        assert battery.node_id == "node_1"
        assert battery.capacity_mw == 10.0
        assert battery.energy_capacity_mwh == 40.0
        assert battery.current_soc_percent == 50.0
        assert battery.status == AssetStatus.OFFLINE
        assert battery.current_output_mw == 0.0

    def test_battery_creation_with_advanced_parameters(self):
        """Test creation of battery with advanced parameters."""
        battery = Battery(
            asset_id="battery_002",
            name="Advanced Battery",
            node_id="node_2",
            capacity_mw=50.0,
            energy_capacity_mwh=200.0,
            initial_soc_percent=80.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.92,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            nominal_voltage_v=800.0,
            current_temperature_c=25.0,
        )
        assert battery.charge_efficiency == 0.95
        assert battery.discharge_efficiency == 0.92
        assert battery.min_soc_percent == 10.0
        assert battery.max_soc_percent == 90.0
        assert battery.nominal_voltage_v == 800.0
        assert battery.current_temperature_c == 25.0

    def test_battery_validation_constraints(self):
        """Test battery validation constraints."""
        # Test negative capacity
        with pytest.raises(ValidationError):
            Battery(
                asset_id="invalid_battery",
                name="Invalid Battery",
                node_id="node_1",
                capacity_mw=-10.0,
                energy_capacity_mwh=40.0,
            )

        # Test negative energy capacity
        with pytest.raises(ValidationError):
            Battery(
                asset_id="invalid_battery",
                name="Invalid Battery",
                node_id="node_1",
                capacity_mw=10.0,
                energy_capacity_mwh=-40.0,
            )

        # Test invalid SoC range
        with pytest.raises(ValidationError):
            Battery(
                asset_id="invalid_battery",
                name="Invalid Battery",
                node_id="node_1",
                capacity_mw=10.0,
                energy_capacity_mwh=40.0,
                initial_soc_percent=110.0,  # > 100%
            )

        # Test invalid efficiency
        with pytest.raises(ValidationError):
            Battery(
                asset_id="invalid_battery",
                name="Invalid Battery",
                node_id="node_1",
                capacity_mw=10.0,
                energy_capacity_mwh=40.0,
                charge_efficiency=1.5,  # > 1.0
            )

        # Test invalid SoC limits
        with pytest.raises(ValidationError):
            Battery(
                asset_id="invalid_battery",
                name="Invalid Battery",
                node_id="node_1",
                capacity_mw=10.0,
                energy_capacity_mwh=40.0,
                min_soc_percent=60.0,
                max_soc_percent=40.0,  # min > max
            )


class TestBatteryStateOfCharge:
    """Test Battery State of Charge (SoC) management."""

    def test_soc_initialization(self):
        """Test SoC initialization."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=75.0,
        )
        assert battery.current_soc_percent == 75.0
        assert battery.get_stored_energy_mwh() == 30.0  # 75% of 40 MWh

    def test_soc_updates_with_charging(self):
        """Test SoC updates during charging."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Charge at 5 MW for 2 hours (10 MWh)
        battery.set_power_setpoint(5.0)  # Positive for charging
        battery.tick(7200.0)  # 2 hours in seconds

        # SoC should increase (accounting for efficiency)
        expected_energy_added = 10.0 * battery.charge_efficiency
        expected_soc = 50.0 + (expected_energy_added / 40.0) * 100.0
        assert abs(battery.current_soc_percent - expected_soc) < 1.0

    def test_soc_updates_with_discharging(self):
        """Test SoC updates during discharging."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=80.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Discharge at 8 MW for 1 hour (8 MWh)
        battery.set_power_setpoint(-8.0)  # Negative for discharging
        battery.tick(3600.0)  # 1 hour in seconds

        # SoC should decrease (accounting for efficiency)
        expected_energy_removed = 8.0 / battery.discharge_efficiency
        expected_soc = 80.0 - (expected_energy_removed / 40.0) * 100.0
        assert abs(battery.current_soc_percent - expected_soc) < 1.0

    def test_soc_limits_enforcement(self):
        """Test SoC limits enforcement."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=15.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Try to discharge below minimum SoC
        battery.set_power_setpoint(-10.0)
        battery.tick(3600.0)  # 1 hour

        # SoC should not go below minimum
        assert battery.current_soc_percent >= battery.min_soc_percent

        # Reset to high SoC
        battery.current_soc_percent = 85.0

        # Try to charge above maximum SoC
        battery.set_power_setpoint(10.0)
        battery.tick(3600.0)  # 1 hour

        # SoC should not exceed maximum
        assert battery.current_soc_percent <= battery.max_soc_percent


class TestBatteryEfficiency:
    """Test Battery charge/discharge efficiency."""

    def test_charge_efficiency_basic(self):
        """Test basic charge efficiency."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
            charge_efficiency=0.90,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test charging power calculation
        battery.set_power_setpoint(10.0)
        actual_power = battery.calculate_power_output()
        assert actual_power == 10.0  # Charging power input

        # Test energy efficiency in tick
        initial_energy = battery.get_stored_energy_mwh()
        battery.tick(3600.0)  # 1 hour
        final_energy = battery.get_stored_energy_mwh()

        # Energy added should account for efficiency
        energy_added = final_energy - initial_energy
        expected_energy_added = 10.0 * 0.90  # 90% efficiency
        assert abs(energy_added - expected_energy_added) < 0.1

    def test_discharge_efficiency_basic(self):
        """Test basic discharge efficiency."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=80.0,
            discharge_efficiency=0.92,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test discharging power calculation
        battery.set_power_setpoint(-8.0)
        actual_power = battery.calculate_power_output()
        assert actual_power == -8.0  # Discharging power output

        # Test energy efficiency in tick
        initial_energy = battery.get_stored_energy_mwh()
        battery.tick(3600.0)  # 1 hour
        final_energy = battery.get_stored_energy_mwh()

        # Energy removed should account for efficiency
        energy_removed = initial_energy - final_energy
        expected_energy_removed = 8.0 / 0.92  # Accounting for discharge efficiency
        assert abs(energy_removed - expected_energy_removed) < 0.1

    def test_efficiency_temperature_dependence(self):
        """Test efficiency temperature dependence."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.92,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test at normal temperature
        battery.set_temperature(25.0)
        normal_charge_eff = battery.get_current_charge_efficiency()
        normal_discharge_eff = battery.get_current_discharge_efficiency()

        # Test at cold temperature
        battery.set_temperature(-10.0)
        cold_charge_eff = battery.get_current_charge_efficiency()
        cold_discharge_eff = battery.get_current_discharge_efficiency()

        # Efficiency should decrease at cold temperatures
        assert cold_charge_eff < normal_charge_eff
        assert cold_discharge_eff < normal_discharge_eff

        # Test at hot temperature
        battery.set_temperature(45.0)
        hot_charge_eff = battery.get_current_charge_efficiency()
        hot_discharge_eff = battery.get_current_discharge_efficiency()

        # Efficiency should decrease at hot temperatures
        assert hot_charge_eff < normal_charge_eff
        assert hot_discharge_eff < normal_discharge_eff


class TestBatteryVoltageSensing:
    """Test Battery voltage sensing functionality."""

    def test_voltage_soc_relationship(self):
        """Test voltage-SoC relationship."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=100.0,
            nominal_voltage_v=800.0,
        )

        # Test at various SoC levels
        soc_levels = [100.0, 80.0, 60.0, 40.0, 20.0, 10.0]
        voltages = []

        for soc in soc_levels:
            battery.current_soc_percent = soc
            voltage = battery.get_terminal_voltage()
            voltages.append(voltage)

        # Voltage should generally decrease with SoC
        for i in range(len(voltages) - 1):
            assert voltages[i] >= voltages[i + 1], f"Voltage should decrease with SoC: {voltages}"

    def test_voltage_load_dependence(self):
        """Test voltage dependence on load."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
            nominal_voltage_v=800.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test no load voltage
        battery.set_power_setpoint(0.0)
        no_load_voltage = battery.get_terminal_voltage()

        # Test discharge load voltage
        battery.set_power_setpoint(-8.0)
        discharge_voltage = battery.get_terminal_voltage()

        # Test charge load voltage
        battery.set_power_setpoint(8.0)
        charge_voltage = battery.get_terminal_voltage()

        # Voltage should drop under discharge load
        assert discharge_voltage < no_load_voltage
        # Voltage should increase under charge load
        assert charge_voltage > no_load_voltage

    def test_voltage_temperature_effects(self):
        """Test voltage temperature effects."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
            nominal_voltage_v=800.0,
            current_temperature_c=25.0,
        )

        # Test at normal temperature
        normal_voltage = battery.get_terminal_voltage()

        # Test at cold temperature
        battery.set_temperature(-10.0)
        cold_voltage = battery.get_terminal_voltage()

        # Test at hot temperature
        battery.set_temperature(45.0)
        hot_voltage = battery.get_terminal_voltage()

        # Voltage should have temperature dependence
        assert abs(cold_voltage - normal_voltage) > 0.01
        assert abs(hot_voltage - normal_voltage) > 0.01


class TestBatteryPowerLimits:
    """Test Battery power limit management."""

    def test_power_limits_soc_dependence(self):
        """Test power limits dependence on SoC."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test at various SoC levels
        test_socs = [10.0, 30.0, 50.0, 70.0, 90.0]

        for soc in test_socs:
            battery.current_soc_percent = soc
            max_charge = battery.get_max_charge_power()
            max_discharge = battery.get_max_discharge_power()

            # Power limits should be affected by SoC
            assert 0.0 <= max_charge <= battery.capacity_mw
            assert 0.0 <= max_discharge <= battery.capacity_mw

        # At minimum SoC, discharge power should be limited
        battery.current_soc_percent = 10.0
        min_soc_discharge = battery.get_max_discharge_power()

        # At maximum SoC, charge power should be limited
        battery.current_soc_percent = 90.0
        max_soc_charge = battery.get_max_charge_power()

        # Should have some power limitation at extremes
        assert min_soc_discharge < battery.capacity_mw
        assert max_soc_charge < battery.capacity_mw

    def test_power_limits_temperature_dependence(self):
        """Test power limits dependence on temperature."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test at normal temperature
        battery.set_temperature(25.0)
        normal_charge = battery.get_max_charge_power()
        normal_discharge = battery.get_max_discharge_power()

        # Test at cold temperature
        battery.set_temperature(-10.0)
        cold_charge = battery.get_max_charge_power()
        cold_discharge = battery.get_max_discharge_power()

        # Power limits should be reduced at cold temperatures
        assert cold_charge <= normal_charge
        assert cold_discharge <= normal_discharge

        # Test at hot temperature
        battery.set_temperature(45.0)
        hot_charge = battery.get_max_charge_power()
        hot_discharge = battery.get_max_discharge_power()

        # Power limits should be reduced at hot temperatures
        assert hot_charge <= normal_charge
        assert hot_discharge <= normal_discharge

    def test_power_setpoint_enforcement(self):
        """Test power setpoint enforcement within limits."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test setpoint within limits
        battery.set_power_setpoint(5.0)
        assert battery.power_setpoint_mw == 5.0

        # Test setpoint exceeding charge limit
        battery.set_power_setpoint(15.0)  # Exceeds capacity
        assert battery.power_setpoint_mw <= battery.capacity_mw

        # Test setpoint exceeding discharge limit
        battery.set_power_setpoint(-15.0)  # Exceeds capacity
        assert battery.power_setpoint_mw >= -battery.capacity_mw


class TestBatteryThermalModel:
    """Test Battery thermal modeling."""

    def test_thermal_initialization(self):
        """Test thermal model initialization."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            current_temperature_c=25.0,
            ambient_temperature_c=20.0,
        )

        assert battery.current_temperature_c == 25.0
        assert battery.ambient_temperature_c == 20.0

    def test_thermal_response_to_power(self):
        """Test thermal response to power charging/discharging."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            current_temperature_c=25.0,
            ambient_temperature_c=20.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test thermal response to high power discharge
        battery.set_power_setpoint(-8.0)
        initial_temp = battery.current_temperature_c
        battery.tick(3600.0)  # 1 hour
        final_temp = battery.current_temperature_c

        # Temperature should increase due to power losses
        assert final_temp > initial_temp

    def test_thermal_cooling(self):
        """Test thermal cooling when idle."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            current_temperature_c=40.0,
            ambient_temperature_c=20.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test cooling when idle
        battery.set_power_setpoint(0.0)
        initial_temp = battery.current_temperature_c
        battery.tick(3600.0)  # 1 hour
        final_temp = battery.current_temperature_c

        # Temperature should decrease toward ambient
        assert final_temp < initial_temp
        assert final_temp > battery.ambient_temperature_c


class TestBatteryDegradation:
    """Test Battery degradation modeling."""

    def test_degradation_initialization(self):
        """Test degradation model initialization."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_health_percent=100.0,
        )

        assert battery.current_health_percent == 100.0
        assert battery.get_effective_capacity_mwh() == 40.0

    def test_degradation_cycle_tracking(self):
        """Test degradation cycle tracking."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
            initial_health_percent=100.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Simulate charge/discharge cycles
        initial_health = battery.current_health_percent
        initial_cycles = battery.get_cycle_count()

        # Perform multiple charge/discharge cycles
        for _ in range(10):
            # Charge
            battery.set_power_setpoint(10.0)
            battery.tick(1800.0)  # 30 minutes

            # Discharge
            battery.set_power_setpoint(-10.0)
            battery.tick(1800.0)  # 30 minutes

        final_health = battery.current_health_percent
        final_cycles = battery.get_cycle_count()

        # Health should decrease (slightly)
        assert final_health <= initial_health
        # Cycle count should increase
        assert final_cycles > initial_cycles

    def test_degradation_capacity_impact(self):
        """Test degradation impact on capacity."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_health_percent=80.0,  # Already degraded
        )

        # Effective capacity should be reduced
        effective_capacity = battery.get_effective_capacity_mwh()
        assert effective_capacity < 40.0
        assert effective_capacity == 40.0 * 0.80  # 80% health


class TestBatterySimulationIntegration:
    """Test Battery integration with simulation engine."""

    def test_battery_tick_method(self):
        """Test battery tick method."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        battery.set_status(AssetStatus.ONLINE)
        battery.set_power_setpoint(5.0)

        # Test tick method
        initial_soc = battery.current_soc_percent
        power_change = battery.tick(900.0)  # 15 minutes

        # SoC should change
        assert battery.current_soc_percent != initial_soc
        # Power change should be reported
        assert power_change is not None

    def test_battery_state_interface(self):
        """Test battery state interface."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=75.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        state = battery.get_state()

        # Check standard asset state
        assert "asset_id" in state
        assert "capacity_mw" in state
        assert "current_output_mw" in state
        assert "status" in state
        assert state["is_storage"] is True

        # Check battery-specific state
        assert "current_soc_percent" in state
        assert "stored_energy_mwh" in state
        assert "terminal_voltage_v" in state
        assert "current_temperature_c" in state
        assert "current_health_percent" in state
        assert "charge_efficiency" in state
        assert "discharge_efficiency" in state
        assert "max_charge_power_mw" in state
        assert "max_discharge_power_mw" in state

    def test_battery_offline_behavior(self):
        """Test battery behavior when offline."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        # Keep offline
        battery.set_power_setpoint(5.0)

        # Power output should be 0
        power = battery.calculate_power_output()
        assert power == 0.0

        # Tick should not change SoC
        initial_soc = battery.current_soc_percent
        battery.tick(900.0)
        assert battery.current_soc_percent == initial_soc

    def test_battery_reset_method(self):
        """Test battery reset method."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )

        # Change battery state
        battery.set_status(AssetStatus.ONLINE)
        battery.set_power_setpoint(5.0)
        battery.current_soc_percent = 80.0

        # Reset battery
        battery.reset()

        # Check that battery is reset
        assert battery.status == AssetStatus.OFFLINE
        assert battery.current_output_mw == 0.0
        assert battery.power_setpoint_mw == 0.0
        # SoC should reset to initial value
        assert battery.current_soc_percent == 50.0


class TestBatteryEdgeCases:
    """Test Battery edge cases and error conditions."""

    def test_zero_capacity_battery(self):
        """Test battery with zero capacity."""
        with pytest.raises(ValidationError):
            Battery(
                asset_id="zero_battery",
                name="Zero Battery",
                node_id="node_1",
                capacity_mw=0.0,
                energy_capacity_mwh=40.0,
            )

    def test_rapid_power_changes(self):
        """Test rapid power setpoint changes."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test rapid changes
        power_levels = [5.0, -8.0, 10.0, -3.0, 0.0]
        for power in power_levels:
            battery.set_power_setpoint(power)
            battery.tick(60.0)  # 1 minute

            # Battery should handle rapid changes
            assert battery.current_soc_percent >= 0.0
            assert battery.current_soc_percent <= 100.0

    def test_extreme_temperature_conditions(self):
        """Test battery behavior at extreme temperatures."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Test extremely cold temperature
        battery.set_temperature(-40.0)
        cold_power = battery.get_max_charge_power()
        assert cold_power >= 0.0  # Should not be negative

        # Test extremely hot temperature
        battery.set_temperature(60.0)
        hot_power = battery.get_max_discharge_power()
        assert hot_power >= 0.0  # Should not be negative

    def test_long_duration_simulation(self):
        """Test battery behavior over long simulation periods."""
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Simulate 24 hours of operation
        battery.set_power_setpoint(2.0)  # Light charging
        initial_health = battery.current_health_percent

        for _ in range(24):  # 24 hours
            battery.tick(3600.0)  # 1 hour

        # Battery should still be functional
        assert battery.current_soc_percent >= 0.0
        assert battery.current_soc_percent <= 100.0
        assert battery.current_health_percent <= initial_health  # May degrade slightly
        assert battery.current_health_percent > 0.0  # Should not fail completely
