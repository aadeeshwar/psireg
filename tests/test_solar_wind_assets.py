"""Tests for SolarPanel and WindTurbine renewable energy assets.

This module contains comprehensive tests for solar and wind energy assets,
including power curves, curtailment logic, and environmental response.
"""

import pytest
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.assets.wind import WindTurbine
from psireg.utils.enums import AssetStatus, AssetType, WeatherCondition
from pydantic import ValidationError


class TestSolarPanel:
    """Test SolarPanel asset functionality."""

    def test_solar_panel_creation(self):
        """Test creation of solar panel asset."""
        panel = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
            tilt_degrees=25.0,
            azimuth_degrees=180.0,
        )
        assert panel.asset_id == "solar_001"
        assert panel.asset_type == AssetType.SOLAR
        assert panel.name == "Test Solar Panel"
        assert panel.node_id == "node_1"
        assert panel.capacity_mw == 100.0
        assert panel.panel_efficiency == 0.20
        assert panel.panel_area_m2 == 50000.0
        assert panel.tilt_degrees == 25.0
        assert panel.azimuth_degrees == 180.0
        assert panel.current_irradiance_w_m2 == 0.0
        assert panel.current_temperature_c == 25.0
        assert panel.curtailment_factor == 1.0
        assert panel.status == AssetStatus.OFFLINE

    def test_solar_panel_validation(self):
        """Test solar panel validation."""
        # Test invalid efficiency
        with pytest.raises(ValidationError):
            SolarPanel(
                asset_id="solar_001",
                name="Invalid Panel",
                node_id="node_1",
                capacity_mw=100.0,
                panel_efficiency=1.5,  # Invalid > 1.0
                panel_area_m2=50000.0,
            )

        # Test invalid area
        with pytest.raises(ValidationError):
            SolarPanel(
                asset_id="solar_001",
                name="Invalid Panel",
                node_id="node_1",
                capacity_mw=100.0,
                panel_efficiency=0.20,
                panel_area_m2=-1000.0,  # Invalid negative
            )

        # Test invalid tilt
        with pytest.raises(ValidationError):
            SolarPanel(
                asset_id="solar_001",
                name="Invalid Panel",
                node_id="node_1",
                capacity_mw=100.0,
                panel_efficiency=0.20,
                panel_area_m2=50000.0,
                tilt_degrees=95.0,  # Invalid > 90
            )

    def test_solar_irradiance_response(self):
        """Test solar panel power response to irradiance."""
        panel = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
        )
        panel.set_status(AssetStatus.ONLINE)

        # Test with no irradiance
        panel.set_irradiance(0.0)
        power = panel.calculate_power_output()
        assert power == 0.0

        # Test with 500 W/m2 irradiance
        panel.set_irradiance(500.0)
        power = panel.calculate_power_output()
        assert power > 0.0
        assert power < panel.capacity_mw

        # Test with peak irradiance (1000 W/m2)
        panel.set_irradiance(1000.0)
        power = panel.calculate_power_output()
        assert power > 0.0
        # Should be close to theoretical max but consider temperature derating
        expected_theoretical = panel.panel_area_m2 * 1000.0 * panel.panel_efficiency / 1_000_000
        assert abs(power - expected_theoretical) < expected_theoretical * 0.2

    def test_solar_temperature_derating(self):
        """Test solar panel temperature derating."""
        panel = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
        )
        panel.set_status(AssetStatus.ONLINE)
        panel.set_irradiance(1000.0)

        # Test at standard test conditions (25°C)
        panel.set_temperature(25.0)
        power_25c = panel.calculate_power_output()

        # Test at higher temperature (50°C)
        panel.set_temperature(50.0)
        power_50c = panel.calculate_power_output()

        # Power should decrease with higher temperature
        assert power_50c < power_25c

        # Test at lower temperature (0°C)
        panel.set_temperature(0.0)
        power_0c = panel.calculate_power_output()

        # Power should increase with lower temperature
        assert power_0c > power_25c

    def test_solar_curtailment_logic(self):
        """Test solar panel curtailment logic."""
        panel = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
        )
        panel.set_status(AssetStatus.ONLINE)
        panel.set_irradiance(1000.0)

        # Test no curtailment
        panel.set_curtailment_factor(1.0)
        power_no_curtail = panel.calculate_power_output()

        # Test 50% curtailment
        panel.set_curtailment_factor(0.5)
        power_50_curtail = panel.calculate_power_output()
        assert abs(power_50_curtail - power_no_curtail * 0.5) < 0.1

        # Test full curtailment
        panel.set_curtailment_factor(0.0)
        power_full_curtail = panel.calculate_power_output()
        assert power_full_curtail == 0.0

    def test_solar_weather_condition_response(self):
        """Test solar panel response to weather conditions."""
        panel = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
        )
        panel.set_status(AssetStatus.ONLINE)
        panel.set_irradiance(800.0)

        # Test clear conditions
        panel.set_weather_condition(WeatherCondition.CLEAR)
        power_clear = panel.calculate_power_output()

        # Test cloudy conditions
        panel.set_weather_condition(WeatherCondition.CLOUDY)
        power_cloudy = panel.calculate_power_output()
        assert power_cloudy < power_clear

        # Test partly cloudy
        panel.set_weather_condition(WeatherCondition.PARTLY_CLOUDY)
        power_partly = panel.calculate_power_output()
        assert power_partly > power_cloudy
        assert power_partly < power_clear

    def test_solar_tick_update(self):
        """Test solar panel tick update method."""
        panel = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
        )
        panel.set_status(AssetStatus.ONLINE)
        panel.set_irradiance(600.0)

        # Test tick update
        initial_output = panel.current_output_mw
        power_change = panel.tick(900.0)  # 15 minutes

        # Should update current output
        assert panel.current_output_mw != initial_output
        assert power_change is not None

    def test_solar_offline_behavior(self):
        """Test solar panel behavior when offline."""
        panel = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
        )
        # Keep offline
        panel.set_irradiance(1000.0)

        power = panel.calculate_power_output()
        assert power == 0.0

        # Tick should also return 0
        power_change = panel.tick(900.0)
        assert power_change == 0.0


class TestWindTurbine:
    """Test WindTurbine asset functionality."""

    def test_wind_turbine_creation(self):
        """Test creation of wind turbine asset."""
        turbine = WindTurbine(
            asset_id="wind_001",
            name="Test Wind Turbine",
            node_id="node_1",
            capacity_mw=3.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
            cut_in_speed_ms=3.0,
            cut_out_speed_ms=25.0,
            rated_speed_ms=12.0,
        )
        assert turbine.asset_id == "wind_001"
        assert turbine.asset_type == AssetType.WIND
        assert turbine.name == "Test Wind Turbine"
        assert turbine.node_id == "node_1"
        assert turbine.capacity_mw == 3.0
        assert turbine.rotor_diameter_m == 150.0
        assert turbine.hub_height_m == 120.0
        assert turbine.cut_in_speed_ms == 3.0
        assert turbine.cut_out_speed_ms == 25.0
        assert turbine.rated_speed_ms == 12.0
        assert turbine.current_wind_speed_ms == 0.0
        assert turbine.current_air_density_kg_m3 == 1.225
        assert turbine.curtailment_factor == 1.0
        assert turbine.status == AssetStatus.OFFLINE

    def test_wind_turbine_validation(self):
        """Test wind turbine validation."""
        # Test invalid cut-in speed
        with pytest.raises(ValidationError):
            WindTurbine(
                asset_id="wind_001",
                name="Invalid Turbine",
                node_id="node_1",
                capacity_mw=3.0,
                rotor_diameter_m=150.0,
                hub_height_m=120.0,
                cut_in_speed_ms=-1.0,  # Invalid negative
            )

        # Test invalid speed relationship
        with pytest.raises(ValidationError):
            WindTurbine(
                asset_id="wind_001",
                name="Invalid Turbine",
                node_id="node_1",
                capacity_mw=3.0,
                rotor_diameter_m=150.0,
                hub_height_m=120.0,
                cut_in_speed_ms=15.0,  # Invalid > rated speed
                rated_speed_ms=12.0,
            )

    def test_wind_speed_power_curve(self):
        """Test wind turbine power curve response."""
        turbine = WindTurbine(
            asset_id="wind_001",
            name="Test Wind Turbine",
            node_id="node_1",
            capacity_mw=3.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
            cut_in_speed_ms=3.0,
            cut_out_speed_ms=25.0,
            rated_speed_ms=12.0,
        )
        turbine.set_status(AssetStatus.ONLINE)

        # Test below cut-in speed
        turbine.set_wind_speed(2.0)
        power = turbine.calculate_power_output()
        assert power == 0.0

        # Test at cut-in speed
        turbine.set_wind_speed(3.0)
        power = turbine.calculate_power_output()
        assert power > 0.0

        # Test in power curve region
        turbine.set_wind_speed(8.0)
        power_8ms = turbine.calculate_power_output()
        assert power_8ms > 0.0

        # Test at rated speed
        turbine.set_wind_speed(12.0)
        power_rated = turbine.calculate_power_output()
        assert power_rated > power_8ms

        # Test above rated speed (should be at rated power)
        turbine.set_wind_speed(20.0)
        power_20ms = turbine.calculate_power_output()
        assert abs(power_20ms - power_rated) < 0.1

        # Test at cut-out speed
        turbine.set_wind_speed(25.0)
        power_cutout = turbine.calculate_power_output()
        assert power_cutout == 0.0

    def test_wind_air_density_effect(self):
        """Test wind turbine air density effect."""
        turbine = WindTurbine(
            asset_id="wind_001",
            name="Test Wind Turbine",
            node_id="node_1",
            capacity_mw=3.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
        )
        turbine.set_status(AssetStatus.ONLINE)
        turbine.set_wind_speed(10.0)

        # Test standard air density
        turbine.set_air_density(1.225)
        power_standard = turbine.calculate_power_output()

        # Test higher air density (cold/high pressure)
        turbine.set_air_density(1.3)
        power_high_density = turbine.calculate_power_output()
        assert power_high_density > power_standard

        # Test lower air density (hot/high altitude)
        turbine.set_air_density(1.1)
        power_low_density = turbine.calculate_power_output()
        assert power_low_density < power_standard

    def test_wind_curtailment_logic(self):
        """Test wind turbine curtailment logic."""
        turbine = WindTurbine(
            asset_id="wind_001",
            name="Test Wind Turbine",
            node_id="node_1",
            capacity_mw=3.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
        )
        turbine.set_status(AssetStatus.ONLINE)
        turbine.set_wind_speed(15.0)

        # Test no curtailment
        turbine.set_curtailment_factor(1.0)
        power_no_curtail = turbine.calculate_power_output()

        # Test 30% curtailment
        turbine.set_curtailment_factor(0.7)
        power_30_curtail = turbine.calculate_power_output()
        assert abs(power_30_curtail - power_no_curtail * 0.7) < 0.1

        # Test full curtailment
        turbine.set_curtailment_factor(0.0)
        power_full_curtail = turbine.calculate_power_output()
        assert power_full_curtail == 0.0

    def test_wind_weather_condition_response(self):
        """Test wind turbine response to weather conditions."""
        turbine = WindTurbine(
            asset_id="wind_001",
            name="Test Wind Turbine",
            node_id="node_1",
            capacity_mw=3.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
        )
        turbine.set_status(AssetStatus.ONLINE)
        turbine.set_wind_speed(10.0)

        # Test windy conditions
        turbine.set_weather_condition(WeatherCondition.WINDY)
        power_windy = turbine.calculate_power_output()

        # Test clear conditions
        turbine.set_weather_condition(WeatherCondition.CLEAR)
        power_clear = turbine.calculate_power_output()
        assert power_windy >= power_clear

        # Test stormy conditions (should shut down)
        turbine.set_weather_condition(WeatherCondition.STORMY)
        power_stormy = turbine.calculate_power_output()
        assert power_stormy == 0.0

    def test_wind_tick_update(self):
        """Test wind turbine tick update method."""
        turbine = WindTurbine(
            asset_id="wind_001",
            name="Test Wind Turbine",
            node_id="node_1",
            capacity_mw=3.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
        )
        turbine.set_status(AssetStatus.ONLINE)
        turbine.set_wind_speed(8.0)

        # Test tick update
        initial_output = turbine.current_output_mw
        power_change = turbine.tick(900.0)  # 15 minutes

        # Should update current output
        assert turbine.current_output_mw != initial_output
        assert power_change is not None

    def test_wind_offline_behavior(self):
        """Test wind turbine behavior when offline."""
        turbine = WindTurbine(
            asset_id="wind_001",
            name="Test Wind Turbine",
            node_id="node_1",
            capacity_mw=3.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
        )
        # Keep offline
        turbine.set_wind_speed(15.0)

        power = turbine.calculate_power_output()
        assert power == 0.0

        # Tick should also return 0
        power_change = turbine.tick(900.0)
        assert power_change == 0.0


class TestRenewableIntegration:
    """Test renewable asset integration scenarios."""

    def test_solar_wind_capacity_factors(self):
        """Test realistic capacity factors for solar and wind."""
        # Solar panel
        solar = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
        )
        solar.set_status(AssetStatus.ONLINE)
        solar.set_irradiance(500.0)  # Moderate irradiance

        solar_power = solar.calculate_power_output()
        solar_capacity_factor = solar_power / solar.capacity_mw
        assert 0.0 <= solar_capacity_factor <= 1.0

        # Wind turbine
        wind = WindTurbine(
            asset_id="wind_001",
            name="Test Wind Turbine",
            node_id="node_1",
            capacity_mw=3.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
        )
        wind.set_status(AssetStatus.ONLINE)
        wind.set_wind_speed(8.0)  # Good wind

        wind_power = wind.calculate_power_output()
        wind_capacity_factor = wind_power / wind.capacity_mw
        assert 0.0 <= wind_capacity_factor <= 1.0

    def test_renewable_curtailment_scenarios(self):
        """Test grid-wide renewable curtailment scenarios."""
        # Create multiple renewable assets
        solar = SolarPanel(
            asset_id="solar_001",
            name="Solar Farm",
            node_id="node_1",
            capacity_mw=200.0,
            panel_efficiency=0.22,
            panel_area_m2=100000.0,
        )

        wind = WindTurbine(
            asset_id="wind_001",
            name="Wind Farm",
            node_id="node_2",
            capacity_mw=150.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
        )

        # Set favorable conditions
        solar.set_status(AssetStatus.ONLINE)
        solar.set_irradiance(1000.0)

        wind.set_status(AssetStatus.ONLINE)
        wind.set_wind_speed(12.0)

        # Test coordinated curtailment
        curtailment_factor = 0.6
        solar.set_curtailment_factor(curtailment_factor)
        wind.set_curtailment_factor(curtailment_factor)

        solar_power = solar.calculate_power_output()
        wind_power = wind.calculate_power_output()

        # Both should be curtailed
        assert solar_power < solar.capacity_mw * curtailment_factor * 1.1
        assert wind_power < wind.capacity_mw * curtailment_factor * 1.1

    def test_renewable_state_interface(self):
        """Test uniform state interface for renewable assets."""
        solar = SolarPanel(
            asset_id="solar_001",
            name="Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
        )

        wind = WindTurbine(
            asset_id="wind_001",
            name="Wind Turbine",
            node_id="node_1",
            capacity_mw=3.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
        )

        # Test state interface
        solar_state = solar.get_state()
        wind_state = wind.get_state()

        # Both should have renewable-specific state
        assert "current_irradiance_w_m2" in solar_state
        assert "current_temperature_c" in solar_state
        assert "curtailment_factor" in solar_state

        assert "current_wind_speed_ms" in wind_state
        assert "current_air_density_kg_m3" in wind_state
        assert "curtailment_factor" in wind_state

        # Both should have standard asset state
        for state in [solar_state, wind_state]:
            assert "asset_id" in state
            assert "capacity_mw" in state
            assert "current_output_mw" in state
            assert "is_renewable" in state
            assert state["is_renewable"] is True
