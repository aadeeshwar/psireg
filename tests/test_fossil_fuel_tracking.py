"""Tests for fossil fuel tracking and conventional generator integration."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from psireg.sim.assets.base import AssetStatus, AssetType
from psireg.sim.assets.thermal import CoalPlant, NaturalGasPlant, NuclearPlant
from psireg.sim.metrics import FossilFuelCalculator, MetricsCollector


class TestConventionalGeneratorAssets:
    """Test conventional generator asset types."""

    def test_coal_plant_initialization(self):
        """Test coal plant initialization and basic properties."""
        coal_plant = CoalPlant(
            asset_id="coal_001",
            name="Coal Plant 1",
            node_id="thermal_node",
            capacity_mw=300.0,
            efficiency=0.35,
            min_load_mw=100.0,
            ramp_rate_mw_per_min=5.0,
            heat_rate_btu_per_kwh=9500.0,
            fuel_cost_per_mmbtu=2.5,
        )

        assert coal_plant.asset_id == "coal_001"
        assert coal_plant.asset_type == AssetType.COAL
        assert coal_plant.capacity_mw == 300.0
        assert coal_plant.efficiency == 0.35
        assert coal_plant.min_load_mw == 100.0
        assert coal_plant.ramp_rate_mw_per_min == 5.0
        assert coal_plant.heat_rate_btu_per_kwh == 9500.0
        assert coal_plant.fuel_cost_per_mmbtu == 2.5
        assert coal_plant.is_fossil_fuel() is True
        assert coal_plant.is_renewable() is False

    def test_natural_gas_plant_initialization(self):
        """Test natural gas plant initialization and basic properties."""
        gas_plant = NaturalGasPlant(
            asset_id="gas_001",
            name="Gas Plant 1",
            node_id="thermal_node",
            capacity_mw=200.0,
            efficiency=0.50,
            min_load_mw=40.0,
            ramp_rate_mw_per_min=10.0,
            heat_rate_btu_per_kwh=7000.0,
            fuel_cost_per_mmbtu=4.0,
        )

        assert gas_plant.asset_id == "gas_001"
        assert gas_plant.asset_type == AssetType.GAS
        assert gas_plant.capacity_mw == 200.0
        assert gas_plant.efficiency == 0.50
        assert gas_plant.min_load_mw == 40.0
        assert gas_plant.ramp_rate_mw_per_min == 10.0
        assert gas_plant.heat_rate_btu_per_kwh == 7000.0
        assert gas_plant.fuel_cost_per_mmbtu == 4.0
        assert gas_plant.is_fossil_fuel() is True
        assert gas_plant.is_renewable() is False

    def test_nuclear_plant_initialization(self):
        """Test nuclear plant initialization and basic properties."""
        nuclear_plant = NuclearPlant(
            asset_id="nuclear_001",
            name="Nuclear Plant 1",
            node_id="nuclear_node",
            capacity_mw=1000.0,
            efficiency=0.33,
            min_load_mw=700.0,
            ramp_rate_mw_per_min=1.0,
            fuel_cost_per_mwh=5.0,
        )

        assert nuclear_plant.asset_id == "nuclear_001"
        assert nuclear_plant.asset_type == AssetType.NUCLEAR
        assert nuclear_plant.capacity_mw == 1000.0
        assert nuclear_plant.efficiency == 0.33
        assert nuclear_plant.min_load_mw == 700.0
        assert nuclear_plant.ramp_rate_mw_per_min == 1.0
        assert nuclear_plant.fuel_cost_per_mwh == 5.0
        assert nuclear_plant.is_fossil_fuel() is False  # Nuclear is not fossil fuel
        assert nuclear_plant.is_renewable() is False

    def test_coal_plant_power_calculation(self):
        """Test coal plant power output calculation."""
        coal_plant = CoalPlant(
            asset_id="coal_001",
            name="Coal Plant 1",
            node_id="thermal_node",
            capacity_mw=300.0,
            efficiency=0.35,
            min_load_mw=100.0,
            ramp_rate_mw_per_min=5.0,
            heat_rate_btu_per_kwh=9500.0,
            fuel_cost_per_mmbtu=2.5,
        )

        # Test offline state
        coal_plant.set_status(AssetStatus.OFFLINE)
        assert coal_plant.calculate_power_output() == 0.0

        # Test online state
        coal_plant.set_status(AssetStatus.ONLINE)
        coal_plant.set_load_setpoint(200.0)

        power_output = coal_plant.calculate_power_output()
        assert power_output >= coal_plant.min_load_mw
        assert power_output <= coal_plant.capacity_mw
        assert power_output == 200.0

    def test_gas_plant_ramping_constraints(self):
        """Test natural gas plant ramping constraints."""
        gas_plant = NaturalGasPlant(
            asset_id="gas_001",
            name="Gas Plant 1",
            node_id="thermal_node",
            capacity_mw=200.0,
            efficiency=0.50,
            min_load_mw=40.0,
            ramp_rate_mw_per_min=10.0,
            heat_rate_btu_per_kwh=7000.0,
            fuel_cost_per_mmbtu=4.0,
        )

        gas_plant.set_status(AssetStatus.ONLINE)
        gas_plant.current_output_mw = 100.0

        # Test ramping up
        gas_plant.set_load_setpoint(150.0)
        gas_plant.tick(60.0)  # 1 minute

        # Should ramp up by max 10 MW/min
        assert gas_plant.current_output_mw == 110.0

        # Test ramping down
        gas_plant.set_load_setpoint(80.0)
        gas_plant.tick(60.0)  # 1 minute

        # Should ramp down by max 10 MW/min
        assert gas_plant.current_output_mw == 100.0

    def test_nuclear_plant_baseload_operation(self):
        """Test nuclear plant baseload operation characteristics."""
        nuclear_plant = NuclearPlant(
            asset_id="nuclear_001",
            name="Nuclear Plant 1",
            node_id="nuclear_node",
            capacity_mw=1000.0,
            efficiency=0.33,
            min_load_mw=700.0,
            ramp_rate_mw_per_min=1.0,
            fuel_cost_per_mwh=5.0,
        )

        nuclear_plant.set_status(AssetStatus.ONLINE)

        # Nuclear plants typically run at constant output
        nuclear_plant.set_load_setpoint(900.0)
        power_output = nuclear_plant.calculate_power_output()

        assert power_output >= nuclear_plant.min_load_mw
        assert power_output <= nuclear_plant.capacity_mw
        assert power_output == 900.0

    def test_thermal_plant_fuel_cost_calculation(self):
        """Test fuel cost calculation for thermal plants."""
        coal_plant = CoalPlant(
            asset_id="coal_001",
            name="Coal Plant 1",
            node_id="thermal_node",
            capacity_mw=300.0,
            efficiency=0.35,
            min_load_mw=100.0,
            ramp_rate_mw_per_min=5.0,
            heat_rate_btu_per_kwh=9500.0,
            fuel_cost_per_mmbtu=2.5,
        )

        coal_plant.set_status(AssetStatus.ONLINE)
        coal_plant.set_load_setpoint(200.0)

        # Update plant state - allow enough time for ramping (40 minutes to go from 0 to 200 MW at 5 MW/min)
        coal_plant.tick(40 * 60.0)  # 40 minutes simulation step

        # Calculate fuel cost
        fuel_cost = coal_plant.calculate_fuel_cost_per_hour()

        # Expected: 200 MW * 1000 kW/MW * 9500 BTU/kWh * $2.5/MMBTU / 1,000,000 BTU/MMBTU
        expected_cost = 200 * 1000 * 9500 * 2.5 / 1_000_000

        assert abs(fuel_cost - expected_cost) < 0.01

    def test_thermal_plant_emissions_calculation(self):
        """Test emissions calculation for thermal plants."""
        coal_plant = CoalPlant(
            asset_id="coal_001",
            name="Coal Plant 1",
            node_id="thermal_node",
            capacity_mw=300.0,
            efficiency=0.35,
            min_load_mw=100.0,
            ramp_rate_mw_per_min=5.0,
            heat_rate_btu_per_kwh=9500.0,
            fuel_cost_per_mmbtu=2.5,
            co2_emissions_lb_per_mmbtu=205.0,  # Coal emissions factor
        )

        coal_plant.set_status(AssetStatus.ONLINE)
        coal_plant.set_load_setpoint(200.0)

        # Update plant state - allow enough time for ramping (40 minutes to go from 0 to 200 MW at 5 MW/min)
        coal_plant.tick(40 * 60.0)  # 40 minutes simulation step

        # Calculate CO2 emissions
        emissions_lb_per_hour = coal_plant.calculate_co2_emissions_per_hour()

        # Expected: 200 MW * 1000 kW/MW * 9500 BTU/kWh * 205 lb/MMBTU / 1,000,000 BTU/MMBTU
        expected_emissions = 200 * 1000 * 9500 * 205 / 1_000_000

        assert abs(emissions_lb_per_hour - expected_emissions) < 0.01


class TestFossilFuelCalculator:
    """Test fossil fuel percentage calculation."""

    def test_fossil_fuel_calculator_initialization(self):
        """Test FossilFuelCalculator initialization."""
        calculator = FossilFuelCalculator()

        assert calculator.fossil_fuel_types == [AssetType.COAL, AssetType.GAS]
        assert calculator.renewable_types == [AssetType.SOLAR, AssetType.WIND, AssetType.HYDRO]
        assert calculator.non_fossil_types == [AssetType.NUCLEAR, AssetType.SOLAR, AssetType.WIND, AssetType.HYDRO]

    def test_calculate_fossil_fuel_percentage_all_renewable(self):
        """Test fossil fuel percentage calculation with all renewable generation."""
        calculator = FossilFuelCalculator()

        # Create mock assets
        solar_asset = Mock()
        solar_asset.asset_type = AssetType.SOLAR
        solar_asset.current_output_mw = 100.0
        solar_asset.is_online.return_value = True

        wind_asset = Mock()
        wind_asset.asset_type = AssetType.WIND
        wind_asset.current_output_mw = 80.0
        wind_asset.is_online.return_value = True

        assets = [solar_asset, wind_asset]

        result = calculator.calculate_fossil_fuel_percentage(assets)

        assert result["fossil_fuel_percentage"] == 0.0
        assert result["total_generation_mw"] == 180.0
        assert result["fossil_fuel_generation_mw"] == 0.0
        assert result["renewable_generation_mw"] == 180.0

    def test_calculate_fossil_fuel_percentage_mixed_generation(self):
        """Test fossil fuel percentage calculation with mixed generation."""
        calculator = FossilFuelCalculator()

        # Create mock assets
        solar_asset = Mock()
        solar_asset.asset_type = AssetType.SOLAR
        solar_asset.current_output_mw = 100.0
        solar_asset.is_online.return_value = True

        coal_asset = Mock()
        coal_asset.asset_type = AssetType.COAL
        coal_asset.current_output_mw = 200.0
        coal_asset.is_online.return_value = True

        gas_asset = Mock()
        gas_asset.asset_type = AssetType.GAS
        gas_asset.current_output_mw = 150.0
        gas_asset.is_online.return_value = True

        assets = [solar_asset, coal_asset, gas_asset]

        result = calculator.calculate_fossil_fuel_percentage(assets)

        # Total: 450 MW, Fossil: 350 MW, Percentage: 77.78%
        assert abs(result["fossil_fuel_percentage"] - 77.78) < 0.01
        assert result["total_generation_mw"] == 450.0
        assert result["fossil_fuel_generation_mw"] == 350.0
        assert result["renewable_generation_mw"] == 100.0

    def test_calculate_fossil_fuel_percentage_with_nuclear(self):
        """Test fossil fuel percentage calculation including nuclear."""
        calculator = FossilFuelCalculator()

        # Create mock assets
        nuclear_asset = Mock()
        nuclear_asset.asset_type = AssetType.NUCLEAR
        nuclear_asset.current_output_mw = 800.0
        nuclear_asset.is_online.return_value = True

        coal_asset = Mock()
        coal_asset.asset_type = AssetType.COAL
        coal_asset.current_output_mw = 200.0
        coal_asset.is_online.return_value = True

        assets = [nuclear_asset, coal_asset]

        result = calculator.calculate_fossil_fuel_percentage(assets)

        # Total: 1000 MW, Fossil: 200 MW, Percentage: 20%
        assert result["fossil_fuel_percentage"] == 20.0
        assert result["total_generation_mw"] == 1000.0
        assert result["fossil_fuel_generation_mw"] == 200.0
        assert result["nuclear_generation_mw"] == 800.0

    def test_calculate_fossil_fuel_percentage_no_generation(self):
        """Test fossil fuel percentage calculation with no generation."""
        calculator = FossilFuelCalculator()

        # Create mock assets with zero output
        solar_asset = Mock()
        solar_asset.asset_type = AssetType.SOLAR
        solar_asset.current_output_mw = 0.0
        solar_asset.is_online.return_value = True

        assets = [solar_asset]

        result = calculator.calculate_fossil_fuel_percentage(assets)

        assert result["fossil_fuel_percentage"] == 0.0
        assert result["total_generation_mw"] == 0.0
        assert result["fossil_fuel_generation_mw"] == 0.0

    def test_calculate_fossil_fuel_percentage_with_storage(self):
        """Test fossil fuel percentage calculation with storage assets."""
        calculator = FossilFuelCalculator()

        # Create mock assets including storage
        solar_asset = Mock()
        solar_asset.asset_type = AssetType.SOLAR
        solar_asset.current_output_mw = 100.0
        solar_asset.is_online.return_value = True

        battery_asset = Mock()
        battery_asset.asset_type = AssetType.BATTERY
        battery_asset.current_output_mw = -50.0  # Discharging
        battery_asset.is_online.return_value = True

        coal_asset = Mock()
        coal_asset.asset_type = AssetType.COAL
        coal_asset.current_output_mw = 150.0
        coal_asset.is_online.return_value = True

        assets = [solar_asset, battery_asset, coal_asset]

        result = calculator.calculate_fossil_fuel_percentage(assets)

        # Total generation: 300 MW (solar 100 + battery discharge 50 + coal 150)
        # Fossil fuel: 150 MW (coal only)
        # Percentage: 50%
        assert result["fossil_fuel_percentage"] == 50.0
        assert result["total_generation_mw"] == 300.0
        assert result["fossil_fuel_generation_mw"] == 150.0

    def test_calculate_detailed_generation_breakdown(self):
        """Test detailed generation breakdown calculation."""
        calculator = FossilFuelCalculator()

        # Create comprehensive set of assets
        solar_asset = Mock()
        solar_asset.asset_type = AssetType.SOLAR
        solar_asset.current_output_mw = 100.0
        solar_asset.is_online.return_value = True

        wind_asset = Mock()
        wind_asset.asset_type = AssetType.WIND
        wind_asset.current_output_mw = 80.0
        wind_asset.is_online.return_value = True

        coal_asset = Mock()
        coal_asset.asset_type = AssetType.COAL
        coal_asset.current_output_mw = 200.0
        coal_asset.is_online.return_value = True

        gas_asset = Mock()
        gas_asset.asset_type = AssetType.GAS
        gas_asset.current_output_mw = 150.0
        gas_asset.is_online.return_value = True

        nuclear_asset = Mock()
        nuclear_asset.asset_type = AssetType.NUCLEAR
        nuclear_asset.current_output_mw = 800.0
        nuclear_asset.is_online.return_value = True

        assets = [solar_asset, wind_asset, coal_asset, gas_asset, nuclear_asset]

        result = calculator.calculate_generation_breakdown(assets)

        assert result["solar_generation_mw"] == 100.0
        assert result["wind_generation_mw"] == 80.0
        assert result["coal_generation_mw"] == 200.0
        assert result["gas_generation_mw"] == 150.0
        assert result["nuclear_generation_mw"] == 800.0
        assert result["total_renewable_mw"] == 180.0
        assert result["total_fossil_fuel_mw"] == 350.0
        assert result["total_generation_mw"] == 1330.0

    def test_calculate_emissions_totals(self):
        """Test calculation of total emissions from fossil fuel plants."""
        calculator = FossilFuelCalculator()

        # Create mock coal plant with emissions
        coal_plant = Mock()
        coal_plant.asset_type = AssetType.COAL
        coal_plant.current_output_mw = 200.0
        coal_plant.is_online.return_value = True

        # Ensure the method returns a proper float value
        def coal_emissions():
            return 400.0

        coal_plant.calculate_co2_emissions_per_hour = coal_emissions

        # Create mock gas plant with emissions
        gas_plant = Mock()
        gas_plant.asset_type = AssetType.GAS
        gas_plant.current_output_mw = 150.0
        gas_plant.is_online.return_value = True

        # Ensure the method returns a proper float value
        def gas_emissions():
            return 200.0

        gas_plant.calculate_co2_emissions_per_hour = gas_emissions

        # Create renewable asset (no emissions)
        solar_asset = Mock()
        solar_asset.asset_type = AssetType.SOLAR
        solar_asset.current_output_mw = 100.0
        solar_asset.is_online.return_value = True

        assets = [coal_plant, gas_plant, solar_asset]

        result = calculator.calculate_emissions_totals(assets)

        assert result["total_co2_emissions_lb_per_hour"] == 600.0
        assert result["coal_co2_emissions_lb_per_hour"] == 400.0
        assert result["gas_co2_emissions_lb_per_hour"] == 200.0
        assert result["co2_emissions_rate_lb_per_mwh"] == 600.0 / 450.0  # 600 lb/hr / 450 MW


class TestMetricsCollectorFossilFuelIntegration:
    """Test integration of fossil fuel metrics with MetricsCollector."""

    def test_fossil_fuel_metrics_hook(self):
        """Test fossil fuel metrics hook functionality."""

        # Create mock engine with mixed generation
        mock_engine = Mock()
        mock_engine.current_time = datetime(2024, 1, 1, 12, 0, 0)

        # Create mock assets
        solar_asset = Mock()
        solar_asset.asset_type = AssetType.SOLAR
        solar_asset.current_output_mw = 100.0
        solar_asset.is_online.return_value = True

        coal_asset = Mock()
        coal_asset.asset_type = AssetType.COAL
        coal_asset.current_output_mw = 200.0
        coal_asset.is_online.return_value = True
        coal_asset.calculate_co2_emissions_per_hour.return_value = 400.0

        mock_engine.get_all_assets.return_value = [solar_asset, coal_asset]

        # Create metrics collector
        collector = MetricsCollector(log_directory=Path("/tmp"), enable_time_series_logging=True)

        # Register default hooks
        collector.register_default_hooks()

        # Collect metrics
        collector.collect_metrics(mock_engine)

        # Verify fossil fuel metrics were collected
        data = collector.time_series_data[0]
        assert "fossil_fuel_metrics" in data

        fossil_metrics = data["fossil_fuel_metrics"]
        assert "fossil_fuel_percentage" in fossil_metrics
        assert "total_generation_mw" in fossil_metrics
        assert "fossil_fuel_generation_mw" in fossil_metrics
        assert "renewable_generation_mw" in fossil_metrics

        # Verify values
        assert fossil_metrics["fossil_fuel_percentage"] == 200.0 / 300.0 * 100  # 66.67%
        assert fossil_metrics["total_generation_mw"] == 300.0
        assert fossil_metrics["fossil_fuel_generation_mw"] == 200.0
        assert fossil_metrics["renewable_generation_mw"] == 100.0
