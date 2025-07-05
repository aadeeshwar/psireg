"""Tests for Load/Demand Node assets with stochastic and trace-driven profiles.

This module contains comprehensive tests for load/demand assets including:
- Stochastic demand profile generation
- Trace-driven demand profiles from data files
- Time-of-use patterns with peak/off-peak behavior
- Demand response capabilities with price elasticity
- Integration with GridEngine simulation
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest
from psireg.sim.assets.load import Load
from psireg.utils.enums import AssetStatus, AssetType
from pydantic import ValidationError


class TestLoadCreation:
    """Test Load asset creation and validation."""

    def test_load_creation_basic(self):
        """Test basic load asset creation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )
        assert load.asset_id == "load_001"
        assert load.asset_type == AssetType.LOAD
        assert load.name == "Test Load"
        assert load.node_id == "node_1"
        assert load.capacity_mw == 100.0
        assert load.baseline_demand_mw == 75.0
        assert load.status == AssetStatus.OFFLINE
        assert load.current_output_mw == 0.0
        assert load.current_demand_mw == 0.0

    def test_load_creation_with_profiles(self):
        """Test load creation with profile parameters."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            peak_demand_mw=95.0,
            off_peak_demand_mw=55.0,
            peak_hours_start=8,
            peak_hours_end=20,
            demand_volatility=0.15,
            price_elasticity=-0.2,
        )
        assert load.peak_demand_mw == 95.0
        assert load.off_peak_demand_mw == 55.0
        assert load.peak_hours_start == 8
        assert load.peak_hours_end == 20
        assert load.demand_volatility == 0.15
        assert load.price_elasticity == -0.2

    def test_load_validation_errors(self):
        """Test load validation errors."""
        # Test invalid baseline demand
        with pytest.raises(ValidationError):
            Load(
                asset_id="load_001",
                name="Invalid Load",
                node_id="node_1",
                capacity_mw=100.0,
                baseline_demand_mw=-10.0,  # Invalid negative
            )

        # Test peak demand exceeding capacity
        with pytest.raises(ValidationError):
            Load(
                asset_id="load_001",
                name="Invalid Load",
                node_id="node_1",
                capacity_mw=100.0,
                baseline_demand_mw=75.0,
                peak_demand_mw=150.0,  # Invalid > capacity
            )

        # Test invalid peak hours
        with pytest.raises(ValidationError):
            Load(
                asset_id="load_001",
                name="Invalid Load",
                node_id="node_1",
                capacity_mw=100.0,
                baseline_demand_mw=75.0,
                peak_hours_start=25,  # Invalid > 24
            )

        # Test invalid demand volatility
        with pytest.raises(ValidationError):
            Load(
                asset_id="load_001",
                name="Invalid Load",
                node_id="node_1",
                capacity_mw=100.0,
                baseline_demand_mw=75.0,
                demand_volatility=1.5,  # Invalid > 1.0
            )


class TestStochasticProfiles:
    """Test stochastic demand profile generation."""

    def test_stochastic_profile_generation(self):
        """Test stochastic demand profile generation."""
        load = Load(
            asset_id="load_001",
            name="Stochastic Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            demand_volatility=0.1,
            profile_type="stochastic",
        )
        load.set_status(AssetStatus.ONLINE)

        # Generate profile for 24 hours
        profile = load.generate_stochastic_profile(hours=24, timestep_minutes=15)
        assert len(profile) == 24 * 4  # 15-minute intervals
        assert all(0 <= demand <= load.capacity_mw for demand in profile)

        # Check profile has variability
        assert max(profile) > min(profile)
        assert abs(sum(profile) / len(profile) - load.baseline_demand_mw) < 10.0

    def test_stochastic_profile_with_daily_pattern(self):
        """Test stochastic profile with daily pattern."""
        load = Load(
            asset_id="load_001",
            name="Daily Pattern Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            peak_demand_mw=95.0,
            off_peak_demand_mw=55.0,
            peak_hours_start=8,
            peak_hours_end=20,
            demand_volatility=0.05,
            profile_type="stochastic",
        )
        load.set_status(AssetStatus.ONLINE)

        # Generate profile for peak and off-peak hours
        load.set_current_time(datetime(2023, 1, 1, 10, 0))  # Peak hour
        peak_demand = load.calculate_demand_at_time()

        load.set_current_time(datetime(2023, 1, 1, 2, 0))  # Off-peak hour
        off_peak_demand = load.calculate_demand_at_time()

        assert peak_demand > off_peak_demand
        assert peak_demand <= load.capacity_mw
        assert off_peak_demand >= 0

    def test_stochastic_profile_with_noise(self):
        """Test stochastic profile noise generation."""
        load = Load(
            asset_id="load_001",
            name="Noisy Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            demand_volatility=0.2,
            profile_type="stochastic",
        )
        load.set_status(AssetStatus.ONLINE)

        # Generate multiple samples to check randomness
        samples = []
        for _ in range(100):
            noise = load._generate_demand_noise()
            samples.append(noise)

        # Check noise is centered around 1.0 (no bias)
        avg_noise = sum(samples) / len(samples)
        assert abs(avg_noise - 1.0) < 0.1

        # Check noise has appropriate variability
        assert max(samples) > min(samples)


class TestTraceDrivenProfiles:
    """Test trace-driven demand profiles from data files."""

    def test_trace_driven_profile_from_csv(self):
        """Test trace-driven profile from CSV data."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,demand_mw\n")
            f.write("2023-01-01 00:00:00,65.0\n")
            f.write("2023-01-01 01:00:00,60.0\n")
            f.write("2023-01-01 02:00:00,58.0\n")
            f.write("2023-01-01 03:00:00,55.0\n")
            f.write("2023-01-01 04:00:00,52.0\n")
            f.write("2023-01-01 05:00:00,50.0\n")
            temp_file = f.name

        try:
            load = Load(
                asset_id="load_001",
                name="Trace-Driven Load",
                node_id="node_1",
                capacity_mw=100.0,
                baseline_demand_mw=75.0,
                profile_type="trace_driven",
                trace_file_path=temp_file,
            )
            load.set_status(AssetStatus.ONLINE)

            # Load trace data
            load.load_trace_data()
            assert len(load.trace_data) == 6

            # Test interpolation
            load.set_current_time(datetime(2023, 1, 1, 0, 30))  # Between first two points
            demand = load.calculate_demand_from_trace()
            assert 60.0 < demand < 65.0

        finally:
            os.unlink(temp_file)

    def test_trace_driven_profile_interpolation(self):
        """Test trace-driven profile interpolation."""
        # Create temporary CSV file with sparse data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,demand_mw\n")
            f.write("2023-01-01 00:00:00,60.0\n")
            f.write("2023-01-01 06:00:00,80.0\n")
            f.write("2023-01-01 12:00:00,95.0\n")
            f.write("2023-01-01 18:00:00,85.0\n")
            f.write("2023-01-01 23:59:59,65.0\n")
            temp_file = f.name

        try:
            load = Load(
                asset_id="load_001",
                name="Interpolated Load",
                node_id="node_1",
                capacity_mw=100.0,
                baseline_demand_mw=75.0,
                profile_type="trace_driven",
                trace_file_path=temp_file,
            )
            load.set_status(AssetStatus.ONLINE)
            load.load_trace_data()

            # Test interpolation at midpoint
            load.set_current_time(datetime(2023, 1, 1, 9, 0))  # Between 6:00 and 12:00
            demand = load.calculate_demand_from_trace()
            assert 80.0 < demand < 95.0

        finally:
            os.unlink(temp_file)

    def test_trace_driven_profile_edge_cases(self):
        """Test trace-driven profile edge cases."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,demand_mw\n")
            f.write("2023-01-01 00:00:00,60.0\n")
            f.write("2023-01-01 23:59:59,65.0\n")
            temp_file = f.name

        try:
            load = Load(
                asset_id="load_001",
                name="Edge Case Load",
                node_id="node_1",
                capacity_mw=100.0,
                baseline_demand_mw=75.0,
                profile_type="trace_driven",
                trace_file_path=temp_file,
            )
            load.set_status(AssetStatus.ONLINE)
            load.load_trace_data()

            # Test before first timestamp (should use baseline)
            load.set_current_time(datetime(2022, 12, 31, 23, 0))
            demand = load.calculate_demand_from_trace()
            assert demand == load.baseline_demand_mw

            # Test after last timestamp (should use baseline)
            load.set_current_time(datetime(2023, 1, 2, 1, 0))
            demand = load.calculate_demand_from_trace()
            assert demand == load.baseline_demand_mw

        finally:
            os.unlink(temp_file)


class TestTimeOfUsePatterns:
    """Test time-of-use demand patterns."""

    def test_peak_off_peak_detection(self):
        """Test peak and off-peak hour detection."""
        load = Load(
            asset_id="load_001",
            name="TOU Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            peak_demand_mw=95.0,
            off_peak_demand_mw=55.0,
            peak_hours_start=8,
            peak_hours_end=20,
        )

        # Test peak hour
        assert load.is_peak_hour(10)
        assert load.is_peak_hour(8)
        assert load.is_peak_hour(19)

        # Test off-peak hour
        assert not load.is_peak_hour(2)
        assert not load.is_peak_hour(22)
        assert not load.is_peak_hour(20)  # End hour is exclusive

    def test_time_of_use_demand_calculation(self):
        """Test time-of-use demand calculation."""
        load = Load(
            asset_id="load_001",
            name="TOU Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            peak_demand_mw=95.0,
            off_peak_demand_mw=55.0,
            peak_hours_start=8,
            peak_hours_end=20,
        )
        load.set_status(AssetStatus.ONLINE)

        # Test peak hour demand
        load.set_current_time(datetime(2023, 1, 1, 10, 0))
        peak_demand = load.calculate_time_of_use_demand()
        assert peak_demand == load.peak_demand_mw

        # Test off-peak hour demand
        load.set_current_time(datetime(2023, 1, 1, 2, 0))
        off_peak_demand = load.calculate_time_of_use_demand()
        assert off_peak_demand == load.off_peak_demand_mw

    def test_seasonal_variations(self):
        """Test seasonal demand variations."""
        load = Load(
            asset_id="load_001",
            name="Seasonal Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            seasonal_factor_summer=1.2,
            seasonal_factor_winter=1.1,
            seasonal_factor_spring=0.9,
            seasonal_factor_fall=0.95,
        )
        load.set_status(AssetStatus.ONLINE)

        # Test summer demand (July)
        load.set_current_time(datetime(2023, 7, 15, 10, 0))
        summer_factor = load.get_seasonal_factor()
        assert summer_factor == 1.2

        # Test winter demand (January)
        load.set_current_time(datetime(2023, 1, 15, 10, 0))
        winter_factor = load.get_seasonal_factor()
        assert winter_factor == 1.1

        # Test spring demand (April)
        load.set_current_time(datetime(2023, 4, 15, 10, 0))
        spring_factor = load.get_seasonal_factor()
        assert spring_factor == 0.9


class TestDemandResponse:
    """Test demand response capabilities."""

    def test_price_elasticity_response(self):
        """Test demand response to price changes."""
        load = Load(
            asset_id="load_001",
            name="DR Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            price_elasticity=-0.2,
            baseline_price=50.0,
        )
        load.set_status(AssetStatus.ONLINE)

        # Test demand response to high price
        high_price_demand = load.calculate_price_response_demand(100.0)  # 100% price increase
        assert high_price_demand < load.baseline_demand_mw

        # Test demand response to low price
        low_price_demand = load.calculate_price_response_demand(25.0)  # 50% price decrease
        assert low_price_demand > load.baseline_demand_mw

    def test_demand_response_signals(self):
        """Test demand response to grid signals."""
        load = Load(
            asset_id="load_001",
            name="DR Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
            dr_response_rate=0.8,
        )
        load.set_status(AssetStatus.ONLINE)

        # Test demand reduction signal
        load.set_demand_response_signal(-15.0)  # Reduce 15 MW
        response = load.calculate_demand_response()
        assert response < 0  # Demand reduction
        assert abs(response) <= load.dr_capability_mw

        # Test demand increase signal
        load.set_demand_response_signal(10.0)  # Increase 10 MW
        response = load.calculate_demand_response()
        assert response > 0  # Demand increase
        assert response <= load.dr_capability_mw

    def test_demand_response_limits(self):
        """Test demand response limits and constraints."""
        load = Load(
            asset_id="load_001",
            name="DR Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
            min_demand_mw=10.0,
            max_demand_mw=95.0,
        )
        load.set_status(AssetStatus.ONLINE)

        # Test demand reduction beyond minimum
        load.set_demand_response_signal(-80.0)  # Excessive reduction
        final_demand = load.calculate_final_demand()
        assert final_demand >= load.min_demand_mw

        # Test demand increase beyond maximum
        load.set_demand_response_signal(50.0)  # Excessive increase
        final_demand = load.calculate_final_demand()
        assert final_demand <= load.max_demand_mw


class TestLoadSimulation:
    """Test load asset simulation behavior."""

    def test_load_tick_update(self):
        """Test load tick update method."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            profile_type="stochastic",
            demand_volatility=0.1,
        )
        load.set_status(AssetStatus.ONLINE)

        # Initial state
        assert load.current_output_mw == 0.0

        # Tick update
        power_change = load.tick(900.0)  # 15 minutes
        assert load.current_output_mw < 0  # Loads have negative power output
        assert abs(load.current_output_mw) <= load.capacity_mw
        assert power_change != 0

    def test_load_offline_behavior(self):
        """Test load behavior when offline."""
        load = Load(
            asset_id="load_001",
            name="Offline Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )
        load.set_status(AssetStatus.OFFLINE)

        # Tick should return 0 when offline
        power_change = load.tick(900.0)
        assert load.current_output_mw == 0.0
        assert power_change == 0.0

    def test_load_get_power_output(self):
        """Test load power output method."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )
        load.set_status(AssetStatus.ONLINE)

        # Set demand and check power output
        load.current_demand_mw = 80.0
        load.current_output_mw = -80.0  # Negative for loads
        power_output = load.get_power_output()
        assert power_output == -80.0

    def test_load_state_interface(self):
        """Test load state interface."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            peak_demand_mw=95.0,
            off_peak_demand_mw=55.0,
            demand_volatility=0.1,
        )
        load.set_status(AssetStatus.ONLINE)

        state = load.get_state()
        assert state["asset_id"] == "load_001"
        assert state["asset_type"] == AssetType.LOAD
        assert state["capacity_mw"] == 100.0
        assert state["baseline_demand_mw"] == 75.0
        assert state["current_demand_mw"] is not None
        assert state["is_load"] is True
        assert state["is_renewable"] is False
        assert state["is_storage"] is False


class TestLoadIntegration:
    """Test load integration with simulation engine."""

    def test_load_with_grid_engine(self):
        """Test load integration with GridEngine."""
        from psireg.config.schema import GridConfig, SimulationConfig
        from psireg.sim.engine import GridEngine

        # Create grid engine
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Create load
        load = Load(
            asset_id="load_001",
            name="Grid Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            profile_type="stochastic",
        )
        load.set_status(AssetStatus.ONLINE)

        # Add to engine
        engine.add_asset(load)

        # Run simulation step
        engine.step(timedelta(minutes=15))

        # Check state
        state = engine.get_state()
        assert state.total_load_mw > 0
        assert len(engine.assets) == 1

    def test_multiple_loads_coordination(self):
        """Test multiple loads in coordination."""
        from psireg.config.schema import GridConfig, SimulationConfig
        from psireg.sim.engine import GridEngine

        # Create grid engine
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Create multiple loads
        loads = []
        for i in range(3):
            load = Load(
                asset_id=f"load_{i:03d}",
                name=f"Load {i}",
                node_id=f"node_{i}",
                capacity_mw=50.0,
                baseline_demand_mw=30.0 + i * 5,
                profile_type="stochastic",
            )
            load.set_status(AssetStatus.ONLINE)
            loads.append(load)
            engine.add_asset(load)

        # Run simulation
        for _ in range(4):  # 1 hour simulation
            engine.step(timedelta(minutes=15))

        # Check final state
        state = engine.get_state()
        assert state.total_load_mw > 0
        assert len(engine.assets) == 3

    def test_load_demand_response_coordination(self):
        """Test load demand response coordination."""
        # Create load with demand response
        load = Load(
            asset_id="load_001",
            name="DR Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
            price_elasticity=-0.2,
        )
        load.set_status(AssetStatus.ONLINE)

        # Simulate demand response event
        load.set_demand_response_signal(-15.0)  # Reduce demand
        load.set_electricity_price(100.0)  # High price

        # Calculate response
        demand_before = load.baseline_demand_mw
        final_demand = load.calculate_final_demand()

        assert final_demand < demand_before  # Demand reduced
        assert final_demand > 0  # Still positive
