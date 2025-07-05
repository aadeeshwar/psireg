"""Tests for simulation engine functionality.

This module tests the grid simulation engine that evolves grid state with minimal physics,
including network topology, power flow balance, frequency/voltage tracking, and asset scheduling.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest
from psireg.config.schema import GridConfig, SimulationConfig
from psireg.sim.assets.base import Asset, AssetType
from psireg.sim.engine import GridEngine, GridState, NetworkNode, TransmissionLine
from psireg.utils.enums import AssetStatus, SimulationMode
from pydantic import ValidationError


class TestAssetBase:
    """Test base asset functionality."""

    def test_asset_creation(self):
        """Test creation of base asset."""
        asset = Asset(
            asset_id="test_asset_1",
            asset_type=AssetType.SOLAR,
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=10.0,
        )
        assert asset.asset_id == "test_asset_1"
        assert asset.asset_type == AssetType.SOLAR
        assert asset.name == "Test Solar Panel"
        assert asset.node_id == "node_1"
        assert asset.capacity_mw == 10.0
        assert asset.status == AssetStatus.OFFLINE
        assert asset.current_output_mw == 0.0

    def test_asset_validation(self):
        """Test asset validation."""
        # Test negative capacity
        with pytest.raises(ValidationError):
            Asset(
                asset_id="invalid",
                asset_type=AssetType.SOLAR,
                name="Invalid Asset",
                node_id="node_1",
                capacity_mw=-10.0,
            )

        # Test empty asset_id
        with pytest.raises(ValidationError):
            Asset(
                asset_id="",
                asset_type=AssetType.SOLAR,
                name="Invalid Asset",
                node_id="node_1",
                capacity_mw=10.0,
            )

    def test_asset_tick_interface(self):
        """Test asset tick method interface."""
        asset = Asset(
            asset_id="test_asset",
            asset_type=AssetType.SOLAR,
            name="Test Asset",
            node_id="node_1",
            capacity_mw=10.0,
        )
        # Should not raise exception (abstract method returns None)
        result = asset.tick(1.0)
        assert result is None

    def test_asset_status_management(self):
        """Test asset status management."""
        asset = Asset(
            asset_id="test_asset",
            asset_type=AssetType.SOLAR,
            name="Test Asset",
            node_id="node_1",
            capacity_mw=10.0,
        )

        # Test status transitions
        asset.set_status(AssetStatus.ONLINE)
        assert asset.status == AssetStatus.ONLINE

        asset.set_status(AssetStatus.MAINTENANCE)
        assert asset.status == AssetStatus.MAINTENANCE

    def test_asset_get_state(self):
        """Test asset get_state method for uniform interface."""
        asset = Asset(
            asset_id="test_asset",
            asset_type=AssetType.SOLAR,
            name="Test Asset",
            node_id="node_1",
            capacity_mw=100.0,
        )
        asset.set_status(AssetStatus.ONLINE)
        asset.set_power_output(75.0)

        state = asset.get_state()
        assert isinstance(state, dict)
        assert state["asset_id"] == "test_asset"
        assert state["asset_type"] == AssetType.SOLAR
        assert state["name"] == "Test Asset"
        assert state["node_id"] == "node_1"
        assert state["capacity_mw"] == 100.0
        assert state["status"] == AssetStatus.ONLINE
        assert state["current_output_mw"] == 75.0
        assert state["is_online"] is True
        assert state["is_renewable"] is True
        assert state["utilization_percent"] == 75.0

    def test_asset_utilization_calculation(self):
        """Test asset utilization percentage calculation."""
        asset = Asset(
            asset_id="test_asset",
            asset_type=AssetType.SOLAR,
            name="Test Asset",
            node_id="node_1",
            capacity_mw=200.0,
        )

        # Test with no output
        assert asset.get_utilization_percent() == 0.0

        # Test with 50% output
        asset.set_power_output(100.0)
        assert asset.get_utilization_percent() == 50.0

        # Test with 100% output
        asset.set_power_output(200.0)
        assert asset.get_utilization_percent() == 100.0

        # Test with negative output (load)
        asset.set_power_output(-50.0)
        assert asset.get_utilization_percent() == 25.0

    def test_asset_efficiency_calculation(self):
        """Test asset efficiency calculation."""
        asset = Asset(
            asset_id="test_asset",
            asset_type=AssetType.SOLAR,
            name="Test Asset",
            node_id="node_1",
            capacity_mw=100.0,
        )

        # Test default efficiency (should be 1.0 for base class)
        assert asset.get_efficiency() == 1.0

        # Test with power output
        asset.set_power_output(80.0)
        assert asset.get_efficiency() == 1.0  # Base class has no efficiency loss

    def test_asset_reset_method(self):
        """Test asset reset method."""
        asset = Asset(
            asset_id="test_asset",
            asset_type=AssetType.SOLAR,
            name="Test Asset",
            node_id="node_1",
            capacity_mw=100.0,
        )

        # Change asset state
        asset.set_status(AssetStatus.ONLINE)
        asset.set_power_output(75.0)

        # Reset asset
        asset.reset()

        # Check that asset is reset to defaults
        assert asset.status == AssetStatus.OFFLINE
        assert asset.current_output_mw == 0.0

    def test_asset_power_limits(self):
        """Test asset power output limits."""
        asset = Asset(
            asset_id="test_asset",
            asset_type=AssetType.SOLAR,
            name="Test Asset",
            node_id="node_1",
            capacity_mw=100.0,
        )

        # Test within limits
        asset.set_power_output(50.0)
        assert asset.current_output_mw == 50.0

        # Test exceeding capacity (should be clamped)
        asset.set_power_output(150.0)
        assert asset.current_output_mw <= 100.0

        # Test negative power (allowed for loads)
        asset.set_power_output(-50.0)
        assert asset.current_output_mw == -50.0


class TestNetworkTopology:
    """Test network topology components."""

    def test_network_node_creation(self):
        """Test creation of network node."""
        node = NetworkNode(
            node_id="node_1",
            name="Substation Alpha",
            voltage_kv=230.0,
            latitude=37.7749,
            longitude=-122.4194,
        )
        assert node.node_id == "node_1"
        assert node.name == "Substation Alpha"
        assert node.voltage_kv == 230.0
        assert node.latitude == 37.7749
        assert node.longitude == -122.4194
        assert node.current_frequency_hz == 60.0  # Default
        assert node.current_voltage_kv == 230.0  # Default to nominal

    def test_transmission_line_creation(self):
        """Test creation of transmission line."""
        line = TransmissionLine(
            line_id="line_1",
            name="Alpha-Beta Transmission",
            from_node="node_1",
            to_node="node_2",
            capacity_mw=500.0,
            length_km=50.0,
            resistance=0.1,
        )
        assert line.line_id == "line_1"
        assert line.name == "Alpha-Beta Transmission"
        assert line.from_node == "node_1"
        assert line.to_node == "node_2"
        assert line.capacity_mw == 500.0
        assert line.length_km == 50.0
        assert line.resistance == 0.1
        assert line.current_flow_mw == 0.0

    def test_transmission_line_validation(self):
        """Test transmission line validation."""
        # Test negative capacity
        with pytest.raises(ValidationError):
            TransmissionLine(
                line_id="invalid",
                name="Invalid Line",
                from_node="node_1",
                to_node="node_2",
                capacity_mw=-100.0,
                length_km=10.0,
                resistance=0.1,
            )

        # Test same from/to nodes
        with pytest.raises(ValidationError):
            TransmissionLine(
                line_id="invalid",
                name="Invalid Line",
                from_node="node_1",
                to_node="node_1",
                capacity_mw=100.0,
                length_km=10.0,
                resistance=0.1,
            )


class TestGridState:
    """Test grid state management."""

    def test_grid_state_creation(self):
        """Test creation of grid state."""
        timestamp = datetime.now()
        state = GridState(
            timestamp=timestamp,
            frequency_hz=60.0,
            total_generation_mw=1000.0,
            total_load_mw=950.0,
            total_storage_mw=50.0,
            grid_losses_mw=25.0,
        )
        assert state.timestamp == timestamp
        assert state.frequency_hz == 60.0
        assert state.total_generation_mw == 1000.0
        assert state.total_load_mw == 950.0
        assert state.total_storage_mw == 50.0
        assert state.grid_losses_mw == 25.0

    def test_grid_state_power_balance(self):
        """Test power balance calculation."""
        state = GridState(
            timestamp=datetime.now(),
            frequency_hz=60.0,
            total_generation_mw=1000.0,
            total_load_mw=950.0,
            total_storage_mw=50.0,
            grid_losses_mw=25.0,
        )
        # Power balance = generation - load - storage - losses
        expected_balance = 1000.0 - 950.0 - 50.0 - 25.0
        assert state.power_balance_mw == expected_balance

    def test_grid_state_validation(self):
        """Test grid state validation."""
        # Test negative frequency
        with pytest.raises(ValidationError):
            GridState(
                timestamp=datetime.now(),
                frequency_hz=-60.0,
                total_generation_mw=1000.0,
                total_load_mw=950.0,
                total_storage_mw=50.0,
                grid_losses_mw=25.0,
            )


class TestGridEngine:
    """Test grid simulation engine."""

    def test_engine_creation(self):
        """Test creation of grid engine."""
        sim_config = SimulationConfig(
            timestep_minutes=15,
            horizon_hours=24,
            mode=SimulationMode.REAL_TIME,
            max_assets=100,
        )
        grid_config = GridConfig(
            frequency_hz=60.0,
            voltage_kv=230.0,
            stability_threshold=0.1,
            max_power_mw=1000.0,
        )

        engine = GridEngine(simulation_config=sim_config, grid_config=grid_config)
        assert engine.simulation_config == sim_config
        assert engine.grid_config == grid_config
        assert engine.current_time == datetime.fromtimestamp(0)  # Default start time
        assert len(engine.assets) == 0
        assert len(engine.nodes) == 0
        assert len(engine.transmission_lines) == 0

    def test_engine_initialization_with_custom_start_time(self):
        """Test engine initialization with custom start time."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        sim_config = SimulationConfig()
        grid_config = GridConfig()

        engine = GridEngine(
            simulation_config=sim_config,
            grid_config=grid_config,
            start_time=start_time,
        )
        assert engine.current_time == start_time

    def test_add_asset(self):
        """Test adding assets to the engine."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        asset = Asset(
            asset_id="solar_1",
            asset_type=AssetType.SOLAR,
            name="Solar Farm 1",
            node_id="node_1",
            capacity_mw=100.0,
        )

        engine.add_asset(asset)
        assert len(engine.assets) == 1
        assert "solar_1" in engine.assets
        assert engine.assets["solar_1"] == asset

    def test_add_duplicate_asset(self):
        """Test adding duplicate asset raises error."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        asset1 = Asset(
            asset_id="solar_1",
            asset_type=AssetType.SOLAR,
            name="Solar Farm 1",
            node_id="node_1",
            capacity_mw=100.0,
        )

        asset2 = Asset(
            asset_id="solar_1",  # Same ID
            asset_type=AssetType.WIND,
            name="Wind Farm 1",
            node_id="node_2",
            capacity_mw=200.0,
        )

        engine.add_asset(asset1)
        with pytest.raises(ValueError, match="Asset with ID 'solar_1' already exists"):
            engine.add_asset(asset2)

    def test_add_node(self):
        """Test adding network nodes."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        node = NetworkNode(
            node_id="node_1",
            name="Substation Alpha",
            voltage_kv=230.0,
            latitude=37.7749,
            longitude=-122.4194,
        )

        engine.add_node(node)
        assert len(engine.nodes) == 1
        assert "node_1" in engine.nodes
        assert engine.nodes["node_1"] == node

    def test_add_transmission_line(self):
        """Test adding transmission lines."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Add nodes first
        node1 = NetworkNode(node_id="node_1", name="Node 1", voltage_kv=230.0)
        node2 = NetworkNode(node_id="node_2", name="Node 2", voltage_kv=230.0)
        engine.add_node(node1)
        engine.add_node(node2)

        line = TransmissionLine(
            line_id="line_1",
            name="Line 1-2",
            from_node="node_1",
            to_node="node_2",
            capacity_mw=500.0,
            length_km=50.0,
            resistance=0.1,
        )

        engine.add_transmission_line(line)
        assert len(engine.transmission_lines) == 1
        assert "line_1" in engine.transmission_lines
        assert engine.transmission_lines["line_1"] == line

    def test_add_transmission_line_invalid_nodes(self):
        """Test adding transmission line with invalid nodes raises error."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        line = TransmissionLine(
            line_id="line_1",
            name="Line 1-2",
            from_node="nonexistent_1",
            to_node="nonexistent_2",
            capacity_mw=500.0,
            length_km=50.0,
            resistance=0.1,
        )

        with pytest.raises(ValueError, match="Node 'nonexistent_1' not found"):
            engine.add_transmission_line(line)

    def test_get_state(self):
        """Test getting current grid state."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        state = engine.get_state()
        assert isinstance(state, GridState)
        assert isinstance(state.timestamp, datetime)
        assert state.frequency_hz == 60.0  # Default grid frequency
        assert state.total_generation_mw == 0.0  # No assets initially
        assert state.total_load_mw == 0.0
        assert state.total_storage_mw == 0.0

    def test_step_basic(self):
        """Test basic step functionality."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
            start_time=start_time,
        )

        dt = timedelta(minutes=15)
        engine.step(dt)

        assert engine.current_time == start_time + dt

    def test_step_with_assets(self):
        """Test step functionality with assets."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Add mock asset
        mock_asset = Mock(spec=Asset)
        mock_asset.asset_id = "mock_asset"
        mock_asset.asset_type = AssetType.SOLAR
        mock_asset.node_id = "node_1"
        mock_asset.tick.return_value = None
        mock_asset.get_power_output.return_value = 0.0

        engine.assets["mock_asset"] = mock_asset

        dt = timedelta(minutes=15)
        engine.step(dt)

        # Verify asset tick was called
        mock_asset.tick.assert_called_once_with(dt.total_seconds())

    def test_step_power_flow_calculation(self):
        """Test power flow calculation during step."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Add nodes
        node1 = NetworkNode(node_id="node_1", name="Gen Node", voltage_kv=230.0)
        node2 = NetworkNode(node_id="node_2", name="Load Node", voltage_kv=230.0)
        engine.add_node(node1)
        engine.add_node(node2)

        # Add mock generation asset
        gen_asset = Mock(spec=Asset)
        gen_asset.asset_id = "gen_1"
        gen_asset.asset_type = AssetType.SOLAR
        gen_asset.node_id = "node_1"
        gen_asset.current_output_mw = 100.0
        gen_asset.get_power_output.return_value = 100.0
        gen_asset.tick.return_value = None
        engine.assets["gen_1"] = gen_asset

        # Add mock load asset
        load_asset = Mock(spec=Asset)
        load_asset.asset_id = "load_1"
        load_asset.asset_type = AssetType.LOAD
        load_asset.node_id = "node_2"
        load_asset.current_output_mw = -80.0  # Negative for load
        load_asset.get_power_output.return_value = -80.0
        load_asset.tick.return_value = None
        engine.assets["load_1"] = load_asset

        dt = timedelta(minutes=15)
        engine.step(dt)

        state = engine.get_state()
        assert state.total_generation_mw == 100.0
        assert state.total_load_mw == 80.0  # Absolute value
        assert state.power_balance_mw == 18.0  # 100 - 80 - 2 (grid losses)

    def test_step_frequency_tracking(self):
        """Test frequency tracking during step."""
        grid_config = GridConfig(frequency_hz=60.0, stability_threshold=0.1)
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=grid_config,
        )

        # Add generation and load to create power imbalance
        node = NetworkNode(node_id="node_1", name="Test Node", voltage_kv=230.0)
        engine.add_node(node)

        gen_asset = Mock(spec=Asset)
        gen_asset.asset_id = "gen_1"
        gen_asset.asset_type = AssetType.SOLAR
        gen_asset.node_id = "node_1"
        gen_asset.current_output_mw = 100.0
        gen_asset.get_power_output.return_value = 100.0
        gen_asset.tick.return_value = None
        engine.assets["gen_1"] = gen_asset

        load_asset = Mock(spec=Asset)
        load_asset.asset_id = "load_1"
        load_asset.asset_type = AssetType.LOAD
        load_asset.node_id = "node_1"
        load_asset.current_output_mw = -90.0
        load_asset.get_power_output.return_value = -90.0
        load_asset.tick.return_value = None
        engine.assets["load_1"] = load_asset

        dt = timedelta(minutes=15)
        engine.step(dt)

        state = engine.get_state()
        # With positive power balance, frequency should increase slightly
        assert state.frequency_hz >= 60.0

    def test_step_voltage_tracking(self):
        """Test voltage tracking at nodes."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        node = NetworkNode(node_id="node_1", name="Test Node", voltage_kv=230.0)
        engine.add_node(node)

        dt = timedelta(minutes=15)
        engine.step(dt)

        # Node voltage should be tracked
        assert node.current_voltage_kv == 230.0  # Should remain stable with no load

    def test_step_transmission_line_flows(self):
        """Test transmission line power flow calculation."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Add nodes
        gen_node = NetworkNode(node_id="gen_node", name="Generation Node", voltage_kv=230.0)
        load_node = NetworkNode(node_id="load_node", name="Load Node", voltage_kv=230.0)
        engine.add_node(gen_node)
        engine.add_node(load_node)

        # Add transmission line
        line = TransmissionLine(
            line_id="line_1",
            name="Gen-Load Line",
            from_node="gen_node",
            to_node="load_node",
            capacity_mw=500.0,
            length_km=50.0,
            resistance=0.1,
        )
        engine.add_transmission_line(line)

        # Add generation at one node
        gen_asset = Mock(spec=Asset)
        gen_asset.asset_id = "gen_1"
        gen_asset.asset_type = AssetType.SOLAR
        gen_asset.node_id = "gen_node"
        gen_asset.current_output_mw = 100.0
        gen_asset.get_power_output.return_value = 100.0
        gen_asset.tick.return_value = None
        engine.assets["gen_1"] = gen_asset

        # Add load at another node
        load_asset = Mock(spec=Asset)
        load_asset.asset_id = "load_1"
        load_asset.asset_type = AssetType.LOAD
        load_asset.node_id = "load_node"
        load_asset.current_output_mw = -100.0
        load_asset.get_power_output.return_value = -100.0
        load_asset.tick.return_value = None
        engine.assets["load_1"] = load_asset

        dt = timedelta(minutes=15)
        engine.step(dt)

        # Transmission line should have power flow
        assert abs(line.current_flow_mw) > 0

    def test_max_assets_limit(self):
        """Test maximum assets limit enforcement."""
        sim_config = SimulationConfig(max_assets=2)
        engine = GridEngine(
            simulation_config=sim_config,
            grid_config=GridConfig(),
        )

        # Add two assets successfully
        asset1 = Asset(
            asset_id="asset_1",
            asset_type=AssetType.SOLAR,
            name="Asset 1",
            node_id="node_1",
            capacity_mw=100.0,
        )
        asset2 = Asset(
            asset_id="asset_2",
            asset_type=AssetType.WIND,
            name="Asset 2",
            node_id="node_2",
            capacity_mw=200.0,
        )

        engine.add_asset(asset1)
        engine.add_asset(asset2)

        # Third asset should raise error
        asset3 = Asset(
            asset_id="asset_3",
            asset_type=AssetType.BATTERY,
            name="Asset 3",
            node_id="node_3",
            capacity_mw=50.0,
        )

        with pytest.raises(ValueError, match="Maximum number of assets.*exceeded"):
            engine.add_asset(asset3)

    def test_grid_stability_monitoring(self):
        """Test grid stability monitoring."""
        grid_config = GridConfig(frequency_hz=60.0, stability_threshold=0.05)
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=grid_config,
        )

        # Force frequency outside stability threshold
        engine._current_frequency = 60.1  # Outside 0.05 Hz threshold

        dt = timedelta(minutes=15)
        engine.step(dt)

        state = engine.get_state()
        assert abs(state.frequency_hz - 60.0) <= grid_config.stability_threshold

    @pytest.mark.parametrize(
        "simulation_mode",
        [
            SimulationMode.REAL_TIME,
            SimulationMode.HISTORICAL,
            SimulationMode.BATCH,
        ],
    )
    def test_different_simulation_modes(self, simulation_mode):
        """Test engine works with different simulation modes."""
        sim_config = SimulationConfig(mode=simulation_mode)
        engine = GridEngine(
            simulation_config=sim_config,
            grid_config=GridConfig(),
        )

        dt = timedelta(minutes=15)
        engine.step(dt)

        # Should complete without errors
        state = engine.get_state()
        assert isinstance(state, GridState)

    def test_engine_reset(self):
        """Test engine reset functionality."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
            start_time=start_time,
        )

        # Add some assets and advance time
        asset = Asset(
            asset_id="test_asset",
            asset_type=AssetType.SOLAR,
            name="Test Asset",
            node_id="node_1",
            capacity_mw=100.0,
        )
        engine.add_asset(asset)
        engine.step(timedelta(hours=1))

        # Reset engine
        new_start_time = datetime(2024, 1, 2, 12, 0, 0)
        engine.reset(start_time=new_start_time)

        assert engine.current_time == new_start_time
        assert len(engine.assets) == 0  # Assets should be cleared
        assert len(engine.nodes) == 0
        assert len(engine.transmission_lines) == 0

    def test_engine_state_persistence(self):
        """Test that engine state persists across steps."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Add asset
        asset = Asset(
            asset_id="persistent_asset",
            asset_type=AssetType.SOLAR,
            name="Persistent Asset",
            node_id="node_1",
            capacity_mw=100.0,
        )
        engine.add_asset(asset)

        # Step multiple times
        for _ in range(5):
            engine.step(timedelta(minutes=15))

        # Asset should still be there
        assert "persistent_asset" in engine.assets
        assert engine.assets["persistent_asset"] == asset

    def test_asset_registry_methods(self):
        """Test enhanced asset registry functionality."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Add multiple assets
        solar_asset = Asset(
            asset_id="solar_1",
            asset_type=AssetType.SOLAR,
            name="Solar Farm 1",
            node_id="node_1",
            capacity_mw=100.0,
        )
        wind_asset = Asset(
            asset_id="wind_1",
            asset_type=AssetType.WIND,
            name="Wind Farm 1",
            node_id="node_2",
            capacity_mw=150.0,
        )
        battery_asset = Asset(
            asset_id="battery_1",
            asset_type=AssetType.BATTERY,
            name="Battery Storage 1",
            node_id="node_1",
            capacity_mw=50.0,
        )

        engine.add_asset(solar_asset)
        engine.add_asset(wind_asset)
        engine.add_asset(battery_asset)

        # Test get_assets_by_type (already exists)
        solar_assets = engine.get_assets_by_type(AssetType.SOLAR)
        assert len(solar_assets) == 1
        assert solar_assets[0] == solar_asset

        # Test get_assets_by_node (already exists)
        node1_assets = engine.get_assets_by_node("node_1")
        assert len(node1_assets) == 2
        assert solar_asset in node1_assets
        assert battery_asset in node1_assets

        # Test get_asset_by_id (already exists)
        found_asset = engine.get_asset_by_id("wind_1")
        assert found_asset == wind_asset

    def test_asset_registry_lifecycle(self):
        """Test asset registry lifecycle management."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        asset = Asset(
            asset_id="lifecycle_asset",
            asset_type=AssetType.SOLAR,
            name="Lifecycle Asset",
            node_id="node_1",
            capacity_mw=100.0,
        )

        # Add asset
        engine.add_asset(asset)
        assert "lifecycle_asset" in engine.assets

        # Remove asset (new functionality)
        removed_asset = engine.remove_asset("lifecycle_asset")
        assert removed_asset == asset
        assert "lifecycle_asset" not in engine.assets

        # Try to remove non-existent asset
        result = engine.remove_asset("non_existent")
        assert result is None

    def test_asset_registry_bulk_operations(self):
        """Test bulk operations on asset registry."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        assets = [
            Asset(
                asset_id=f"asset_{i}",
                asset_type=AssetType.SOLAR,
                name=f"Asset {i}",
                node_id=f"node_{i % 3}",
                capacity_mw=100.0,
            )
            for i in range(5)
        ]

        # Bulk add assets (new functionality)
        engine.add_assets(assets)
        assert len(engine.assets) == 5

        # Get all assets
        all_assets = engine.get_all_assets()
        assert len(all_assets) == 5
        for asset in assets:
            assert asset in all_assets

        # Clear all assets (new functionality)
        engine.clear_assets()
        assert len(engine.assets) == 0

    def test_asset_registry_filtering(self):
        """Test advanced asset filtering capabilities."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Add assets with different statuses
        online_asset = Asset(
            asset_id="online_asset",
            asset_type=AssetType.SOLAR,
            name="Online Asset",
            node_id="node_1",
            capacity_mw=100.0,
        )
        online_asset.set_status(AssetStatus.ONLINE)

        offline_asset = Asset(
            asset_id="offline_asset",
            asset_type=AssetType.WIND,
            name="Offline Asset",
            node_id="node_2",
            capacity_mw=150.0,
        )
        offline_asset.set_status(AssetStatus.OFFLINE)

        maintenance_asset = Asset(
            asset_id="maintenance_asset",
            asset_type=AssetType.BATTERY,
            name="Maintenance Asset",
            node_id="node_3",
            capacity_mw=50.0,
        )
        maintenance_asset.set_status(AssetStatus.MAINTENANCE)

        engine.add_asset(online_asset)
        engine.add_asset(offline_asset)
        engine.add_asset(maintenance_asset)

        # Test filtering by status (new functionality)
        online_assets = engine.get_assets_by_status(AssetStatus.ONLINE)
        assert len(online_assets) == 1
        assert online_assets[0] == online_asset

        offline_assets = engine.get_assets_by_status(AssetStatus.OFFLINE)
        assert len(offline_assets) == 1
        assert offline_assets[0] == offline_asset

        # Test filtering by capacity range (new functionality)
        large_assets = engine.get_assets_by_capacity_range(min_mw=100.0)
        assert len(large_assets) == 2
        assert online_asset in large_assets
        assert offline_asset in large_assets

        small_assets = engine.get_assets_by_capacity_range(max_mw=100.0)
        assert len(small_assets) == 2
        assert online_asset in small_assets
        assert maintenance_asset in small_assets


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_step_with_zero_dt(self):
        """Test step with zero time delta."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        initial_time = engine.current_time
        engine.step(timedelta(0))

        # Time should not advance
        assert engine.current_time == initial_time

    def test_step_with_negative_dt(self):
        """Test step with negative time delta raises error."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        with pytest.raises(ValueError, match="Time delta must be non-negative"):
            engine.step(timedelta(seconds=-1))

    def test_get_state_with_no_assets(self):
        """Test getting state with no assets."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        state = engine.get_state()
        assert state.total_generation_mw == 0.0
        assert state.total_load_mw == 0.0
        assert state.total_storage_mw == 0.0
        assert state.power_balance_mw == 0.0

    def test_asset_with_invalid_node(self):
        """Test asset referencing non-existent node."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        asset = Asset(
            asset_id="orphan_asset",
            asset_type=AssetType.SOLAR,
            name="Orphan Asset",
            node_id="nonexistent_node",
            capacity_mw=100.0,
        )

        # Should add successfully (lazy validation)
        engine.add_asset(asset)

        # Warning should be logged during step (not tested here but would be in integration)
        dt = timedelta(minutes=15)
        engine.step(dt)  # Should not crash

    def test_large_simulation_timestep(self):
        """Test with very large simulation timestep."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Very large timestep
        dt = timedelta(days=1)
        engine.step(dt)

        # Should handle gracefully
        state = engine.get_state()
        assert isinstance(state, GridState)


class TestIntegrationScenarios:
    """Test integrated simulation scenarios."""

    def test_simple_renewable_grid(self):
        """Test simple renewable energy grid scenario."""
        engine = GridEngine(
            simulation_config=SimulationConfig(timestep_minutes=15),
            grid_config=GridConfig(frequency_hz=60.0, max_power_mw=1000.0),
        )

        # Add network topology
        gen_node = NetworkNode(node_id="gen_node", name="Generation", voltage_kv=230.0)
        load_node = NetworkNode(node_id="load_node", name="Load Center", voltage_kv=230.0)
        engine.add_node(gen_node)
        engine.add_node(load_node)

        line = TransmissionLine(
            line_id="main_line",
            name="Main Transmission",
            from_node="gen_node",
            to_node="load_node",
            capacity_mw=800.0,
            length_km=100.0,
            resistance=0.05,
        )
        engine.add_transmission_line(line)

        # Add renewable generation
        solar = Asset(
            asset_id="solar_farm",
            asset_type=AssetType.SOLAR,
            name="Solar Farm",
            node_id="gen_node",
            capacity_mw=300.0,
        )
        wind = Asset(
            asset_id="wind_farm",
            asset_type=AssetType.WIND,
            name="Wind Farm",
            node_id="gen_node",
            capacity_mw=200.0,
        )

        # Add storage
        battery = Asset(
            asset_id="battery_storage",
            asset_type=AssetType.BATTERY,
            name="Battery Storage",
            node_id="load_node",
            capacity_mw=100.0,
        )

        # Add load
        city_load = Asset(
            asset_id="city_load",
            asset_type=AssetType.LOAD,
            name="City Load",
            node_id="load_node",
            capacity_mw=400.0,
        )

        engine.add_asset(solar)
        engine.add_asset(wind)
        engine.add_asset(battery)
        engine.add_asset(city_load)

        # Simulate for one hour
        for _ in range(4):  # 4 * 15 minutes = 1 hour
            engine.step(timedelta(minutes=15))

        final_state = engine.get_state()
        assert isinstance(final_state, GridState)
        assert final_state.frequency_hz > 0

        # Should have 4 assets
        assert len(engine.assets) == 4

    def test_grid_with_transmission_constraints(self):
        """Test grid with transmission line capacity constraints."""
        engine = GridEngine(
            simulation_config=SimulationConfig(),
            grid_config=GridConfig(),
        )

        # Create nodes with limited transmission capacity
        node1 = NetworkNode(node_id="node_1", name="High Generation", voltage_kv=230.0)
        node2 = NetworkNode(node_id="node_2", name="High Load", voltage_kv=230.0)
        engine.add_node(node1)
        engine.add_node(node2)

        # Limited capacity transmission line
        line = TransmissionLine(
            line_id="limited_line",
            name="Limited Capacity Line",
            from_node="node_1",
            to_node="node_2",
            capacity_mw=50.0,  # Very limited
            length_km=10.0,
            resistance=0.01,
        )
        engine.add_transmission_line(line)

        # High generation at node 1
        gen = Mock(spec=Asset)
        gen.asset_id = "big_gen"
        gen.asset_type = AssetType.SOLAR
        gen.node_id = "node_1"
        gen.current_output_mw = 100.0  # Exceeds line capacity
        gen.get_power_output.return_value = 100.0
        gen.tick.return_value = None
        engine.assets["big_gen"] = gen

        # High load at node 2
        load = Mock(spec=Asset)
        load.asset_id = "big_load"
        load.asset_type = AssetType.LOAD
        load.node_id = "node_2"
        load.current_output_mw = -100.0
        load.get_power_output.return_value = -100.0
        load.tick.return_value = None
        engine.assets["big_load"] = load

        dt = timedelta(minutes=15)
        engine.step(dt)

        # Transmission line flow should be constrained
        assert abs(line.current_flow_mw) <= line.capacity_mw
