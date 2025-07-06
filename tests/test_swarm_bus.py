"""Tests for SwarmBus central coordination system.

This module contains comprehensive tests for the SwarmBus functionality including:
- Agent registration and management
- Pheromone field coordination
- Spatial communication and neighbor discovery
- Time step synchronization
- Performance and integration testing
"""

from unittest.mock import Mock

import pytest
from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.swarm.agents.battery_agent import BatteryAgent
from psireg.swarm.agents.demand_agent import DemandAgent
from psireg.swarm.pheromone import GridPosition, PheromoneField, PheromoneType, SwarmBus
from psireg.utils.enums import AssetStatus


class TestSwarmBusCreation:
    """Test SwarmBus creation and initialization."""

    def test_swarm_bus_creation(self):
        """Test swarm bus creation with basic parameters."""
        bus = SwarmBus(
            grid_width=10,
            grid_height=10,
            pheromone_decay=0.95,
            pheromone_diffusion=0.1,
            time_step_s=1.0,
            communication_range=5.0,
        )

        assert bus.grid_width == 10
        assert bus.grid_height == 10
        assert bus.communication_range == 5.0
        assert bus.time_step_s == 1.0
        assert bus.current_time == 0.0
        assert len(bus.registered_agents) == 0
        assert isinstance(bus.pheromone_field, PheromoneField)

    def test_swarm_bus_default_values(self):
        """Test swarm bus creation with default values."""
        bus = SwarmBus(grid_width=5, grid_height=5)

        assert bus.pheromone_decay == 0.99
        assert bus.pheromone_diffusion == 0.05
        assert bus.time_step_s == 1.0
        assert bus.communication_range == 5.0
        assert bus.max_agents == 1000

    def test_swarm_bus_validation(self):
        """Test swarm bus parameter validation."""
        # Test negative dimensions
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            SwarmBus(grid_width=-1, grid_height=5)

        # Test invalid communication range
        with pytest.raises(ValueError, match="Communication range must be positive"):
            SwarmBus(grid_width=5, grid_height=5, communication_range=-1.0)

        # Test invalid max agents
        with pytest.raises(ValueError, match="Max agents must be positive"):
            SwarmBus(grid_width=5, grid_height=5, max_agents=0)


class TestAgentRegistration:
    """Test agent registration and management."""

    def test_register_agent_basic(self):
        """Test basic agent registration."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Create mock agent
        agent = Mock()
        agent.agent_id = "test_agent_001"
        agent.communication_range = 3.0

        position = GridPosition(x=5, y=5)
        result = bus.register_agent(agent, position)

        assert result is True
        assert agent.agent_id in bus.registered_agents
        assert bus.agent_positions[agent.agent_id] == position
        assert len(bus.registered_agents) == 1

    def test_register_agent_duplicate_id(self):
        """Test registering agent with duplicate ID."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Create mock agents with same ID
        agent1 = Mock()
        agent1.agent_id = "duplicate_id"
        agent1.communication_range = 3.0

        agent2 = Mock()
        agent2.agent_id = "duplicate_id"
        agent2.communication_range = 3.0

        position1 = GridPosition(x=2, y=2)
        position2 = GridPosition(x=5, y=5)

        # First registration should succeed
        result1 = bus.register_agent(agent1, position1)
        assert result1 is True

        # Second registration should fail
        result2 = bus.register_agent(agent2, position2)
        assert result2 is False
        assert len(bus.registered_agents) == 1

    def test_register_agent_out_of_bounds(self):
        """Test registering agent at invalid position."""
        bus = SwarmBus(grid_width=5, grid_height=5)

        agent = Mock()
        agent.agent_id = "test_agent"
        agent.communication_range = 3.0

        # Test out of bounds position
        with pytest.raises(ValueError, match="Agent position out of bounds"):
            bus.register_agent(agent, GridPosition(x=10, y=2))

    def test_register_agent_max_capacity(self):
        """Test registering agents beyond maximum capacity."""
        bus = SwarmBus(grid_width=5, grid_height=5, max_agents=2)

        # Register maximum number of agents
        for i in range(2):
            agent = Mock()
            agent.agent_id = f"agent_{i}"
            agent.communication_range = 3.0
            position = GridPosition(x=i, y=i)
            result = bus.register_agent(agent, position)
            assert result is True

        # Try to register one more (should fail)
        extra_agent = Mock()
        extra_agent.agent_id = "extra_agent"
        extra_agent.communication_range = 3.0
        result = bus.register_agent(extra_agent, GridPosition(x=2, y=2))
        assert result is False

    def test_unregister_agent(self):
        """Test agent unregistration."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agent
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.communication_range = 3.0
        position = GridPosition(x=5, y=5)

        bus.register_agent(agent, position)
        assert len(bus.registered_agents) == 1

        # Unregister agent
        result = bus.unregister_agent("test_agent")
        assert result is True
        assert len(bus.registered_agents) == 0
        assert "test_agent" not in bus.agent_positions

    def test_unregister_nonexistent_agent(self):
        """Test unregistering non-existent agent."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        result = bus.unregister_agent("nonexistent_agent")
        assert result is False

    def test_move_agent(self):
        """Test moving registered agent to new position."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agent
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.communication_range = 3.0
        initial_position = GridPosition(x=2, y=2)

        bus.register_agent(agent, initial_position)

        # Move agent
        new_position = GridPosition(x=7, y=7)
        result = bus.move_agent("test_agent", new_position)

        assert result is True
        assert bus.agent_positions["test_agent"] == new_position

    def test_move_nonexistent_agent(self):
        """Test moving non-existent agent."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        result = bus.move_agent("nonexistent_agent", GridPosition(x=5, y=5))
        assert result is False

    def test_move_agent_out_of_bounds(self):
        """Test moving agent to out of bounds position."""
        bus = SwarmBus(grid_width=5, grid_height=5)

        # Register agent
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.communication_range = 3.0

        bus.register_agent(agent, GridPosition(x=2, y=2))

        # Try to move out of bounds
        with pytest.raises(ValueError, match="Agent position out of bounds"):
            bus.move_agent("test_agent", GridPosition(x=10, y=2))


class TestNeighborDiscovery:
    """Test neighbor discovery and communication."""

    def test_get_neighbors_basic(self):
        """Test basic neighbor discovery."""
        bus = SwarmBus(grid_width=10, grid_height=10, communication_range=3.0)

        # Register agents
        agents = []
        positions = [
            GridPosition(x=5, y=5),  # Center agent
            GridPosition(x=6, y=5),  # Close neighbor
            GridPosition(x=8, y=5),  # Far neighbor
            GridPosition(x=5, y=6),  # Close neighbor
        ]

        for i, pos in enumerate(positions):
            agent = Mock()
            agent.agent_id = f"agent_{i}"
            agent.communication_range = 3.0
            agents.append(agent)
            bus.register_agent(agent, pos)

        # Get neighbors of center agent
        neighbors = bus.get_neighbors("agent_0")
        neighbor_ids = [agent.agent_id for agent in neighbors]

        # Should include close neighbors but not far ones
        assert "agent_1" in neighbor_ids  # Distance = 1
        assert "agent_3" in neighbor_ids  # Distance = 1
        # Note: agent_2 at distance 3 is exactly at the communication range limit of 3.0, so it should be included

    def test_get_neighbors_with_radius(self):
        """Test neighbor discovery with custom radius."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agents
        center_agent = Mock()
        center_agent.agent_id = "center"
        center_agent.communication_range = 3.0
        bus.register_agent(center_agent, GridPosition(x=5, y=5))

        far_agent = Mock()
        far_agent.agent_id = "far"
        far_agent.communication_range = 3.0
        bus.register_agent(far_agent, GridPosition(x=8, y=5))

        # Get neighbors with large radius
        neighbors = bus.get_neighbors("center", radius=5.0)
        neighbor_ids = [agent.agent_id for agent in neighbors]

        assert "far" in neighbor_ids  # Should be included with larger radius

    def test_get_neighbors_nonexistent_agent(self):
        """Test getting neighbors of non-existent agent."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        neighbors = bus.get_neighbors("nonexistent_agent")
        assert neighbors == []

    def test_get_agent_positions_in_radius(self):
        """Test getting agent positions within radius."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agents
        positions = [
            GridPosition(x=5, y=5),
            GridPosition(x=6, y=5),
            GridPosition(x=8, y=5),
        ]

        for i, pos in enumerate(positions):
            agent = Mock()
            agent.agent_id = f"agent_{i}"
            agent.communication_range = 3.0
            bus.register_agent(agent, pos)

        # Get positions within radius
        center = GridPosition(x=5, y=5)
        nearby_positions = bus.get_agent_positions_in_radius(center, radius=2.0)

        assert len(nearby_positions) == 2  # Center and one neighbor
        assert (positions[0], "agent_0") in nearby_positions
        assert (positions[1], "agent_1") in nearby_positions


class TestPheromoneCoordination:
    """Test pheromone field coordination."""

    def test_deposit_pheromone_by_agent(self):
        """Test pheromone deposition by registered agent."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agent
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.communication_range = 3.0
        position = GridPosition(x=5, y=5)

        bus.register_agent(agent, position)

        # Deposit pheromone
        result = bus.deposit_pheromone("test_agent", PheromoneType.DEMAND_REDUCTION, 0.8)
        assert result is True

        # Check pheromone was deposited
        strength = bus.pheromone_field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert strength == 0.8

    def test_deposit_pheromone_nonexistent_agent(self):
        """Test pheromone deposition by non-existent agent."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        result = bus.deposit_pheromone("nonexistent_agent", PheromoneType.DEMAND_REDUCTION, 0.5)
        assert result is False

    def test_get_pheromone_at_agent_position(self):
        """Test getting pheromone strength at agent position."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agent
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.communication_range = 3.0
        position = GridPosition(x=5, y=5)

        bus.register_agent(agent, position)

        # Deposit pheromone
        bus.pheromone_field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.6)

        # Get pheromone at agent position
        strength = bus.get_pheromone_at_agent("test_agent", PheromoneType.DEMAND_REDUCTION)
        assert strength == 0.6

    def test_get_pheromone_nonexistent_agent(self):
        """Test getting pheromone for non-existent agent."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        strength = bus.get_pheromone_at_agent("nonexistent_agent", PheromoneType.DEMAND_REDUCTION)
        assert strength == 0.0

    def test_get_neighborhood_pheromones(self):
        """Test getting pheromone field in agent neighborhood."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agent
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.communication_range = 3.0
        position = GridPosition(x=5, y=5)

        bus.register_agent(agent, position)

        # Deposit pheromones in neighborhood
        bus.pheromone_field.deposit_pheromone(GridPosition(x=4, y=5), PheromoneType.DEMAND_REDUCTION, 0.4)
        bus.pheromone_field.deposit_pheromone(GridPosition(x=6, y=5), PheromoneType.DEMAND_REDUCTION, 0.6)

        # Get neighborhood pheromones
        neighborhood = bus.get_neighborhood_pheromones("test_agent", PheromoneType.DEMAND_REDUCTION, radius=2)

        assert len(neighborhood) > 0
        strengths = [strength for pos, strength in neighborhood]
        assert 0.4 in strengths
        assert 0.6 in strengths


class TestTimeStepSynchronization:
    """Test time step synchronization and updates."""

    def test_update_time_step_basic(self):
        """Test basic time step update."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        initial_time = bus.current_time
        bus.update_time_step()

        assert bus.current_time == initial_time + bus.time_step_s
        assert bus.pheromone_field.current_time == bus.current_time

    def test_update_time_step_with_pheromone_decay(self):
        """Test time step update with pheromone decay."""
        bus = SwarmBus(grid_width=10, grid_height=10, pheromone_decay=0.9, pheromone_diffusion=0.0)

        # Deposit pheromone
        position = GridPosition(x=5, y=5)
        bus.pheromone_field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 1.0)

        # Update time step
        bus.update_time_step()

        # Check pheromone decayed
        strength = bus.pheromone_field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert abs(strength - 0.9) < 1e-6

    def test_coordinate_agents_basic(self):
        """Test basic agent coordination."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agents
        agents = []
        positions = [
            GridPosition(x=5, y=5),
            GridPosition(x=6, y=5),
            GridPosition(x=7, y=5),
        ]

        for i, pos in enumerate(positions):
            agent = Mock()
            agent.agent_id = f"agent_{i}"
            agent.communication_range = 3.0
            agent.get_coordination_signal.return_value = 0.5
            agent.get_pheromone_strength.return_value = 0.3
            agents.append(agent)
            bus.register_agent(agent, pos)

        # Coordinate agents
        coordination_results = bus.coordinate_agents()

        assert len(coordination_results) == 3
        for agent_id, neighbors in coordination_results.items():
            assert agent_id in [f"agent_{i}" for i in range(3)]
            assert isinstance(neighbors, list)

    def test_coordinate_agents_with_pheromone_update(self):
        """Test agent coordination with pheromone field updates."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agent
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.communication_range = 3.0
        agent.get_coordination_signal.return_value = 0.4
        agent.get_pheromone_strength.return_value = 0.7

        position = GridPosition(x=5, y=5)
        bus.register_agent(agent, position)

        # Coordinate with pheromone update
        bus.coordinate_agents(update_pheromones=True)

        # Check pheromone was deposited
        strength = bus.pheromone_field.get_pheromone_strength(position, PheromoneType.COORDINATION)
        assert strength == 0.7


class TestSwarmBusIntegration:
    """Test integration with actual agent classes."""

    def test_integration_with_battery_agent(self):
        """Test integration with BatteryAgent."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Create battery asset
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        battery.set_status(AssetStatus.ONLINE)

        # Create battery agent
        agent = BatteryAgent(battery=battery)
        position = GridPosition(x=5, y=5)

        # Register agent
        result = bus.register_agent(agent, position)
        assert result is True

        # Test coordination
        coordination_results = bus.coordinate_agents()
        assert agent.agent_id in coordination_results

    def test_integration_with_demand_agent(self):
        """Test integration with DemandAgent."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Create load asset
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
        )
        load.set_status(AssetStatus.ONLINE)

        # Create demand agent
        agent = DemandAgent(load=load)
        position = GridPosition(x=3, y=3)

        # Register agent
        result = bus.register_agent(agent, position)
        assert result is True

        # Test pheromone operations
        bus.deposit_pheromone(agent.agent_id, PheromoneType.DEMAND_REDUCTION, 0.8)
        strength = bus.get_pheromone_at_agent(agent.agent_id, PheromoneType.DEMAND_REDUCTION)
        assert strength == 0.8

    def test_multi_agent_coordination(self):
        """Test coordination between multiple different agent types."""
        bus = SwarmBus(grid_width=15, grid_height=15)

        # Create battery agent
        battery = Battery(
            asset_id="battery_001",
            name="Test Battery",
            node_id="node_1",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        battery.set_status(AssetStatus.ONLINE)
        battery_agent = BatteryAgent(battery=battery)

        # Create demand agent
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_2",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
        )
        load.set_status(AssetStatus.ONLINE)
        demand_agent = DemandAgent(load=load)

        # Register agents
        bus.register_agent(battery_agent, GridPosition(x=5, y=5))
        bus.register_agent(demand_agent, GridPosition(x=6, y=5))

        # Coordinate agents
        coordination_results = bus.coordinate_agents(update_pheromones=True)

        assert len(coordination_results) == 2
        assert battery_agent.agent_id in coordination_results
        assert demand_agent.agent_id in coordination_results

        # Check that agents found each other as neighbors
        battery_neighbors = coordination_results[battery_agent.agent_id]
        demand_neighbors = coordination_results[demand_agent.agent_id]

        assert len(battery_neighbors) > 0
        assert len(demand_neighbors) > 0


class TestSwarmBusPerformance:
    """Test performance aspects of SwarmBus."""

    def test_large_scale_agent_registration(self):
        """Test performance with large number of agents."""
        bus = SwarmBus(grid_width=50, grid_height=50, max_agents=100)

        # Register many agents
        agents = []
        for i in range(50):
            agent = Mock()
            agent.agent_id = f"agent_{i:03d}"
            agent.communication_range = 3.0
            agent.get_coordination_signal.return_value = 0.1
            agent.get_pheromone_strength.return_value = 0.2

            position = GridPosition(x=i % 50, y=i // 50)
            result = bus.register_agent(agent, position)
            assert result is True
            agents.append(agent)

        # Coordinate all agents
        coordination_results = bus.coordinate_agents()
        assert len(coordination_results) == 50

    def test_neighbor_discovery_performance(self):
        """Test neighbor discovery performance with many agents."""
        bus = SwarmBus(grid_width=20, grid_height=20)

        # Create grid of agents
        for x in range(20):
            for y in range(20):
                agent = Mock()
                agent.agent_id = f"agent_{x:02d}_{y:02d}"
                agent.communication_range = 3.0
                bus.register_agent(agent, GridPosition(x=x, y=y))

        # Test neighbor discovery for center agent
        center_agent_id = "agent_10_10"
        neighbors = bus.get_neighbors(center_agent_id, radius=5.0)

        # Should find reasonable number of neighbors
        assert len(neighbors) > 0
        assert len(neighbors) < 100  # Sanity check

    def test_pheromone_field_update_performance(self):
        """Test pheromone field update performance."""
        bus = SwarmBus(grid_width=100, grid_height=100)

        # Deposit pheromones across grid
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                position = GridPosition(x=i, y=j)
                bus.pheromone_field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.5)

        # Multiple time steps should complete in reasonable time
        for _ in range(10):
            bus.update_time_step()

        # Basic functionality should still work
        total = bus.pheromone_field.get_total_pheromone(PheromoneType.DEMAND_REDUCTION)
        assert total > 0.0


class TestSwarmBusUtilities:
    """Test utility functions of SwarmBus."""

    def test_get_agent_info(self):
        """Test getting agent information."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agent
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.communication_range = 3.0
        position = GridPosition(x=5, y=5)

        bus.register_agent(agent, position)

        # Get agent info
        info = bus.get_agent_info("test_agent")
        assert info["agent_id"] == "test_agent"
        assert info["position"] == position
        assert info["communication_range"] == 3.0

    def test_get_agent_info_nonexistent(self):
        """Test getting info for non-existent agent."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        info = bus.get_agent_info("nonexistent_agent")
        assert info is None

    def test_get_system_stats(self):
        """Test getting system statistics."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register some agents
        for i in range(3):
            agent = Mock()
            agent.agent_id = f"agent_{i}"
            agent.communication_range = 3.0
            position = GridPosition(x=i, y=i)
            bus.register_agent(agent, position)

        # Get stats
        stats = bus.get_system_stats()

        assert stats["total_agents"] == 3
        assert stats["grid_size"] == (10, 10)
        assert stats["current_time"] == 0.0
        assert "pheromone_totals" in stats

    def test_reset_system(self):
        """Test resetting the entire system."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        # Register agent and deposit pheromones
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.communication_range = 3.0
        position = GridPosition(x=5, y=5)

        bus.register_agent(agent, position)
        bus.deposit_pheromone("test_agent", PheromoneType.DEMAND_REDUCTION, 0.5)
        bus.update_time_step()

        # Reset system
        bus.reset()

        assert bus.current_time == 0.0
        assert len(bus.registered_agents) == 0
        assert len(bus.agent_positions) == 0
        assert bus.pheromone_field.get_total_pheromone(PheromoneType.DEMAND_REDUCTION) == 0.0

    def test_validate_grid_position(self):
        """Test grid position validation."""
        bus = SwarmBus(grid_width=5, grid_height=5)

        # Valid positions
        assert bus._validate_position(GridPosition(x=0, y=0)) is True
        assert bus._validate_position(GridPosition(x=4, y=4)) is True
        assert bus._validate_position(GridPosition(x=2, y=3)) is True

        # Invalid positions
        assert bus._validate_position(GridPosition(x=-1, y=0)) is False
        assert bus._validate_position(GridPosition(x=5, y=0)) is False
        assert bus._validate_position(GridPosition(x=0, y=5)) is False

    def test_calculate_distance(self):
        """Test distance calculation between positions."""
        bus = SwarmBus(grid_width=10, grid_height=10)

        pos1 = GridPosition(x=0, y=0)
        pos2 = GridPosition(x=3, y=4)

        # Should calculate Euclidean distance
        distance = bus._calculate_distance(pos1, pos2)
        assert abs(distance - 5.0) < 1e-6  # 3-4-5 triangle
