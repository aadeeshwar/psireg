"""Tests for DemandAgent swarm intelligence coordination.

This module contains comprehensive tests for demand agent functionality including:
- Swarm coordination and communication
- Demand response optimization
- Load scheduling and shifting
- Grid frequency support
- Economic optimization
- Integration with Load assets
"""

from psireg.sim.assets.load import Load
from psireg.swarm.agents.demand_agent import DemandAgent, DemandSwarmState
from psireg.utils.enums import AssetStatus


class TestDemandAgentCreation:
    """Test DemandAgent creation and initialization."""

    def test_demand_agent_creation(self):
        """Test demand agent creation with load asset."""
        # Create load asset
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
            price_elasticity=-0.2,
        )

        # Create demand agent
        agent = DemandAgent(
            load=load,
            agent_id="agent_001",
            communication_range=10.0,
            response_time_s=5.0,
            coordination_weight=0.3,
        )

        assert agent.agent_id == "agent_001"
        assert agent.load == load
        assert agent.communication_range == 10.0
        assert agent.response_time_s == 5.0
        assert agent.coordination_weight == 0.3
        assert agent.target_demand_factor == 1.0
        assert agent.pheromone_strength == 0.0
        assert agent.local_grid_stress == 0.0

    def test_demand_agent_default_id(self):
        """Test demand agent creation with default ID."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )

        agent = DemandAgent(load=load)
        assert agent.agent_id == "load_001"


class TestGridConditionsUpdate:
    """Test grid conditions update functionality."""

    def test_update_grid_conditions(self):
        """Test updating grid conditions."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
        )

        agent = DemandAgent(load=load)

        # Update grid conditions
        agent.update_grid_conditions(
            frequency_hz=59.8,  # Low frequency
            voltage_kv=230.0,
            local_load_mw=150.0,
            local_generation_mw=100.0,  # Generation < load
            electricity_price=80.0,
        )

        assert agent.local_grid_stress > 0.0  # Should detect stress
        assert load.current_price == 80.0  # Price should be updated

    def test_grid_stress_calculation(self):
        """Test grid stress calculation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )

        agent = DemandAgent(load=load)

        # High frequency deviation
        agent.update_grid_conditions(
            frequency_hz=60.3,  # High frequency
            voltage_kv=230.0,
            local_load_mw=100.0,
            local_generation_mw=100.0,
        )

        assert agent.local_grid_stress > 0.0


class TestSwarmCoordination:
    """Test swarm coordination functionality."""

    def test_swarm_signal_update(self):
        """Test swarm signal update from neighbors."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )

        agent = DemandAgent(load=load)

        # Update with neighbor signals
        neighbor_signals = [0.5, -0.3, 0.2]
        agent.update_swarm_signals(neighbor_signals)

        assert len(agent.neighbor_signals) == 3
        assert agent.coordination_signal != 0.0

    def test_swarm_signal_decay(self):
        """Test swarm signal decay without neighbors."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )

        agent = DemandAgent(load=load)

        # Set initial signal
        agent.coordination_signal = 0.8

        # Update with no neighbors
        agent.update_swarm_signals([])

        assert agent.coordination_signal < 0.8  # Should decay

    def test_coordination_signal_calculation(self):
        """Test coordination signal calculation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )
        load.current_demand_mw = 80.0  # Higher than baseline

        agent = DemandAgent(load=load)

        signal = agent.get_coordination_signal()
        assert signal > 0.0  # Higher demand should give positive signal


class TestDemandOptimization:
    """Test demand optimization functionality."""

    def test_optimal_demand_calculation(self):
        """Test optimal demand calculation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
            price_elasticity=-0.2,
        )
        load.set_status(AssetStatus.ONLINE)

        agent = DemandAgent(load=load)

        # Calculate optimal demand
        forecast_prices = [80.0, 70.0, 60.0, 50.0]
        forecast_generation = [100.0, 110.0, 120.0, 130.0]
        forecast_grid_stress = [0.3, 0.2, 0.1, 0.0]

        optimal_demand = agent.calculate_optimal_demand(
            forecast_prices=forecast_prices,
            forecast_generation=forecast_generation,
            forecast_grid_stress=forecast_grid_stress,
            time_horizon_hours=4,
        )

        assert optimal_demand > 0.0
        assert optimal_demand <= load.capacity_mw

    def test_frequency_response_calculation(self):
        """Test frequency response calculation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
        )

        agent = DemandAgent(load=load)

        # Set high grid stress
        agent.local_grid_stress = 0.8

        frequency_response = agent._calculate_frequency_response()
        assert frequency_response < 0.0  # Should reduce demand

    def test_economic_response_calculation(self):
        """Test economic response calculation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            price_elasticity=-0.2,
        )
        load.set_electricity_price(80.0)  # High price

        agent = DemandAgent(load=load)

        # High current price compared to forecast
        forecast_prices = [50.0, 55.0, 60.0, 65.0]
        economic_response = agent._calculate_economic_response(forecast_prices)

        # Should reduce demand when price is high
        assert economic_response < 0.0

    def test_coordination_response_calculation(self):
        """Test coordination response calculation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
        )

        agent = DemandAgent(load=load)

        # Set positive coordination signal
        agent.coordination_signal = 0.5

        coordination_response = agent._calculate_coordination_response()
        assert coordination_response < 0.0  # Should reduce demand

    def test_comfort_response_calculation(self):
        """Test comfort response calculation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )
        load.current_demand_mw = 85.0  # Higher than baseline

        agent = DemandAgent(load=load)

        comfort_response = agent._calculate_comfort_response()
        assert comfort_response < 0.0  # Should restore toward baseline


class TestDemandScheduling:
    """Test demand scheduling functionality."""

    def test_schedule_demand_shift(self):
        """Test scheduling demand shift."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )

        agent = DemandAgent(load=load)

        # Schedule shift from hour 10 to hour 14
        success = agent.schedule_demand_shift(
            shift_mw=10.0,
            from_hour=10,
            to_hour=14,
            duration_hours=2,
        )

        assert success is True
        assert 10 in agent.scheduled_adjustments
        assert 14 in agent.scheduled_adjustments
        assert agent.scheduled_adjustments[10] == -10.0  # Reduction
        assert agent.scheduled_adjustments[14] == 10.0  # Increase

    def test_schedule_demand_shift_limits(self):
        """Test demand shift limits."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )

        agent = DemandAgent(load=load)

        # Try to shift too much (more than flexible fraction)
        success = agent.schedule_demand_shift(
            shift_mw=50.0,  # Too much
            from_hour=10,
            to_hour=14,
        )

        assert success is False

    def test_schedule_demand_shift_time_limits(self):
        """Test demand shift time limits."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )

        agent = DemandAgent(load=load)

        # Try to shift too far in time
        success = agent.schedule_demand_shift(
            shift_mw=10.0,
            from_hour=8,
            to_hour=18,  # 10 hours apart > max_shift_hours
        )

        assert success is False


class TestAgentControl:
    """Test agent control functionality."""

    def test_execute_control_action(self):
        """Test executing control action."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
        )

        agent = DemandAgent(load=load)

        # Execute control action
        optimal_demand = 70.0  # Reduce demand
        agent.execute_control_action(optimal_demand)

        assert load.dr_signal_mw == -5.0  # 70 - 75 = -5
        assert agent.pheromone_strength > 0.0

    def test_grid_support_priority_calculation(self):
        """Test grid support priority calculation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
        )
        load.set_electricity_price(30.0)  # Low price

        agent = DemandAgent(load=load)
        agent.local_grid_stress = 0.6  # High stress

        priority = agent._calculate_grid_support_priority()
        assert 0.0 <= priority <= 1.0
        assert priority > 0.4  # Should be reasonably high due to stress and low price

    def test_target_demand_factor_update(self):
        """Test target demand factor update."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )

        agent = DemandAgent(load=load)

        # Update target demand factor
        agent.update_target_demand_factor(1.2)
        assert agent.target_demand_factor == 1.2

        # Test bounds
        agent.update_target_demand_factor(2.5)  # Too high
        assert agent.target_demand_factor <= 1.8

        agent.update_target_demand_factor(0.1)  # Too low
        assert agent.target_demand_factor >= 0.2


class TestAgentState:
    """Test agent state functionality."""

    def test_get_agent_state(self):
        """Test getting agent state."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            peak_demand_mw=90.0,
            dr_capability_mw=20.0,
            price_elasticity=-0.2,
        )
        load.current_demand_mw = 80.0
        load.set_electricity_price(60.0)

        agent = DemandAgent(load=load)

        state = agent.get_agent_state()

        assert isinstance(state, DemandSwarmState)
        assert state.agent_id == "load_001"
        assert state.current_demand_mw == 80.0
        assert state.baseline_demand_mw == 75.0
        assert state.peak_demand_mw == 90.0
        assert state.dr_capability_mw == 20.0
        assert state.price_elasticity == -0.2
        assert state.current_price == 60.0
        assert 0.0 <= state.flexibility_factor <= 1.0

    def test_pheromone_strength_calculation(self):
        """Test pheromone strength calculation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
        )

        agent = DemandAgent(load=load)

        # Execute control action to set pheromone strength
        agent.execute_control_action(65.0)  # Reduce demand

        pheromone_strength = agent.get_pheromone_strength()
        assert 0.0 <= pheromone_strength <= 1.0
        assert pheromone_strength > 0.0

    def test_agent_reset(self):
        """Test agent reset functionality."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
        )

        agent = DemandAgent(load=load)

        # Set some state
        agent.pheromone_strength = 0.5
        agent.local_grid_stress = 0.3
        agent.coordination_signal = 0.2
        agent.neighbor_signals = [0.1, 0.2, 0.3]
        agent.scheduled_adjustments = {10: -5.0, 14: 5.0}
        agent.target_demand_factor = 1.2

        # Reset
        agent.reset()

        assert agent.pheromone_strength == 0.0
        assert agent.local_grid_stress == 0.0
        assert agent.coordination_signal == 0.0
        assert agent.neighbor_signals == []
        assert agent.scheduled_adjustments == {}
        assert agent.target_demand_factor == 1.0


class TestAgentIntegration:
    """Test agent integration with load assets."""

    def test_agent_with_load_simulation(self):
        """Test agent integration with load simulation."""
        load = Load(
            asset_id="load_001",
            name="Test Load",
            node_id="node_1",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
            price_elasticity=-0.2,
        )
        load.set_status(AssetStatus.ONLINE)

        agent = DemandAgent(load=load)

        # Simulate grid conditions
        agent.update_grid_conditions(
            frequency_hz=59.9,
            voltage_kv=230.0,
            local_load_mw=200.0,
            local_generation_mw=180.0,
            electricity_price=75.0,
        )

        # Calculate optimal demand
        forecast_prices = [80.0, 70.0, 60.0, 50.0]
        forecast_generation = [180.0, 190.0, 200.0, 210.0]
        forecast_grid_stress = [0.4, 0.3, 0.2, 0.1]

        optimal_demand = agent.calculate_optimal_demand(
            forecast_prices=forecast_prices,
            forecast_generation=forecast_generation,
            forecast_grid_stress=forecast_grid_stress,
        )

        # Execute control action
        agent.execute_control_action(optimal_demand)

        # Check that load asset was updated
        assert load.dr_signal_mw != 0.0
        assert agent.pheromone_strength > 0.0

    def test_multiple_agents_coordination(self):
        """Test coordination between multiple agents."""
        # Create multiple loads and agents
        loads = []
        agents = []

        for i in range(3):
            load = Load(
                asset_id=f"load_{i:03d}",
                name=f"Load {i}",
                node_id=f"node_{i}",
                capacity_mw=100.0,
                baseline_demand_mw=50.0 + i * 10,
                dr_capability_mw=15.0,
            )
            loads.append(load)

            agent = DemandAgent(load=load)
            agents.append(agent)

        # Simulate coordination
        for i, agent in enumerate(agents):
            # Get signals from other agents
            neighbor_signals = [agents[j].get_coordination_signal() for j in range(len(agents)) if j != i]

            agent.update_swarm_signals(neighbor_signals)

        # Check that coordination signals were updated
        for agent in agents:
            assert len(agent.neighbor_signals) == 2  # Two neighbors each
