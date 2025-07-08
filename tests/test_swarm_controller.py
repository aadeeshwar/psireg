"""Tests for swarm-only controller implementation."""

from unittest.mock import Mock, patch

from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.assets.wind import WindTurbine
from psireg.sim.engine import GridEngine, GridState
from psireg.swarm.pheromone import PheromoneType
from psireg.utils.enums import AssetType


class TestSwarmController:
    """Test swarm-only controller implementation."""

    def test_swarm_controller_creation(self):
        """Test that swarm-only controller can be created."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()
        assert controller is not None

    def test_swarm_controller_initialization(self):
        """Test swarm-only controller initialization."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()
        grid_engine = Mock(spec=GridEngine)

        # Mock assets with required attributes
        battery = Mock(spec=Battery)
        battery.asset_id = "battery_1"
        battery.asset_type = AssetType.BATTERY
        battery.nominal_voltage_v = 480.0  # Required by BatteryAgent

        load = Mock(spec=Load)
        load.asset_id = "load_1"
        load.asset_type = AssetType.LOAD

        grid_engine.get_all_assets.return_value = [battery, load]

        with patch("psireg.controllers.swarm.BatteryAgent"):
            with patch("psireg.controllers.swarm.DemandAgent"):
                with patch("psireg.controllers.swarm.SwarmBus"):
                    result = controller.initialize(grid_engine)
                    assert result is True
                    assert controller.grid_engine == grid_engine

    def test_swarm_controller_agent_creation(self):
        """Test that swarm controller creates appropriate agents."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()
        grid_engine = Mock(spec=GridEngine)

        # Mock assets with required attributes
        battery = Mock(spec=Battery)
        battery.asset_id = "battery_1"
        battery.asset_type = AssetType.BATTERY
        battery.nominal_voltage_v = 480.0  # Required by BatteryAgent

        solar = Mock(spec=SolarPanel)
        solar.asset_id = "solar_1"
        solar.asset_type = AssetType.SOLAR

        wind = Mock(spec=WindTurbine)
        wind.asset_id = "wind_1"
        wind.asset_type = AssetType.WIND

        load = Mock(spec=Load)
        load.asset_id = "load_1"
        load.asset_type = AssetType.LOAD

        grid_engine.get_all_assets.return_value = [battery, solar, wind, load]

        with patch("psireg.controllers.swarm.BatteryAgent") as mock_battery_agent:
            with patch("psireg.controllers.swarm.SolarAgent") as mock_solar_agent:
                with patch("psireg.controllers.swarm.WindAgent") as mock_wind_agent:
                    with patch("psireg.controllers.swarm.DemandAgent") as mock_demand_agent:
                        with patch("psireg.controllers.swarm.SwarmBus") as mock_swarm_bus:
                            controller.initialize(grid_engine)

                            # Verify agents were created with correct parameters
                            mock_battery_agent.assert_called_once_with(
                                battery=battery,
                                communication_range=controller.communication_range,
                                coordination_weight=0.4,
                            )
                            mock_solar_agent.assert_called_once_with(
                                solar_panel=solar,
                                communication_range=controller.communication_range,
                                coordination_weight=0.3,
                            )
                            mock_wind_agent.assert_called_once_with(
                                wind_turbine=wind,
                                communication_range=controller.communication_range,
                                coordination_weight=0.3,
                            )
                            mock_demand_agent.assert_called_once_with(
                                load=load, communication_range=controller.communication_range, coordination_weight=0.35
                            )
                            mock_swarm_bus.assert_called_once()

    def test_swarm_controller_coordination_update(self):
        """Test swarm controller coordination update."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()
        Mock(spec=GridEngine)

        # Mock swarm bus and agents
        mock_swarm_bus = Mock()
        mock_agent = Mock()
        mock_agent.agent_id = "battery_1"

        controller.swarm_bus = mock_swarm_bus
        controller.agents = [mock_agent]

        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 59.8
        grid_state.total_generation_mw = 90.0
        grid_state.total_load_mw = 100.0
        grid_state.power_balance_mw = -10.0

        controller.update(grid_state, 1.0)

        # Verify swarm coordination was updated
        mock_agent.update_grid_conditions.assert_called_once()
        mock_swarm_bus.coordinate_agents.assert_called_once()

    def test_swarm_controller_pheromone_field_interaction(self):
        """Test swarm controller pheromone field interaction."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()
        Mock(spec=GridEngine)

        # Mock swarm bus
        mock_swarm_bus = Mock()
        controller.swarm_bus = mock_swarm_bus

        # Mock agents
        battery_agent = Mock()
        battery_agent.agent_id = "battery_1"
        battery_agent.pheromone_strength = 0.8

        demand_agent = Mock()
        demand_agent.agent_id = "load_1"
        demand_agent.pheromone_strength = 0.6

        controller.agents = [battery_agent, demand_agent]

        grid_state = Mock(spec=GridState)
        controller.update(grid_state, 1.0)

        # Verify pheromone interactions
        mock_swarm_bus.deposit_pheromone.assert_called()

    def test_swarm_controller_agent_optimization(self):
        """Test swarm controller agent optimization."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()

        # Mock battery agent
        battery_agent = Mock()
        battery_agent.agent_id = "battery_1"
        battery_agent.calculate_optimal_power.return_value = 25.0

        # Mock demand agent
        demand_agent = Mock()
        demand_agent.agent_id = "load_1"
        demand_agent.calculate_optimal_demand.return_value = 70.0

        controller.agents = [battery_agent, demand_agent]

        grid_state = Mock(spec=GridState)
        controller.update(grid_state, 1.0)

        actions = controller.get_control_actions()

        # Verify optimization was called
        battery_agent.calculate_optimal_power.assert_called()
        demand_agent.calculate_optimal_demand.assert_called()

        # Verify actions were generated
        assert "battery_1" in actions
        assert "load_1" in actions

    def test_swarm_controller_emergency_response(self):
        """Test swarm controller emergency response."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()

        # Mock agents
        battery_agent = Mock()
        battery_agent.agent_id = "battery_1"
        battery_agent.calculate_local_stabilization_signal.return_value = {
            "power_mw": -30.0,
            "priority": 0.9,
            "response_time_s": 1.0,
        }

        controller.agents = [battery_agent]

        # Test emergency grid state
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 59.5  # Emergency frequency
        grid_state.power_balance_mw = -50.0  # Large imbalance

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        # Verify emergency response was triggered
        battery_agent.calculate_local_stabilization_signal.assert_called()
        assert "battery_1" in actions

    def test_swarm_controller_multi_objective_optimization(self):
        """Test swarm controller multi-objective optimization."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()

        # Mock solar agent with demand response
        solar_agent = Mock()
        solar_agent.agent_id = "solar_1"
        solar_agent.calculate_demand_response_signal.return_value = {
            "curtailment_factor": 0.2,
            "demand_response_mw": 5.0,
            "economic_value": 150.0,
        }

        controller.agents = [solar_agent]

        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 60.2  # High frequency

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        # Verify multi-objective optimization
        solar_agent.calculate_demand_response_signal.assert_called()
        assert "solar_1" in actions

    def test_swarm_controller_performance_metrics(self):
        """Test swarm controller performance metrics."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()

        # Mock agents with metrics
        battery_agent = Mock()
        battery_agent.get_agent_state.return_value = Mock(
            agent_id="battery_1", grid_support_priority=0.8, coordination_signal=0.6
        )

        solar_agent = Mock()
        solar_agent.get_agent_state.return_value = Mock(
            agent_id="solar_1", grid_support_priority=0.7, coordination_signal=0.4
        )

        controller.agents = [battery_agent, solar_agent]

        # Mock swarm bus metrics
        mock_swarm_bus = Mock()
        mock_swarm_bus.get_system_stats.return_value = {
            "active_agents": 2,
            "pheromone_totals": {PheromoneType.FREQUENCY_SUPPORT: 0.8},
            "coordination_effectiveness": 0.75,
        }
        controller.swarm_bus = mock_swarm_bus

        metrics = controller.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert "active_agents" in metrics
        assert "coordination_effectiveness" in metrics
        assert "pheromone_activity" in metrics
        assert "agent_participation" in metrics

    def test_swarm_controller_reset(self):
        """Test swarm controller reset functionality."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()

        # Mock agents and swarm bus
        mock_agent = Mock()
        mock_swarm_bus = Mock()

        controller.agents = [mock_agent]
        controller.swarm_bus = mock_swarm_bus

        controller.reset()

        # Verify reset was called on components
        mock_agent.reset.assert_called_once()
        mock_swarm_bus.reset_system.assert_called_once()

    def test_swarm_controller_scalability(self):
        """Test swarm controller with large number of agents."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()
        grid_engine = Mock(spec=GridEngine)

        # Create many mock assets
        assets = []
        for i in range(50):  # 50 assets
            asset = Mock(spec=Battery)
            asset.asset_id = f"battery_{i}"
            asset.asset_type = AssetType.BATTERY
            assets.append(asset)

        grid_engine.get_all_assets.return_value = assets

        with patch("psireg.controllers.swarm.BatteryAgent") as mock_battery_agent:
            with patch("psireg.controllers.swarm.SwarmBus"):
                result = controller.initialize(grid_engine)

                assert result is True
                assert mock_battery_agent.call_count == 50

    def test_swarm_controller_neighbor_coordination(self):
        """Test swarm controller neighbor coordination."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()

        # Mock agents
        agent1 = Mock()
        agent1.agent_id = "battery_1"
        agent1.coordination_signal = 0.6

        agent2 = Mock()
        agent2.agent_id = "battery_2"
        agent2.coordination_signal = 0.8

        controller.agents = [agent1, agent2]

        # Mock swarm bus neighbor discovery
        mock_swarm_bus = Mock()
        mock_swarm_bus.get_neighbors.return_value = ["battery_2"]
        controller.swarm_bus = mock_swarm_bus

        grid_state = Mock(spec=GridState)
        controller.update(grid_state, 1.0)

        # Verify neighbor coordination
        mock_swarm_bus.get_neighbors.assert_called()
        agent1.update_swarm_signals.assert_called()


class TestSwarmControllerCoordination:
    """Test swarm controller coordination mechanisms."""

    def test_pheromone_gradient_following(self):
        """Test pheromone gradient following behavior."""
        # This will test gradient following once implemented
        pass

    def test_swarm_consensus_building(self):
        """Test swarm consensus building mechanisms."""
        # This will test consensus mechanisms once implemented
        pass

    def test_emergent_behavior_validation(self):
        """Test emergent swarm behavior validation."""
        # This will test emergent behaviors once implemented
        pass

    def test_swarm_adaptation_learning(self):
        """Test swarm adaptation and learning."""
        # This will test adaptation mechanisms once implemented
        pass


class TestSwarmControllerIntegration:
    """Test swarm controller integration scenarios."""

    def test_swarm_controller_frequency_response(self):
        """Test swarm controller coordinated frequency response."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()

        # Multiple battery agents for coordinated response
        agents = []
        for i in range(3):
            agent = Mock()
            agent.agent_id = f"battery_{i}"
            agent.calculate_optimal_power.return_value = -10.0  # Discharge
            agents.append(agent)

        controller.agents = agents

        # Low frequency scenario
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 59.7
        grid_state.power_balance_mw = -30.0

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        # All agents should contribute to frequency response
        assert len(actions) == 3
        for i in range(3):
            assert f"battery_{i}" in actions

    def test_swarm_controller_renewable_coordination(self):
        """Test swarm controller renewable coordination."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()

        # Renewable agents
        solar_agent = Mock()
        solar_agent.agent_id = "solar_1"
        solar_agent.calculate_demand_response_signal.return_value = {"curtailment_factor": 0.15}

        wind_agent = Mock()
        wind_agent.agent_id = "wind_1"
        wind_agent.calculate_demand_response_signal.return_value = {"curtailment_factor": 0.10}

        controller.agents = [solar_agent, wind_agent]

        # Over-generation scenario
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 60.3
        grid_state.power_balance_mw = 25.0

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        # Both should coordinate curtailment
        assert "solar_1" in actions
        assert "wind_1" in actions

    def test_swarm_controller_demand_response_coordination(self):
        """Test swarm controller demand response coordination."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()

        # Multiple demand agents
        demand_agents = []
        for i in range(4):
            agent = Mock()
            agent.agent_id = f"load_{i}"
            agent.calculate_optimal_demand.return_value = 75.0 - i * 5  # Progressive reduction
            demand_agents.append(agent)

        controller.agents = demand_agents

        # High demand scenario
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 59.8
        grid_state.power_balance_mw = -20.0

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        # All demand agents should participate
        assert len(actions) == 4

    def test_swarm_controller_mixed_asset_coordination(self):
        """Test swarm controller with mixed asset types."""
        from psireg.controllers.swarm import SwarmController

        controller = SwarmController()

        # Mixed agents
        battery_agent = Mock()
        battery_agent.agent_id = "battery_1"
        battery_agent.calculate_optimal_power.return_value = 15.0

        solar_agent = Mock()
        solar_agent.agent_id = "solar_1"
        solar_agent.calculate_demand_response_signal.return_value = {"curtailment_factor": 0.05}

        demand_agent = Mock()
        demand_agent.agent_id = "load_1"
        demand_agent.calculate_optimal_demand.return_value = 70.0

        controller.agents = [battery_agent, solar_agent, demand_agent]

        grid_state = Mock(spec=GridState)
        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        # All asset types should be coordinated
        assert "battery_1" in actions
        assert "solar_1" in actions
        assert "load_1" in actions
