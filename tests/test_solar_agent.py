"""Tests for SolarAgent swarm intelligence coordination.

This module provides comprehensive tests for the SolarAgent class that combines
PPO forecasting with pheromone gradient coordination to produce demand response signals.
"""

from unittest.mock import Mock

import numpy as np
from psireg.sim.assets.base import AssetStatus
from psireg.sim.assets.solar import SolarPanel
from psireg.swarm.pheromone import GridPosition, PheromoneField, PheromoneType, SwarmBus
from psireg.utils.enums import WeatherCondition


class TestSolarAgentCore:
    """Test SolarAgent core functionality."""

    def test_solar_agent_initialization(self):
        """Test SolarAgent initialization."""
        # Create solar panel
        solar_panel = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="solar_node",
            capacity_mw=100.0,
            panel_efficiency=0.22,
            panel_area_m2=15000.0,
        )

        # Import SolarAgent (will be implemented)
        from psireg.swarm.agents.solar_agent import SolarAgent

        # Create agent
        agent = SolarAgent(
            solar_panel=solar_panel,
            agent_id="solar_agent_001",
            communication_range=5.0,
            response_time_s=2.0,
            coordination_weight=0.3,
        )

        # Verify initialization
        assert agent.solar_panel == solar_panel
        assert agent.agent_id == "solar_agent_001"
        assert agent.communication_range == 5.0
        assert agent.response_time_s == 2.0
        assert agent.coordination_weight == 0.3

        # Verify control parameters
        assert agent.target_curtailment_factor == 0.0
        assert agent.curtailment_deadband_percent == 2.0
        assert agent.frequency_deadband_hz == 0.03
        assert agent.max_frequency_response_rate == 0.08

        # Verify swarm coordination state
        assert agent.pheromone_strength == 0.0
        assert agent.local_grid_stress == 0.0
        assert agent.coordination_signal == 0.0
        assert agent.neighbor_signals == []

        # Verify economic parameters
        assert agent.electricity_price > 0
        assert agent.curtailment_cost > 0
        assert agent.grid_service_value > 0

        # Verify PPO integration
        assert agent.ppo_predictor is None  # Initially None
        assert agent.forecast_horizon_hours == 24
        assert agent.forecast_cache == {}

    def test_solar_agent_default_agent_id(self):
        """Test SolarAgent with default agent ID."""
        solar_panel = SolarPanel(
            asset_id="solar_002",
            name="Test Solar Panel 2",
            node_id="solar_node",
            capacity_mw=50.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Should use solar panel asset_id as default
        assert agent.agent_id == "solar_002"

    def test_solar_agent_state_management(self):
        """Test SolarAgent state management."""
        solar_panel = SolarPanel(
            asset_id="solar_003",
            name="Test Solar Panel 3",
            node_id="solar_node",
            capacity_mw=75.0,
        )
        solar_panel.set_status(AssetStatus.ONLINE)

        from psireg.swarm.agents.solar_agent import SolarAgent, SolarSwarmState

        agent = SolarAgent(solar_panel=solar_panel)

        # Get agent state
        state = agent.get_agent_state()

        # Verify state type and contents
        assert isinstance(state, SolarSwarmState)
        assert state.agent_id == "solar_003"
        assert state.current_generation_mw == 0.0  # Initial generation
        assert state.capacity_mw == 75.0
        assert state.panel_efficiency == 0.20  # Default efficiency
        assert state.current_curtailment_factor == 1.0  # No curtailment initially
        assert 0.0 <= state.grid_support_priority <= 1.0
        assert state.coordination_signal == 0.0


class TestSolarAgentPPOIntegration:
    """Test SolarAgent PPO forecasting integration."""

    def test_ppo_predictor_integration(self):
        """Test PPO predictor integration."""
        solar_panel = SolarPanel(
            asset_id="solar_004",
            name="Test Solar Panel 4",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Mock PPO predictor
        mock_predictor = Mock()
        mock_predictor.predict.return_value = np.array([0.5, -0.3, 0.8])  # Mock action

        # Set PPO predictor
        agent.set_ppo_predictor(mock_predictor)

        assert agent.ppo_predictor == mock_predictor
        assert agent.ppo_predictor is not None

    def test_generation_forecasting(self):
        """Test generation forecasting with PPO."""
        solar_panel = SolarPanel(
            asset_id="solar_005",
            name="Test Solar Panel 5",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Mock PPO predictor
        mock_predictor = Mock()
        mock_predictor.predict.return_value = np.array([0.7, 0.8, 0.6, 0.9])  # 4-hour forecast

        agent.set_ppo_predictor(mock_predictor)

        # Mock grid conditions
        grid_conditions = {
            "frequency_hz": 60.0,
            "voltage_kv": 138.0,
            "irradiance_w_m2": 800.0,
            "temperature_c": 25.0,
            "weather_condition": WeatherCondition.CLEAR,
        }

        # Get generation forecast
        forecast = agent.forecast_generation(grid_conditions, hours=4)

        # Verify forecast structure
        assert isinstance(forecast, list)
        assert len(forecast) == 4
        assert all(isinstance(f, float) for f in forecast)
        assert all(0.0 <= f <= solar_panel.capacity_mw for f in forecast)

    def test_forecast_caching(self):
        """Test forecast caching mechanism."""
        solar_panel = SolarPanel(
            asset_id="solar_006",
            name="Test Solar Panel 6",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Mock PPO predictor
        mock_predictor = Mock()
        mock_predictor.predict.return_value = np.array([0.5, 0.6, 0.7])

        agent.set_ppo_predictor(mock_predictor)

        grid_conditions = {
            "frequency_hz": 60.0,
            "irradiance_w_m2": 800.0,
            "temperature_c": 25.0,
        }

        # First call should use predictor
        forecast1 = agent.forecast_generation(grid_conditions, hours=3)
        assert mock_predictor.predict.call_count == 1

        # Second call with same conditions should use cache
        forecast2 = agent.forecast_generation(grid_conditions, hours=3)
        assert mock_predictor.predict.call_count == 1  # No additional calls
        assert forecast1 == forecast2

    def test_fallback_forecasting(self):
        """Test fallback forecasting when PPO predictor is not available."""
        solar_panel = SolarPanel(
            asset_id="solar_007",
            name="Test Solar Panel 7",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # No PPO predictor set
        assert agent.ppo_predictor is None

        grid_conditions = {
            "irradiance_w_m2": 800.0,
            "temperature_c": 25.0,
            "weather_condition": WeatherCondition.CLEAR,
        }

        # Should fall back to deterministic forecasting
        forecast = agent.forecast_generation(grid_conditions, hours=3)

        assert isinstance(forecast, list)
        assert len(forecast) == 3
        assert all(isinstance(f, float) for f in forecast)


class TestSolarAgentPheromoneCoordination:
    """Test SolarAgent pheromone coordination."""

    def test_grid_conditions_update(self):
        """Test grid conditions update."""
        solar_panel = SolarPanel(
            asset_id="solar_008",
            name="Test Solar Panel 8",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Update grid conditions
        agent.update_grid_conditions(
            frequency_hz=59.95,
            voltage_kv=138.0,
            local_load_mw=150.0,
            local_generation_mw=140.0,
            electricity_price=75.0,
            irradiance_w_m2=900.0,
            temperature_c=30.0,
            weather_condition=WeatherCondition.CLEAR,
        )

        # Verify grid stress calculation
        assert agent.local_grid_stress > 0.0  # Should detect power imbalance
        assert agent.electricity_price == 75.0
        assert solar_panel.current_irradiance_w_m2 == 900.0
        assert solar_panel.current_temperature_c == 30.0
        assert solar_panel.current_weather_condition == WeatherCondition.CLEAR

    def test_swarm_signal_update(self):
        """Test swarm signal update."""
        solar_panel = SolarPanel(
            asset_id="solar_009",
            name="Test Solar Panel 9",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Initial coordination signal
        initial_signal = agent.coordination_signal

        # Update with neighbor signals
        neighbor_signals = [0.3, 0.5, 0.2, 0.4]
        agent.update_swarm_signals(neighbor_signals)

        # Verify coordination signal update
        assert agent.neighbor_signals == neighbor_signals
        assert agent.coordination_signal != initial_signal
        assert -1.0 <= agent.coordination_signal <= 1.0

    def test_pheromone_strength_calculation(self):
        """Test pheromone strength calculation."""
        solar_panel = SolarPanel(
            asset_id="solar_010",
            name="Test Solar Panel 10",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Execute some control action
        agent.execute_control_action(0.8)  # 80% generation

        # Verify pheromone strength
        pheromone_strength = agent.get_pheromone_strength()
        assert 0.0 <= pheromone_strength <= 1.0

        # Higher generation should create higher pheromone strength
        agent.execute_control_action(0.9)
        new_strength = agent.get_pheromone_strength()
        assert new_strength >= pheromone_strength

    def test_coordination_signal_generation(self):
        """Test coordination signal generation."""
        solar_panel = SolarPanel(
            asset_id="solar_011",
            name="Test Solar Panel 11",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Set some generation
        solar_panel.current_output_mw = 75.0

        # Get coordination signal
        signal = agent.get_coordination_signal()

        assert isinstance(signal, float)
        assert -1.0 <= signal <= 1.0


class TestSolarAgentDemandResponse:
    """Test SolarAgent demand response generation (primary output)."""

    def test_demand_response_calculation(self):
        """Test demand response signal calculation."""
        solar_panel = SolarPanel(
            asset_id="solar_012",
            name="Test Solar Panel 12",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Set conditions for demand response
        agent.local_grid_stress = 0.8  # High stress
        agent.coordination_signal = 0.5  # Positive coordination

        # Mock generation forecast
        mock_predictor = Mock()
        mock_predictor.predict.return_value = np.array([0.7, 0.8, 0.6])
        agent.set_ppo_predictor(mock_predictor)

        # Calculate demand response
        demand_response = agent.calculate_demand_response_signal(
            forecast_irradiance=[800.0, 900.0, 700.0],
            forecast_prices=[60.0, 70.0, 50.0],
            forecast_grid_stress=[0.8, 0.6, 0.4],
            time_horizon_hours=3,
        )

        # Verify demand response structure
        assert isinstance(demand_response, dict)
        assert "signal_mw" in demand_response
        assert "curtailment_factor" in demand_response
        assert "confidence" in demand_response
        assert "reason" in demand_response

        # Verify signal properties
        assert isinstance(demand_response["signal_mw"], float)
        assert 0.0 <= demand_response["curtailment_factor"] <= 1.0
        assert 0.0 <= demand_response["confidence"] <= 1.0
        assert isinstance(demand_response["reason"], str)

    def test_curtailment_based_demand_response(self):
        """Test curtailment-based demand response."""
        solar_panel = SolarPanel(
            asset_id="solar_013",
            name="Test Solar Panel 13",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Set high irradiance conditions
        solar_panel.current_irradiance_w_m2 = 1000.0
        solar_panel.current_temperature_c = 25.0
        solar_panel.current_weather_condition = WeatherCondition.CLEAR

        # Set grid stress (excess generation)
        agent.local_grid_stress = 0.9

        # Calculate curtailment response
        curtailment_response = agent._calculate_curtailment_response()

        # Should suggest curtailment to reduce generation
        assert curtailment_response > 0.0  # Positive curtailment
        assert curtailment_response <= 1.0  # Within bounds

    def test_economic_demand_response(self):
        """Test economic-based demand response."""
        solar_panel = SolarPanel(
            asset_id="solar_014",
            name="Test Solar Panel 14",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Set low electricity price
        agent.electricity_price = 30.0  # Low price

        # Calculate economic response
        economic_response = agent._calculate_economic_response([25.0, 30.0, 35.0])

        # Low prices should encourage curtailment to signal demand increase
        assert economic_response >= 0.0

    def test_frequency_regulation_demand_response(self):
        """Test frequency regulation demand response."""
        solar_panel = SolarPanel(
            asset_id="solar_015",
            name="Test Solar Panel 15",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Set high frequency (excess generation)
        agent.local_grid_stress = 0.7

        # Calculate frequency response
        frequency_response = agent._calculate_frequency_response()

        # Should suggest curtailment to reduce generation
        assert frequency_response >= 0.0

    def test_coordination_demand_response(self):
        """Test coordination-based demand response."""
        solar_panel = SolarPanel(
            asset_id="solar_016",
            name="Test Solar Panel 16",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Set positive coordination signal
        agent.coordination_signal = 0.6

        # Calculate coordination response
        coordination_response = agent._calculate_coordination_response()

        # Should respond to swarm coordination
        assert coordination_response != 0.0

    def test_demand_response_signal_integration(self):
        """Test integration of demand response signals."""
        solar_panel = SolarPanel(
            asset_id="solar_017",
            name="Test Solar Panel 17",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Set solar panel conditions for generation
        solar_panel.set_irradiance(800.0)
        solar_panel.set_temperature(25.0)
        solar_panel.set_weather_condition(WeatherCondition.CLEAR)

        # Set various conditions
        agent.local_grid_stress = 0.5
        agent.coordination_signal = 0.3
        agent.electricity_price = 60.0

        # Test multi-objective demand response
        demand_response = agent.calculate_demand_response_signal(
            forecast_irradiance=[800.0, 900.0, 700.0],
            forecast_prices=[60.0, 70.0, 50.0],
            forecast_grid_stress=[0.5, 0.3, 0.2],
        )

        # Verify integrated response
        assert isinstance(demand_response, dict)
        assert "signal_mw" in demand_response
        assert demand_response["signal_mw"] >= 0.0  # Should produce non-negative signal


class TestSolarAgentSwarmIntegration:
    """Test SolarAgent integration with swarm infrastructure."""

    def test_swarm_bus_integration(self):
        """Test integration with SwarmBus."""
        # Create SwarmBus
        swarm_bus = SwarmBus(
            grid_width=10,
            grid_height=10,
            max_agents=100,
        )

        # Create solar panel and agent
        solar_panel = SolarPanel(
            asset_id="solar_018",
            name="Test Solar Panel 18",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Register agent with swarm bus
        position = GridPosition(x=5, y=5)
        success = swarm_bus.register_agent(agent, position)

        assert success is True
        assert agent.agent_id in swarm_bus.registered_agents
        assert swarm_bus.agent_positions[agent.agent_id] == position

    def test_pheromone_field_interaction(self):
        """Test interaction with pheromone field."""
        # Create pheromone field
        pheromone_field = PheromoneField(
            grid_width=10,
            grid_height=10,
            decay_rate=0.99,
            diffusion_rate=0.05,
        )

        # Create solar panel and agent
        solar_panel = SolarPanel(
            asset_id="solar_019",
            name="Test Solar Panel 19",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Execute control action to generate pheromones
        agent.execute_control_action(0.8)

        # Deposit pheromones
        position = GridPosition(x=5, y=5)
        strength = agent.get_pheromone_strength()

        pheromone_field.deposit_pheromone(
            position=position,
            pheromone_type=PheromoneType.RENEWABLE_CURTAILMENT,
            strength=strength,
        )

        # Verify pheromone deposit
        retrieved_strength = pheromone_field.get_pheromone_strength(
            position=position,
            pheromone_type=PheromoneType.RENEWABLE_CURTAILMENT,
        )

        assert retrieved_strength > 0.0


class TestSolarAgentPerformance:
    """Test SolarAgent performance and edge cases."""

    def test_large_scale_coordination(self):
        """Test large-scale coordination performance."""
        solar_panel = SolarPanel(
            asset_id="solar_020",
            name="Test Solar Panel 20",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Simulate large number of neighbor signals
        large_neighbor_signals = [0.1 * i for i in range(100)]

        # Should handle large coordination efficiently
        agent.update_swarm_signals(large_neighbor_signals)

        assert len(agent.neighbor_signals) == 100
        assert agent.coordination_signal != 0.0

    def test_extreme_conditions(self):
        """Test agent behavior under extreme conditions."""
        solar_panel = SolarPanel(
            asset_id="solar_021",
            name="Test Solar Panel 21",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Test extreme grid stress
        agent.local_grid_stress = 1.0

        # Test extreme coordination signals
        agent.coordination_signal = 1.0

        # Test extreme weather conditions
        agent.update_grid_conditions(
            frequency_hz=58.0,  # Very low frequency
            voltage_kv=100.0,  # Low voltage
            local_load_mw=1000.0,
            local_generation_mw=500.0,  # Large imbalance
            electricity_price=200.0,  # High price
            irradiance_w_m2=1400.0,  # High irradiance
            temperature_c=50.0,  # High temperature
            weather_condition=WeatherCondition.STORMY,
        )

        # Should handle extreme conditions gracefully
        demand_response = agent.calculate_demand_response_signal(
            forecast_irradiance=[1400.0, 1200.0, 1000.0],
            forecast_prices=[200.0, 180.0, 160.0],
            forecast_grid_stress=[1.0, 0.9, 0.8],
        )

        assert isinstance(demand_response, dict)
        assert "signal_mw" in demand_response
        assert np.isfinite(demand_response["signal_mw"])

    def test_reset_functionality(self):
        """Test agent reset functionality."""
        solar_panel = SolarPanel(
            asset_id="solar_022",
            name="Test Solar Panel 22",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Set some state
        agent.local_grid_stress = 0.8
        agent.coordination_signal = 0.5
        agent.pheromone_strength = 0.3
        agent.neighbor_signals = [0.1, 0.2, 0.3]

        # Reset agent
        agent.reset()

        # Verify reset state
        assert agent.local_grid_stress == 0.0
        assert agent.coordination_signal == 0.0
        assert agent.pheromone_strength == 0.0
        assert agent.neighbor_signals == []
        assert agent.target_curtailment_factor == 0.0

    def test_string_representations(self):
        """Test string representations."""
        solar_panel = SolarPanel(
            asset_id="solar_023",
            name="Test Solar Panel 23",
            node_id="solar_node",
            capacity_mw=100.0,
        )

        from psireg.swarm.agents.solar_agent import SolarAgent

        agent = SolarAgent(solar_panel=solar_panel)

        # Test string representation
        str_repr = str(agent)
        assert "SolarAgent" in str_repr
        assert "solar_023" in str_repr

        # Test detailed representation
        repr_str = repr(agent)
        assert "SolarAgent" in repr_str
        assert "solar_023" in repr_str
        assert "100.0" in repr_str  # Capacity
