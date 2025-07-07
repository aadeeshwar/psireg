"""Integration tests for renewable agents swarm system.

This module tests the complete integration of Solar and Wind agents with
PPO forecasting, pheromone coordination, and demand response generation.
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest
from psireg.sim.assets import SolarPanel, WindTurbine
from psireg.swarm import PheromoneField, SwarmBus
from psireg.swarm.agents import SolarAgent, WindAgent
from psireg.utils.enums import WeatherCondition

logger = logging.getLogger(__name__)


class MockPPOPredictor:
    """Mock PPO predictor for testing."""

    def __init__(self, agent_type: str = "SolarAgent"):
        self.agent_type = agent_type
        self.prediction_calls = []

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Mock prediction that returns reasonable values."""
        self.prediction_calls.append((observation.copy(), deterministic))

        # Return predictable values for testing
        if self.agent_type == "SolarAgent":
            # Solar prediction - higher during day
            return np.array([0.7])
        elif self.agent_type == "WindAgent":
            # Wind prediction - consistent output
            return np.array([0.8])
        else:
            return np.array([0.5])


class TestRenewableAgentsIntegration:
    """Integration tests for renewable agents swarm system."""

    def setup_method(self):
        """Set up test environment."""
        self.swarm_bus = SwarmBus(grid_width=10, grid_height=10)
        self.pheromone_field = PheromoneField(grid_width=5, grid_height=5)

        # Create test solar agent
        self.solar_panel = SolarPanel(
            asset_id="test_solar",
            name="Test Solar Panel",
            node_id="solar_node",
            capacity_mw=2.0,
            panel_efficiency=0.2,
            panel_area_m2=10000,
        )
        self.solar_agent = SolarAgent(solar_panel=self.solar_panel)

        # Create test wind agent
        self.wind_turbine = WindTurbine(
            asset_id="test_wind",
            name="Test Wind Turbine",
            node_id="wind_node",
            capacity_mw=3.0,
            rotor_diameter_m=120.0,
            hub_height_m=100.0,
        )
        self.wind_agent = WindAgent(wind_turbine=self.wind_turbine)

        # Set up PPO predictors
        self.solar_predictor = MockPPOPredictor("SolarAgent")
        self.wind_predictor = MockPPOPredictor("WindAgent")

        self.solar_agent.set_ppo_predictor(self.solar_predictor)
        self.wind_agent.set_ppo_predictor(self.wind_predictor)

        logger.info("Test setup completed")

    def test_agent_initialization(self):
        """Test that agents are properly initialized."""
        assert self.solar_agent.agent_id == "test_solar"
        assert self.wind_agent.agent_id == "test_wind"
        assert self.solar_agent.ppo_predictor is not None
        assert self.wind_agent.ppo_predictor is not None
        assert self.solar_agent.coordination_weight == 0.3
        assert self.wind_agent.coordination_weight == 0.3

    def test_ppo_integration(self):
        """Test PPO predictor integration."""
        # Test solar agent PPO integration
        grid_conditions = {
            "frequency_hz": 60.0,
            "voltage_kv": 138.0,
            "irradiance_w_m2": 800.0,
            "ambient_temp_c": 25.0,
            "weather_condition": WeatherCondition.CLEAR,
        }

        solar_forecast = self.solar_agent.forecast_generation(grid_conditions, hours=24)
        assert len(solar_forecast) == 24
        assert all(isinstance(x, int | float) for x in solar_forecast)
        assert len(self.solar_predictor.prediction_calls) > 0

        # Test wind agent PPO integration
        wind_conditions = {
            "frequency_hz": 60.0,
            "voltage_kv": 138.0,
            "wind_speed_ms": 12.0,
            "air_density_kg_m3": 1.225,
            "weather_condition": WeatherCondition.WINDY,
        }

        wind_forecast = self.wind_agent.forecast_generation(wind_conditions, hours=24)
        assert len(wind_forecast) == 24
        assert all(isinstance(x, int | float) for x in wind_forecast)
        assert len(self.wind_predictor.prediction_calls) > 0

    def test_pheromone_coordination(self):
        """Test pheromone-based coordination between agents."""
        # Set up grid conditions
        self.solar_agent.update_grid_conditions(
            frequency_hz=60.05,  # Slightly high frequency
            voltage_kv=138.0,
            local_load_mw=100.0,
            local_generation_mw=90.0,
            electricity_price=80.0,
            irradiance_w_m2=900.0,
            temperature_c=25.0,
        )

        self.wind_agent.update_grid_conditions(
            frequency_hz=60.05,
            voltage_kv=138.0,
            local_load_mw=100.0,
            local_generation_mw=90.0,
            electricity_price=80.0,
            wind_speed_ms=15.0,
            air_density_kg_m3=1.225,
        )

        # Test swarm signal exchange
        solar_signal = self.solar_agent.get_coordination_signal()
        wind_signal = self.wind_agent.get_coordination_signal()

        # Update agents with neighbor signals
        self.solar_agent.update_swarm_signals([wind_signal])
        self.wind_agent.update_swarm_signals([solar_signal])

        # Check that coordination signals are within expected range
        assert -1.0 <= solar_signal <= 1.0
        assert -1.0 <= wind_signal <= 1.0

        # Check that agents have updated their coordination state
        assert self.solar_agent.coordination_signal != 0.0
        assert self.wind_agent.coordination_signal != 0.0

    def test_demand_response_generation(self):
        """Test demand response signal generation."""
        # Set up realistic conditions
        self.solar_panel.set_irradiance(800.0)
        self.solar_panel.set_temperature(25.0)
        self.wind_turbine.set_wind_speed(12.0)

        # Update grid conditions
        self.solar_agent.update_grid_conditions(
            frequency_hz=59.95,  # Low frequency requires response
            voltage_kv=138.0,
            local_load_mw=120.0,
            local_generation_mw=100.0,
            electricity_price=70.0,
        )

        self.wind_agent.update_grid_conditions(
            frequency_hz=59.95, voltage_kv=138.0, local_load_mw=120.0, local_generation_mw=100.0, electricity_price=70.0
        )

        # Generate forecasts
        forecast_hours = 24
        solar_forecast_irradiance = [800.0 * max(0, np.sin(np.pi * h / 12)) for h in range(forecast_hours)]
        wind_forecast_speed = [12.0 for h in range(forecast_hours)]
        forecast_prices = [70.0 + 10.0 * np.sin(2 * np.pi * h / 24) for h in range(forecast_hours)]
        forecast_grid_stress = [0.3 for h in range(forecast_hours)]

        # Test solar agent demand response
        solar_dr = self.solar_agent.calculate_demand_response_signal(
            forecast_irradiance=solar_forecast_irradiance,
            forecast_prices=forecast_prices,
            forecast_grid_stress=forecast_grid_stress,
            time_horizon_hours=forecast_hours,
        )

        # Test wind agent demand response
        wind_dr = self.wind_agent.calculate_demand_response_signal(
            forecast_wind_speed=wind_forecast_speed,
            forecast_prices=forecast_prices,
            forecast_grid_stress=forecast_grid_stress,
            time_horizon_hours=forecast_hours,
        )

        # Validate demand response structure
        required_keys = ["signal_mw", "curtailment_factor", "confidence", "reason"]
        for key in required_keys:
            assert key in solar_dr
            assert key in wind_dr

        # Validate demand response values
        assert isinstance(solar_dr["signal_mw"], int | float)
        assert isinstance(wind_dr["signal_mw"], int | float)
        assert 0.0 <= solar_dr["curtailment_factor"] <= 1.0
        assert 0.0 <= wind_dr["curtailment_factor"] <= 1.0
        assert 0.0 <= solar_dr["confidence"] <= 1.0
        assert 0.0 <= wind_dr["confidence"] <= 1.0
        assert isinstance(solar_dr["reason"], str)
        assert isinstance(wind_dr["reason"], str)

    def test_multi_agent_coordination(self):
        """Test coordination between multiple agents."""
        # Create additional agents
        solar_agent_2 = SolarAgent(
            solar_panel=SolarPanel(
                asset_id="test_solar_2",
                name="Test Solar Panel 2",
                node_id="solar_node_2",
                capacity_mw=2.0,
                panel_efficiency=0.2,
                panel_area_m2=10000,
            ),
            agent_id="solar_agent_2",
            communication_range=2.0,
            coordination_weight=0.4,
        )

        wind_agent_2 = WindAgent(
            wind_turbine=WindTurbine(
                asset_id="test_wind_2",
                name="Test Wind Turbine 2",
                node_id="wind_node_2",
                capacity_mw=1.5,
                rotor_diameter_m=80.0,
                hub_height_m=60.0,
            ),
            agent_id="wind_agent_2",
            communication_range=2.0,
            coordination_weight=0.4,
        )

        # Set up PPO predictors
        solar_agent_2.set_ppo_predictor(MockPPOPredictor("SolarAgent"))
        wind_agent_2.set_ppo_predictor(MockPPOPredictor("WindAgent"))

        agents = [self.solar_agent, self.wind_agent, solar_agent_2, wind_agent_2]

        # Update all agents with same grid conditions
        for agent in agents:
            agent.update_grid_conditions(
                frequency_hz=60.02,
                voltage_kv=138.0,
                local_load_mw=100.0,
                local_generation_mw=95.0,
                electricity_price=65.0,
            )

        # Generate coordination signals
        coordination_signals = {}
        for agent in agents:
            coordination_signals[agent.agent_id] = agent.get_coordination_signal()

        # Update each agent with neighbor signals
        for agent in agents:
            neighbor_signals = [
                coordination_signals[other_agent.agent_id]
                for other_agent in agents
                if other_agent.agent_id != agent.agent_id
            ]
            agent.update_swarm_signals(neighbor_signals)

        # Check that all agents have been influenced by neighbors
        for agent in agents:
            assert agent.coordination_signal != 0.0

    def test_demand_response_under_stress(self):
        """Test demand response under grid stress conditions."""
        # Set up high stress conditions
        high_stress_conditions = {
            "frequency_hz": 59.85,  # Very low frequency
            "voltage_kv": 135.0,  # Lower voltage
            "local_load_mw": 150.0,  # High load
            "local_generation_mw": 80.0,  # Low generation
            "electricity_price": 120.0,  # High price
        }

        # Update agents
        self.solar_agent.update_grid_conditions(**high_stress_conditions)
        self.wind_agent.update_grid_conditions(**high_stress_conditions)

        # Set good generation conditions
        self.solar_panel.set_irradiance(1000.0)
        self.wind_turbine.set_wind_speed(15.0)

        # Generate demand response
        forecast_hours = 6
        solar_dr = self.solar_agent.calculate_demand_response_signal(
            forecast_irradiance=[1000.0] * forecast_hours,
            forecast_prices=[120.0] * forecast_hours,
            forecast_grid_stress=[0.8] * forecast_hours,
            time_horizon_hours=forecast_hours,
        )

        wind_dr = self.wind_agent.calculate_demand_response_signal(
            forecast_wind_speed=[15.0] * forecast_hours,
            forecast_prices=[120.0] * forecast_hours,
            forecast_grid_stress=[0.8] * forecast_hours,
            time_horizon_hours=forecast_hours,
        )

        # Under stress, agents should provide demand response
        assert solar_dr["signal_mw"] >= 0.0
        assert wind_dr["signal_mw"] >= 0.0

    def test_agent_state_management(self):
        """Test agent state management and coordination."""
        # Get initial states
        solar_state = self.solar_agent.get_agent_state()
        wind_state = self.wind_agent.get_agent_state()

        # Check state structure
        assert solar_state.agent_id == "test_solar"
        assert wind_state.agent_id == "test_wind"
        assert solar_state.current_generation_mw >= 0.0
        assert wind_state.current_generation_mw >= 0.0
        assert solar_state.capacity_mw > 0.0
        assert wind_state.capacity_mw > 0.0
        assert 0.0 <= solar_state.grid_support_priority <= 1.0
        assert 0.0 <= wind_state.grid_support_priority <= 1.0

        # Update conditions and check state changes
        self.solar_panel.set_irradiance(500.0)  # Reduce irradiance
        self.wind_turbine.set_wind_speed(8.0)  # Reduce wind speed

        new_solar_state = self.solar_agent.get_agent_state()
        new_wind_state = self.wind_agent.get_agent_state()

        # States should reflect new conditions
        assert new_solar_state.current_irradiance_w_m2 == 500.0
        assert new_wind_state.current_wind_speed_ms == 8.0

    def test_control_execution(self):
        """Test control action execution."""
        # Test solar agent control
        initial_curtailment = self.solar_panel.curtailment_factor
        self.solar_agent.execute_control_action(0.7)  # 70% generation
        new_curtailment = self.solar_panel.curtailment_factor
        assert new_curtailment != initial_curtailment

        # Test wind agent control
        initial_wind_curtailment = self.wind_turbine.curtailment_factor
        self.wind_agent.execute_control_action(0.8)  # 80% generation
        new_wind_curtailment = self.wind_turbine.curtailment_factor
        assert new_wind_curtailment != initial_wind_curtailment

        # Check pheromone strength is updated
        assert self.solar_agent.pheromone_strength >= 0.0
        assert self.wind_agent.pheromone_strength >= 0.0

    def test_forecast_caching(self):
        """Test forecast caching functionality."""
        grid_conditions = {"frequency_hz": 60.0, "voltage_kv": 138.0, "irradiance_w_m2": 800.0, "ambient_temp_c": 25.0}

        # First forecast call
        forecast1 = self.solar_agent.forecast_generation(grid_conditions, hours=24)
        initial_calls = len(self.solar_predictor.prediction_calls)

        # Second forecast call with same conditions (should use cache)
        forecast2 = self.solar_agent.forecast_generation(grid_conditions, hours=24)
        final_calls = len(self.solar_predictor.prediction_calls)

        # Should have same result and not make additional PPO calls
        assert forecast1 == forecast2
        assert final_calls == initial_calls  # No new PPO calls

        # Cache should contain the forecast
        assert len(self.solar_agent.forecast_cache) > 0

    def test_reset_functionality(self):
        """Test agent reset functionality."""
        # Modify agent state
        self.solar_agent.target_curtailment_factor = 0.5
        self.solar_agent.pheromone_strength = 0.8
        self.solar_agent.coordination_signal = 0.6
        self.solar_agent.forecast_cache["test"] = [1.0, 2.0, 3.0]

        # Reset agent
        self.solar_agent.reset()

        # Check that state is reset
        assert self.solar_agent.target_curtailment_factor == 0.0
        assert self.solar_agent.pheromone_strength == 0.0
        assert self.solar_agent.coordination_signal == 0.0
        assert len(self.solar_agent.forecast_cache) == 0
        assert len(self.solar_agent.neighbor_signals) == 0

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with None PPO predictor
        self.solar_agent.ppo_predictor = None

        grid_conditions = {
            "frequency_hz": 60.0,
            "voltage_kv": 138.0,
            "irradiance_w_m2": 800.0,
        }

        # Should fall back to physics-based forecasting
        forecast = self.solar_agent.forecast_generation(grid_conditions, hours=24)
        assert len(forecast) == 24
        assert all(isinstance(x, int | float) for x in forecast)

        # Test with empty forecasts
        empty_dr = self.solar_agent.calculate_demand_response_signal(
            forecast_irradiance=[], forecast_prices=[], forecast_grid_stress=[], time_horizon_hours=0
        )
        assert "signal_mw" in empty_dr
        assert "curtailment_factor" in empty_dr


class TestRenewableAgentsDemo:
    """Integration tests for the renewable agents demo."""

    def test_demo_import(self):
        """Test that the demo can be imported."""
        try:
            from examples.renewable_agents_demo import RenewableAgentsDemo

            demo = RenewableAgentsDemo()
            assert len(demo.agents) > 0
        except ImportError:
            pytest.skip("Demo not available for testing")

    def test_demo_execution(self, mock_savefig=None, mock_show=None):
        """Test demo execution."""
        try:
            from examples.renewable_agents_demo import RenewableAgentsDemo

            # Only patch matplotlib if it's available
            try:
                import matplotlib.pyplot as plt

                with patch.object(plt, "show") as _, patch.object(plt, "savefig") as _:
                    # Create demo
                    demo = RenewableAgentsDemo()

                    # Run short demo
                    results = demo.run_coordination_demo(time_steps=6)

                    # Check results
                    assert len(results) == 6
                    assert all("demand_responses" in result for result in results)
                    assert all("total_demand_response_mw" in result for result in results)

                    # Test summary
                    demo.print_summary()

                    # Test visualization (mocked)
                    demo.create_visualizations()
            except ImportError:
                # matplotlib not available, run without visualization
                demo = RenewableAgentsDemo()
                results = demo.run_coordination_demo(time_steps=6)

                # Check results
                assert len(results) == 6
                assert all("demand_responses" in result for result in results)
                assert all("total_demand_response_mw" in result for result in results)

                # Test summary
                demo.print_summary()

        except ImportError:
            pytest.skip("Demo not available for testing")

    def test_mock_ppo_predictor(self):
        """Test the mock PPO predictor."""
        try:
            from examples.renewable_agents_demo import MockPPOPredictor

            # Test solar predictor
            solar_predictor = MockPPOPredictor("SolarAgent")
            obs = np.array([60.0, 138.0, 0.8, 25.0, 0.5, 0.3, 0.2, 0.1])
            result = solar_predictor.predict(obs)
            assert len(result) == 1
            assert 0.0 <= result[0] <= 1.0

            # Test wind predictor
            wind_predictor = MockPPOPredictor("WindAgent")
            result = wind_predictor.predict(obs)
            assert len(result) == 1
            assert 0.0 <= result[0] <= 1.0

        except ImportError:
            pytest.skip("Demo not available for testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
