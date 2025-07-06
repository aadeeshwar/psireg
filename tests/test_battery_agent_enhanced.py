"""Test enhanced BatteryAgent functionality.

This module contains comprehensive tests for the enhanced BatteryAgent functionality including:
- Voltage trigger functionality with deadbands and response curves
- Enhanced pheromone sensitivity with different response types
- Local stabilization signal generation as primary output
- Integration with existing BatteryAgent functionality
"""

import pytest
from psireg.sim.assets.battery import Battery
from psireg.swarm.agents.battery_agent import BatteryAgent
from psireg.swarm.pheromone import GridPosition, PheromoneType, SwarmBus
from psireg.utils.enums import AssetStatus


class TestBatteryAgentVoltageTriggers:
    """Test BatteryAgent voltage trigger functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.battery = Battery(
            asset_id="test_battery",
            name="Test Battery",
            node_id="test_node",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
            nominal_voltage_v=800.0,
            min_voltage_v=700.0,
            max_voltage_v=900.0,
        )
        self.battery.set_status(AssetStatus.ONLINE)
        self.agent = BatteryAgent(battery=self.battery)

    def test_voltage_trigger_initialization(self):
        """Test voltage trigger parameters are properly initialized."""
        # Test default voltage trigger parameters
        assert hasattr(self.agent, "voltage_deadband_v")
        assert hasattr(self.agent, "voltage_trigger_sensitivity")
        assert hasattr(self.agent, "voltage_regulation_weight")
        assert hasattr(self.agent, "nominal_voltage_v")

        # Test defaults
        assert self.agent.voltage_deadband_v == 10.0  # ±10V deadband
        assert self.agent.voltage_trigger_sensitivity == 0.5  # 50% sensitivity
        assert self.agent.voltage_regulation_weight == 0.4  # 40% weight for voltage regulation
        assert self.agent.nominal_voltage_v == 800.0  # From battery

    def test_voltage_trigger_configuration(self):
        """Test voltage trigger configuration."""
        # Test custom voltage trigger parameters
        agent = BatteryAgent(
            battery=self.battery,
            voltage_deadband_v=15.0,
            voltage_trigger_sensitivity=0.8,
            voltage_regulation_weight=0.6,
        )

        assert agent.voltage_deadband_v == 15.0
        assert agent.voltage_trigger_sensitivity == 0.8
        assert agent.voltage_regulation_weight == 0.6

    def test_voltage_deviation_calculation(self):
        """Test voltage deviation calculation."""
        # Test high voltage scenario
        self.agent.update_grid_conditions(
            frequency_hz=60.0,
            voltage_kv=138.5,  # High voltage
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        # Expected nominal voltage in kV
        nominal_kv = self.agent.nominal_voltage_v / 1000.0  # Convert to kV
        expected_deviation = 138.5 - nominal_kv

        deviation = self.agent.calculate_voltage_deviation()
        assert abs(deviation - expected_deviation) < 0.001

    def test_voltage_trigger_deadband(self):
        """Test voltage trigger deadband functionality."""
        # Test voltage within deadband (no response)
        self.agent.update_grid_conditions(
            frequency_hz=60.0,
            voltage_kv=0.805,  # Within deadband (±10V = ±0.01kV)
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        voltage_response = self.agent.calculate_voltage_response()
        assert abs(voltage_response) < 0.001  # Should be near zero

    def test_voltage_trigger_high_voltage_response(self):
        """Test voltage trigger response to high voltage."""
        # Test high voltage (above deadband)
        self.agent.update_grid_conditions(
            frequency_hz=60.0,
            voltage_kv=0.850,  # High voltage (above deadband)
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        voltage_response = self.agent.calculate_voltage_response()

        # Should be positive (charging) to absorb reactive power
        assert voltage_response > 0.0
        assert voltage_response <= self.battery.capacity_mw * 0.5  # Reasonable limit

    def test_voltage_trigger_low_voltage_response(self):
        """Test voltage trigger response to low voltage."""
        # Test low voltage (below deadband)
        self.agent.update_grid_conditions(
            frequency_hz=60.0,
            voltage_kv=0.750,  # Low voltage (below deadband)
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        voltage_response = self.agent.calculate_voltage_response()

        # Should be negative (discharging) to provide reactive power
        assert voltage_response < 0.0
        assert voltage_response >= -self.battery.capacity_mw * 0.5  # Reasonable limit

    def test_voltage_trigger_sensitivity_scaling(self):
        """Test voltage trigger sensitivity scaling."""
        # Test with different sensitivity values
        sensitivities = [0.2, 0.5, 0.8, 1.0]
        responses = []

        for sensitivity in sensitivities:
            self.agent.voltage_trigger_sensitivity = sensitivity
            self.agent.update_grid_conditions(
                frequency_hz=60.0,
                voltage_kv=0.850,  # High voltage
                local_load_mw=100.0,
                local_generation_mw=90.0,
            )

            response = self.agent.calculate_voltage_response()
            responses.append(response)

        # Higher sensitivity should give larger response
        for i in range(len(responses) - 1):
            assert responses[i] <= responses[i + 1]

    def test_voltage_trigger_soc_limitations(self):
        """Test voltage trigger response limitations based on SoC."""
        # Test at low SoC (limited discharge capability)
        self.battery.current_soc_percent = 15.0
        self.agent.update_grid_conditions(
            frequency_hz=60.0,
            voltage_kv=0.750,  # Low voltage (requires discharge)
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        voltage_response = self.agent.calculate_voltage_response()

        # Response should be limited by low SoC
        assert abs(voltage_response) < self.battery.capacity_mw * 0.5

        # Test at high SoC (limited charge capability)
        self.battery.current_soc_percent = 95.0
        self.agent.update_grid_conditions(
            frequency_hz=60.0,
            voltage_kv=0.850,  # High voltage (requires charge)
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        voltage_response = self.agent.calculate_voltage_response()

        # Response should be limited by high SoC
        assert abs(voltage_response) < self.battery.capacity_mw * 0.5

    def test_voltage_trigger_temperature_effects(self):
        """Test voltage trigger response considering temperature effects."""
        # Test at different temperatures
        temperatures = [10.0, 25.0, 40.0]
        power_limits = []

        for temp in temperatures:
            self.battery.set_temperature(temp)
            self.agent.update_grid_conditions(
                frequency_hz=60.0,
                voltage_kv=0.850,  # High voltage
                local_load_mw=100.0,
                local_generation_mw=90.0,
            )

            # Check that temperature affects battery power limits
            max_charge = self.battery.get_max_charge_power()
            power_limits.append(max_charge)

        # Power limits should be affected by temperature
        assert len(set(power_limits)) > 1  # Should have different power limits at different temperatures


class TestBatteryAgentEnhancedPheromones:
    """Test enhanced pheromone sensitivity functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.battery = Battery(
            asset_id="test_battery",
            name="Test Battery",
            node_id="test_node",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        self.battery.set_status(AssetStatus.ONLINE)
        self.agent = BatteryAgent(battery=self.battery)

    def test_enhanced_pheromone_initialization(self):
        """Test enhanced pheromone parameters are properly initialized."""
        # Test pheromone sensitivity parameters
        assert hasattr(self.agent, "pheromone_sensitivity_types")
        assert hasattr(self.agent, "pheromone_response_weights")
        assert hasattr(self.agent, "pheromone_decay_factor")
        assert hasattr(self.agent, "pheromone_gradient_threshold")

        # Test default values
        assert isinstance(self.agent.pheromone_sensitivity_types, dict)
        assert PheromoneType.FREQUENCY_SUPPORT in self.agent.pheromone_sensitivity_types
        assert PheromoneType.COORDINATION in self.agent.pheromone_sensitivity_types
        assert PheromoneType.RENEWABLE_CURTAILMENT in self.agent.pheromone_sensitivity_types
        assert PheromoneType.EMERGENCY_RESPONSE in self.agent.pheromone_sensitivity_types

    def test_pheromone_type_specific_responses(self):
        """Test different response types for different pheromone types."""
        # Test frequency support pheromone response
        self.agent.update_pheromone_gradients(
            {
                PheromoneType.FREQUENCY_SUPPORT: 0.8,
                PheromoneType.COORDINATION: 0.2,
                PheromoneType.RENEWABLE_CURTAILMENT: 0.1,
            }
        )

        frequency_response = self.agent.calculate_pheromone_response(PheromoneType.FREQUENCY_SUPPORT)
        coordination_response = self.agent.calculate_pheromone_response(PheromoneType.COORDINATION)
        curtailment_response = self.agent.calculate_pheromone_response(PheromoneType.RENEWABLE_CURTAILMENT)

        # Different pheromone types should have different responses
        assert frequency_response != coordination_response
        assert frequency_response != curtailment_response
        assert coordination_response != curtailment_response

    def test_pheromone_gradient_calculation(self):
        """Test pheromone gradient calculation with neighbors."""
        # Mock neighbor pheromone strengths
        neighbor_strengths = {
            PheromoneType.FREQUENCY_SUPPORT: [0.5, 0.7, 0.3, 0.8],
            PheromoneType.COORDINATION: [0.2, 0.4, 0.6, 0.1],
        }

        self.agent.update_neighbor_pheromone_strengths(neighbor_strengths)

        # Calculate gradients
        gradients = self.agent.calculate_pheromone_gradients()

        # Should have gradients for each pheromone type
        assert PheromoneType.FREQUENCY_SUPPORT in gradients
        assert PheromoneType.COORDINATION in gradients

        # Gradients should be reasonable values
        assert -1.0 <= gradients[PheromoneType.FREQUENCY_SUPPORT] <= 1.0
        assert -1.0 <= gradients[PheromoneType.COORDINATION] <= 1.0

    def test_pheromone_directional_response(self):
        """Test pheromone directional response based on gradient direction."""
        # Test positive gradient (moving toward higher pheromone concentration)
        self.agent.update_pheromone_gradients(
            {
                PheromoneType.FREQUENCY_SUPPORT: 0.5,  # Positive gradient
            }
        )

        positive_response = self.agent.calculate_pheromone_response(PheromoneType.FREQUENCY_SUPPORT)

        # Test negative gradient (moving away from higher pheromone concentration)
        self.agent.update_pheromone_gradients(
            {
                PheromoneType.FREQUENCY_SUPPORT: -0.5,  # Negative gradient
            }
        )

        negative_response = self.agent.calculate_pheromone_response(PheromoneType.FREQUENCY_SUPPORT)

        # Responses should be opposite in direction
        assert (positive_response > 0) != (negative_response > 0)

    def test_pheromone_threshold_behavior(self):
        """Test pheromone response threshold behavior."""
        # Test below threshold (should have minimal response)
        self.agent.update_pheromone_gradients(
            {
                PheromoneType.FREQUENCY_SUPPORT: 0.05,  # Below threshold
            }
        )

        below_threshold_response = self.agent.calculate_pheromone_response(PheromoneType.FREQUENCY_SUPPORT)

        # Test above threshold (should have significant response)
        self.agent.update_pheromone_gradients(
            {
                PheromoneType.FREQUENCY_SUPPORT: 0.3,  # Above threshold
            }
        )

        above_threshold_response = self.agent.calculate_pheromone_response(PheromoneType.FREQUENCY_SUPPORT)

        # Above threshold should have larger response
        assert abs(above_threshold_response) > abs(below_threshold_response)

    def test_pheromone_memory_and_decay(self):
        """Test pheromone memory and decay functionality."""
        # Set initial pheromone strength
        self.agent.update_pheromone_gradients(
            {
                PheromoneType.FREQUENCY_SUPPORT: 0.8,
            }
        )

        initial_strength = self.agent.pheromone_memory[PheromoneType.FREQUENCY_SUPPORT]

        # Update with no new pheromone (should decay)
        self.agent.update_pheromone_gradients({})

        decayed_strength = self.agent.pheromone_memory[PheromoneType.FREQUENCY_SUPPORT]

        # Should decay toward zero
        assert abs(decayed_strength) < abs(initial_strength)

    def test_pheromone_multi_objective_integration(self):
        """Test pheromone integration with multi-objective optimization."""
        # Set up multiple pheromone types
        self.agent.update_pheromone_gradients(
            {
                PheromoneType.FREQUENCY_SUPPORT: 0.6,
                PheromoneType.COORDINATION: 0.4,
                PheromoneType.RENEWABLE_CURTAILMENT: 0.2,
            }
        )

        # Calculate combined pheromone response
        combined_response = self.agent.calculate_combined_pheromone_response()

        # Should be a weighted combination of individual responses
        assert isinstance(combined_response, float)
        assert -self.battery.capacity_mw <= combined_response <= self.battery.capacity_mw

    def test_pheromone_spatial_awareness(self):
        """Test pheromone spatial awareness with distance weighting."""
        # Mock neighbor distances and pheromone strengths
        neighbor_data = [
            {"distance": 1.0, "pheromone": {PheromoneType.FREQUENCY_SUPPORT: 0.8}},
            {"distance": 3.0, "pheromone": {PheromoneType.FREQUENCY_SUPPORT: 0.6}},
            {"distance": 5.0, "pheromone": {PheromoneType.FREQUENCY_SUPPORT: 0.4}},
        ]

        self.agent.update_spatial_pheromone_data(neighbor_data)

        # Calculate distance-weighted pheromone response
        spatial_response = self.agent.calculate_spatial_pheromone_response()

        # Closer neighbors should have more influence
        assert isinstance(spatial_response, float)


class TestBatteryAgentLocalStabilization:
    """Test local stabilization signal generation as primary output."""

    def setup_method(self):
        """Set up test environment."""
        self.battery = Battery(
            asset_id="test_battery",
            name="Test Battery",
            node_id="test_node",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        self.battery.set_status(AssetStatus.ONLINE)
        self.agent = BatteryAgent(battery=self.battery)

    def test_local_stabilization_signal_structure(self):
        """Test local stabilization signal structure and components."""
        # Update conditions for stabilization calculation
        self.agent.update_grid_conditions(
            frequency_hz=60.05,  # Slightly high frequency
            voltage_kv=0.820,  # Slightly high voltage
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        # Calculate local stabilization signal
        stabilization_signal = self.agent.calculate_local_stabilization_signal()

        # Verify signal structure
        assert isinstance(stabilization_signal, dict)
        assert "power_mw" in stabilization_signal
        assert "voltage_support_mw" in stabilization_signal
        assert "frequency_support_mw" in stabilization_signal
        assert "pheromone_coordination_mw" in stabilization_signal
        assert "confidence" in stabilization_signal
        assert "priority" in stabilization_signal
        assert "response_time_s" in stabilization_signal

    def test_local_stabilization_frequency_component(self):
        """Test frequency stabilization component."""
        # Test high frequency scenario
        self.agent.update_grid_conditions(
            frequency_hz=60.1,  # High frequency
            voltage_kv=0.800,
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()

        # Should suggest charging (positive power) to absorb excess generation
        assert stabilization_signal["frequency_support_mw"] > 0.0
        assert stabilization_signal["power_mw"] > 0.0

        # Test low frequency scenario
        self.agent.update_grid_conditions(
            frequency_hz=59.9,  # Low frequency
            voltage_kv=0.800,
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()

        # Should suggest discharging (negative power) to provide additional generation
        assert stabilization_signal["frequency_support_mw"] < 0.0
        assert stabilization_signal["power_mw"] < 0.0

    def test_local_stabilization_voltage_component(self):
        """Test voltage stabilization component."""
        # Test high voltage scenario
        self.agent.update_grid_conditions(
            frequency_hz=60.0,
            voltage_kv=0.850,  # High voltage
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()

        # Should suggest charging to absorb reactive power
        assert stabilization_signal["voltage_support_mw"] > 0.0

        # Test low voltage scenario
        self.agent.update_grid_conditions(
            frequency_hz=60.0,
            voltage_kv=0.750,  # Low voltage
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()

        # Should suggest discharging to provide reactive power
        assert stabilization_signal["voltage_support_mw"] < 0.0

    def test_local_stabilization_pheromone_component(self):
        """Test pheromone coordination component in stabilization."""
        # Set up pheromone gradients
        self.agent.update_pheromone_gradients(
            {
                PheromoneType.FREQUENCY_SUPPORT: 0.7,
                PheromoneType.COORDINATION: 0.3,
            }
        )

        self.agent.update_grid_conditions(
            frequency_hz=60.0,
            voltage_kv=0.800,
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()

        # Should have pheromone coordination component
        assert "pheromone_coordination_mw" in stabilization_signal
        assert abs(stabilization_signal["pheromone_coordination_mw"]) > 0.0

    def test_local_stabilization_confidence_calculation(self):
        """Test confidence calculation for stabilization signal."""
        # Test high confidence scenario (clear grid conditions)
        self.agent.update_grid_conditions(
            frequency_hz=60.05,  # Clear frequency deviation
            voltage_kv=0.820,  # Clear voltage deviation
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()
        high_confidence = stabilization_signal["confidence"]

        # Test low confidence scenario (mixed signals)
        self.agent.update_grid_conditions(
            frequency_hz=60.01,  # Small frequency deviation
            voltage_kv=0.801,  # Small voltage deviation
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()
        low_confidence = stabilization_signal["confidence"]

        # High confidence should be greater than low confidence
        assert high_confidence > low_confidence

    def test_local_stabilization_priority_calculation(self):
        """Test priority calculation for stabilization actions."""
        # Test high priority scenario (emergency conditions)
        self.agent.update_grid_conditions(
            frequency_hz=60.2,  # High frequency deviation
            voltage_kv=0.900,  # High voltage deviation
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()
        high_priority = stabilization_signal["priority"]

        # Test low priority scenario (normal conditions)
        self.agent.update_grid_conditions(
            frequency_hz=60.01,  # Small frequency deviation
            voltage_kv=0.805,  # Small voltage deviation
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()
        low_priority = stabilization_signal["priority"]

        # High priority should be greater than low priority
        assert high_priority > low_priority

    def test_local_stabilization_soc_constraints(self):
        """Test SoC constraints on stabilization signal."""
        # Test at low SoC (limited discharge capability)
        self.battery.current_soc_percent = 10.0

        self.agent.update_grid_conditions(
            frequency_hz=59.9,  # Low frequency (needs discharge)
            voltage_kv=0.800,
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()

        # Should limit discharge due to low SoC
        assert stabilization_signal["power_mw"] > -self.battery.capacity_mw * 0.5

        # Test at high SoC (limited charge capability)
        self.battery.current_soc_percent = 95.0

        self.agent.update_grid_conditions(
            frequency_hz=60.1,  # High frequency (needs charge)
            voltage_kv=0.800,
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()

        # Should limit charge due to high SoC
        assert stabilization_signal["power_mw"] < self.battery.capacity_mw * 0.5

    def test_local_stabilization_response_time(self):
        """Test response time calculation for stabilization actions."""
        # Test emergency response time
        self.agent.update_grid_conditions(
            frequency_hz=60.3,  # Emergency frequency
            voltage_kv=0.900,  # Emergency voltage
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()
        emergency_response_time = stabilization_signal["response_time_s"]

        # Test normal response time
        self.agent.update_grid_conditions(
            frequency_hz=60.05,  # Normal frequency
            voltage_kv=0.820,  # Normal voltage
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()
        normal_response_time = stabilization_signal["response_time_s"]

        # Emergency response should be faster
        assert emergency_response_time < normal_response_time

    def test_local_stabilization_signal_integration(self):
        """Test integration of all stabilization signal components."""
        # Set up complex scenario
        self.agent.update_grid_conditions(
            frequency_hz=60.05,  # High frequency
            voltage_kv=0.820,  # High voltage
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        # Add pheromone influence
        self.agent.update_pheromone_gradients(
            {
                PheromoneType.FREQUENCY_SUPPORT: 0.5,
                PheromoneType.COORDINATION: 0.3,
            }
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()

        # Total power should be combination of all components
        total_power = stabilization_signal["power_mw"]
        frequency_component = stabilization_signal["frequency_support_mw"]
        voltage_component = stabilization_signal["voltage_support_mw"]
        pheromone_component = stabilization_signal["pheromone_coordination_mw"]

        # Components should contribute to total
        assert abs(total_power) > 0.0
        assert abs(frequency_component) > 0.0
        assert abs(voltage_component) > 0.0
        assert abs(pheromone_component) > 0.0

    def test_local_stabilization_execution(self):
        """Test execution of local stabilization actions."""
        # Calculate stabilization signal
        self.agent.update_grid_conditions(
            frequency_hz=60.05,
            voltage_kv=0.820,
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        stabilization_signal = self.agent.calculate_local_stabilization_signal()

        # Execute stabilization action
        self.agent.execute_local_stabilization(stabilization_signal)

        # Verify battery setpoint was updated
        expected_power = stabilization_signal["power_mw"]
        actual_power = self.battery.power_setpoint_mw

        assert abs(actual_power - expected_power) < 0.1

        # Verify pheromone strength was updated
        assert self.agent.pheromone_strength > 0.0


class TestBatteryAgentEnhancedIntegration:
    """Test integration of enhanced BatteryAgent functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.swarm_bus = SwarmBus(grid_width=10, grid_height=10)
        self.battery = Battery(
            asset_id="test_battery",
            name="Test Battery",
            node_id="test_node",
            capacity_mw=10.0,
            energy_capacity_mwh=40.0,
            initial_soc_percent=50.0,
        )
        self.battery.set_status(AssetStatus.ONLINE)
        self.agent = BatteryAgent(battery=self.battery)

    def test_enhanced_agent_registration(self):
        """Test enhanced agent registration with SwarmBus."""
        position = GridPosition(x=5, y=5)
        result = self.swarm_bus.register_agent(self.agent, position)

        assert result is True
        assert self.agent.agent_id in self.swarm_bus.registered_agents

    def test_enhanced_swarm_coordination(self):
        """Test enhanced swarm coordination with voltage triggers and pheromones."""
        # Register agent
        position = GridPosition(x=5, y=5)
        self.swarm_bus.register_agent(self.agent, position)

        # Set up grid conditions
        self.agent.update_grid_conditions(
            frequency_hz=60.05,
            voltage_kv=0.820,
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        # Perform coordination
        coordination_results = self.swarm_bus.coordinate_agents(update_pheromones=True)

        assert self.agent.agent_id in coordination_results
        assert len(coordination_results[self.agent.agent_id]) >= 0

    def test_enhanced_multi_agent_scenario(self):
        """Test enhanced functionality with multiple agents."""
        # Create multiple agents
        agents = []
        for i in range(3):
            battery = Battery(
                asset_id=f"battery_{i}",
                name=f"Battery {i}",
                node_id=f"node_{i}",
                capacity_mw=10.0,
                energy_capacity_mwh=40.0,
                initial_soc_percent=50.0,
            )
            battery.set_status(AssetStatus.ONLINE)
            agent = BatteryAgent(battery=battery)
            agents.append(agent)

            # Register with SwarmBus
            position = GridPosition(x=i, y=i)
            self.swarm_bus.register_agent(agent, position)

        # Update all agents with similar conditions
        for agent in agents:
            agent.update_grid_conditions(
                frequency_hz=60.05,
                voltage_kv=0.820,
                local_load_mw=100.0,
                local_generation_mw=90.0,
            )

        # Coordinate agents
        coordination_results = self.swarm_bus.coordinate_agents(update_pheromones=True)

        # All agents should coordinate
        assert len(coordination_results) == 3

        # Test local stabilization for all agents
        for agent in agents:
            stabilization_signal = agent.calculate_local_stabilization_signal()
            assert isinstance(stabilization_signal, dict)
            assert "power_mw" in stabilization_signal

    def test_enhanced_system_performance(self):
        """Test enhanced system performance under load."""
        # Register agent
        position = GridPosition(x=5, y=5)
        self.swarm_bus.register_agent(self.agent, position)

        # Simulate extended operation
        for step in range(10):
            # Update conditions
            frequency = 60.0 + (step - 5) * 0.01  # Vary frequency
            voltage = 0.8 + (step - 5) * 0.005  # Vary voltage

            self.agent.update_grid_conditions(
                frequency_hz=frequency,
                voltage_kv=voltage,
                local_load_mw=100.0,
                local_generation_mw=90.0,
            )

            # Calculate stabilization signal
            stabilization_signal = self.agent.calculate_local_stabilization_signal()

            # Execute action
            self.agent.execute_local_stabilization(stabilization_signal)

            # Verify system stability
            assert abs(self.battery.power_setpoint_mw) <= self.battery.capacity_mw
            assert 0.0 <= self.battery.current_soc_percent <= 100.0
            assert self.agent.pheromone_strength >= 0.0

    def test_enhanced_error_handling(self):
        """Test enhanced error handling and edge cases."""
        # Test with invalid grid conditions
        with pytest.raises(ValueError):
            self.agent.update_grid_conditions(
                frequency_hz=-60.0,  # Invalid frequency
                voltage_kv=0.8,
                local_load_mw=100.0,
                local_generation_mw=90.0,
            )

        # Test with extreme conditions
        self.agent.update_grid_conditions(
            frequency_hz=65.0,  # Extreme frequency
            voltage_kv=1.0,  # Extreme voltage
            local_load_mw=1000.0,
            local_generation_mw=10.0,
        )

        # Should handle gracefully
        stabilization_signal = self.agent.calculate_local_stabilization_signal()
        assert isinstance(stabilization_signal, dict)

    def test_enhanced_state_management(self):
        """Test enhanced state management and reset functionality."""
        # Set up enhanced state
        self.agent.update_grid_conditions(
            frequency_hz=60.05,
            voltage_kv=0.820,
            local_load_mw=100.0,
            local_generation_mw=90.0,
        )

        self.agent.update_pheromone_gradients(
            {
                PheromoneType.FREQUENCY_SUPPORT: 0.5,
            }
        )

        # Get enhanced state
        state = self.agent.get_enhanced_state()

        # Verify enhanced state components
        assert "voltage_deviation_v" in state
        assert "pheromone_gradients" in state
        assert "stabilization_priority" in state
        assert "local_grid_conditions" in state

        # Test reset functionality
        self.agent.reset()

        # Verify reset
        assert self.agent.local_grid_stress == 0.0
        assert self.agent.pheromone_strength == 0.0
        assert len(self.agent.pheromone_memory) == 0
