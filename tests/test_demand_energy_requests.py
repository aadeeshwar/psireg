"""Test Demand Node Energy Request Logic.

This module contains comprehensive tests for the pheromone-based energy request
system that allows demand nodes to proactively request energy and coordinate
responses through the swarm intelligence system.

Features tested:
- Energy request pheromone types and broadcasting
- Request generation logic based on anticipated shortfalls
- Request-response coordination between demand and supply agents
- Enhanced demand response incorporating request logic
"""

from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.swarm.agents.battery_agent import BatteryAgent
from psireg.swarm.agents.demand_agent import DemandAgent
from psireg.swarm.pheromone import GridPosition, PheromoneType, SwarmBus
from psireg.utils.enums import AssetStatus


class TestEnergyRequestPheromones:
    """Test energy request pheromone types and broadcasting functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.load = Load(
            asset_id="test_load",
            name="Test Load",
            node_id="test_node",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
            price_elasticity=-0.2,
        )
        self.load.set_status(AssetStatus.ONLINE)
        self.agent = DemandAgent(load=self.load)

    def test_energy_request_pheromone_types_exist(self):
        """Test that energy request pheromone types are available."""
        # Test that new energy request pheromone types exist
        assert hasattr(PheromoneType, "ENERGY_REQUEST_HIGH")
        assert hasattr(PheromoneType, "ENERGY_REQUEST_NORMAL")

        # Test that they have appropriate values
        assert PheromoneType.ENERGY_REQUEST_HIGH.value == "energy_request_high"
        assert PheromoneType.ENERGY_REQUEST_NORMAL.value == "energy_request_normal"

    def test_energy_request_broadcasting(self):
        """Test energy request broadcasting via pheromones."""
        # Test that agent can broadcast energy requests
        assert hasattr(self.agent, "broadcast_energy_request")

        # Test high priority energy request
        result = self.agent.broadcast_energy_request(
            energy_needed_mw=30.0, urgency="high", duration_hours=2, max_price_mwh=150.0
        )

        assert result is True
        assert hasattr(self.agent, "pending_energy_requests")
        assert len(self.agent.pending_energy_requests) == 1

        request = self.agent.pending_energy_requests[0]
        assert request["energy_needed_mw"] == 30.0
        assert request["urgency"] == "high"
        assert request["duration_hours"] == 2
        assert request["max_price_mwh"] == 150.0

    def test_energy_request_pheromone_strength_calculation(self):
        """Test energy request pheromone strength calculation."""
        # Test that agent can calculate appropriate pheromone strength for requests
        assert hasattr(self.agent, "calculate_request_pheromone_strength")

        # High urgency, large energy need should produce strong pheromone
        strength_high = self.agent.calculate_request_pheromone_strength(
            energy_needed_mw=50.0, urgency="high", grid_stress=0.8
        )

        # Normal urgency, smaller energy need should produce weaker pheromone
        strength_normal = self.agent.calculate_request_pheromone_strength(
            energy_needed_mw=20.0, urgency="normal", grid_stress=0.3
        )

        assert strength_high > strength_normal
        assert 0.0 <= strength_high <= 1.0
        assert 0.0 <= strength_normal <= 1.0

    def test_energy_request_deposition_to_swarm(self):
        """Test energy request deposition to swarm bus."""
        # Create swarm bus for testing
        swarm_bus = SwarmBus(grid_width=10, grid_height=10)
        position = GridPosition(x=5, y=5)

        # Register agent with swarm
        swarm_bus.register_agent(self.agent, position)

        # Test that agent can deposit energy request pheromones
        success = swarm_bus.deposit_pheromone(
            agent_id=self.agent.agent_id, pheromone_type=PheromoneType.ENERGY_REQUEST_HIGH, strength=0.8
        )

        assert success is True

        # Verify pheromone was deposited
        strength = swarm_bus.get_pheromone_at_agent(
            agent_id=self.agent.agent_id, pheromone_type=PheromoneType.ENERGY_REQUEST_HIGH
        )
        assert strength == 0.8


class TestEnergyRequestGeneration:
    """Test energy request generation logic based on grid conditions."""

    def setup_method(self):
        """Set up test environment."""
        self.load = Load(
            asset_id="test_load",
            name="Test Load",
            node_id="test_node",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
            min_demand_mw=40.0,
        )
        self.load.set_status(AssetStatus.ONLINE)
        self.agent = DemandAgent(load=self.load)

    def test_energy_request_triggering_conditions(self):
        """Test conditions that trigger energy requests."""
        # Test that agent can detect when energy requests are needed
        assert hasattr(self.agent, "should_request_energy")

        # High demand, low generation should trigger request
        self.agent.update_grid_conditions(
            frequency_hz=59.8,  # Low frequency indicates shortage
            voltage_kv=0.95,  # Low voltage
            local_load_mw=200.0,
            local_generation_mw=150.0,  # Shortage
            electricity_price=120.0,  # High price indicates scarcity
        )

        should_request = self.agent.should_request_energy(
            forecast_demand=[80.0, 85.0, 90.0],
            forecast_generation=[140.0, 130.0, 120.0],
            forecast_prices=[130.0, 140.0, 150.0],
        )

        assert should_request is True

    def test_energy_request_not_triggered_good_conditions(self):
        """Test that energy requests are not triggered in good grid conditions."""
        # Good grid conditions should not trigger requests
        self.agent.update_grid_conditions(
            frequency_hz=60.0,  # Normal frequency
            voltage_kv=1.0,  # Normal voltage
            local_load_mw=150.0,
            local_generation_mw=170.0,  # Surplus
            electricity_price=50.0,  # Low price
        )

        should_request = self.agent.should_request_energy(
            forecast_demand=[70.0, 75.0, 80.0],
            forecast_generation=[180.0, 175.0, 170.0],
            forecast_prices=[45.0, 50.0, 55.0],
        )

        assert should_request is False

    def test_energy_request_calculation(self):
        """Test energy request amount calculation."""
        # Test that agent can calculate how much energy to request
        assert hasattr(self.agent, "calculate_energy_request")

        self.agent.update_grid_conditions(
            frequency_hz=59.9, voltage_kv=0.98, local_load_mw=180.0, local_generation_mw=160.0, electricity_price=100.0
        )

        request = self.agent.calculate_energy_request(
            forecast_demand=[85.0, 90.0, 95.0],
            forecast_generation=[155.0, 150.0, 145.0],
            forecast_prices=[105.0, 110.0, 115.0],
        )

        assert "energy_needed_mw" in request
        assert "urgency" in request
        assert "duration_hours" in request
        assert "max_price_mwh" in request

        assert request["energy_needed_mw"] > 0.0
        assert request["urgency"] in ["high", "normal", "low"]
        assert request["duration_hours"] > 0
        assert request["max_price_mwh"] > 0.0

    def test_energy_request_urgency_determination(self):
        """Test energy request urgency determination."""
        # Critical shortage should produce high urgency
        self.agent.local_grid_stress = 0.9
        self.load.current_demand_mw = 95.0  # Near capacity

        request_critical = self.agent.calculate_energy_request(
            forecast_demand=[100.0, 105.0, 110.0],  # Exceeding capacity
            forecast_generation=[80.0, 75.0, 70.0],  # Declining
            forecast_prices=[200.0, 220.0, 250.0],  # Very high
        )

        # Moderate shortage should produce normal urgency
        self.agent.local_grid_stress = 0.4
        self.load.current_demand_mw = 80.0

        request_moderate = self.agent.calculate_energy_request(
            forecast_demand=[85.0, 87.0, 89.0],
            forecast_generation=[75.0, 73.0, 71.0],
            forecast_prices=[90.0, 95.0, 100.0],
        )

        assert request_critical["urgency"] == "high"
        assert request_moderate["urgency"] in ["normal", "low"]


class TestRequestResponseCoordination:
    """Test request-response coordination between demand and supply agents."""

    def setup_method(self):
        """Set up test environment with demand and battery agents."""
        # Create demand agent
        self.load = Load(
            asset_id="test_load",
            name="Test Load",
            node_id="load_node",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=20.0,
        )
        self.load.set_status(AssetStatus.ONLINE)
        self.demand_agent = DemandAgent(load=self.load)

        # Create battery agent
        self.battery = Battery(
            asset_id="test_battery",
            name="Test Battery",
            node_id="battery_node",
            capacity_mw=50.0,
            energy_capacity_mwh=200.0,
            initial_soc_percent=60.0,
        )
        self.battery.set_status(AssetStatus.ONLINE)
        self.battery_agent = BatteryAgent(battery=self.battery)

        # Create swarm bus
        self.swarm_bus = SwarmBus(grid_width=10, grid_height=10)
        self.demand_position = GridPosition(x=3, y=3)
        self.battery_position = GridPosition(x=5, y=5)

        self.swarm_bus.register_agent(self.demand_agent, self.demand_position)
        self.swarm_bus.register_agent(self.battery_agent, self.battery_position)

    def test_supply_agent_energy_request_detection(self):
        """Test that supply agents can detect energy requests."""
        # Test that battery agent can detect energy request pheromones
        assert hasattr(self.battery_agent, "detect_energy_requests")

        # Deposit energy request pheromone
        self.swarm_bus.deposit_pheromone(
            agent_id=self.demand_agent.agent_id, pheromone_type=PheromoneType.ENERGY_REQUEST_HIGH, strength=0.7
        )

        # Battery agent should detect the request
        detected_requests = self.battery_agent.detect_energy_requests(self.swarm_bus)

        assert len(detected_requests) > 0
        assert detected_requests[0]["pheromone_type"] == PheromoneType.ENERGY_REQUEST_HIGH
        assert detected_requests[0]["strength"] == 0.7

    def test_supply_agent_response_calculation(self):
        """Test supply agent energy request response calculation."""
        # Test that battery agent can calculate response to energy requests
        assert hasattr(self.battery_agent, "calculate_energy_response")

        # Create energy request scenario
        energy_requests = [
            {
                "agent_id": self.demand_agent.agent_id,
                "position": self.demand_position,
                "pheromone_type": PheromoneType.ENERGY_REQUEST_HIGH,
                "strength": 0.8,
                "energy_needed_mw": 25.0,
                "max_price_mwh": 120.0,
                "urgency": "high",
            }
        ]

        response = self.battery_agent.calculate_energy_response(energy_requests)

        assert "can_provide_mw" in response
        assert "response_priority" in response
        assert "estimated_cost_mwh" in response
        assert "response_duration_hours" in response

        assert response["can_provide_mw"] >= 0.0
        assert 0.0 <= response["response_priority"] <= 1.0

    def test_request_response_matching(self):
        """Test request-response matching algorithm."""
        # Test that system can match energy requests with responses
        assert hasattr(self.demand_agent, "process_energy_responses")

        # Create energy request
        self.demand_agent.broadcast_energy_request(
            energy_needed_mw=30.0, urgency="high", duration_hours=2, max_price_mwh=130.0
        )

        # Create energy response from battery
        energy_responses = [
            {
                "agent_id": self.battery_agent.agent_id,
                "can_provide_mw": 25.0,
                "response_priority": 0.8,
                "estimated_cost_mwh": 110.0,
                "response_duration_hours": 3.0,
            }
        ]

        # Process responses
        matches = self.demand_agent.process_energy_responses(energy_responses)

        assert len(matches) > 0
        assert matches[0]["supplier_agent_id"] == self.battery_agent.agent_id
        assert matches[0]["agreed_energy_mw"] <= 30.0  # Not more than requested
        assert matches[0]["agreed_price_mwh"] <= 130.0  # Not more than max price

    def test_coordination_protocol_execution(self):
        """Test coordination protocol execution for energy requests."""
        # Test complete coordination protocol
        assert hasattr(self.demand_agent, "execute_energy_coordination")

        # Setup scenario with energy shortage
        self.demand_agent.update_grid_conditions(
            frequency_hz=59.8, voltage_kv=0.96, local_load_mw=200.0, local_generation_mw=160.0, electricity_price=110.0
        )

        # Execute coordination protocol
        coordination_result = self.demand_agent.execute_energy_coordination(
            swarm_bus=self.swarm_bus,
            forecast_demand=[80.0, 85.0, 90.0],
            forecast_generation=[155.0, 150.0, 145.0],
            forecast_prices=[115.0, 120.0, 125.0],
        )

        assert "requests_sent" in coordination_result
        assert "responses_received" in coordination_result
        assert "energy_secured_mw" in coordination_result
        assert "demand_response_mw" in coordination_result  # Primary output

        # Primary output should be demand response
        assert coordination_result["demand_response_mw"] != 0.0


class TestEnhancedDemandResponse:
    """Test enhanced demand response incorporating energy request logic."""

    def setup_method(self):
        """Set up test environment."""
        self.load = Load(
            asset_id="test_load",
            name="Test Load",
            node_id="test_node",
            capacity_mw=100.0,
            baseline_demand_mw=75.0,
            dr_capability_mw=25.0,
        )
        self.load.set_status(AssetStatus.ONLINE)
        self.agent = DemandAgent(load=self.load)

    def test_enhanced_demand_response_integration(self):
        """Test integration of energy requests with demand response."""
        # Test that energy request logic is integrated with demand response
        assert hasattr(self.agent, "calculate_enhanced_demand_response")

        # Setup scenario
        self.agent.update_grid_conditions(
            frequency_hz=59.9, voltage_kv=0.97, local_load_mw=180.0, local_generation_mw=165.0, electricity_price=95.0
        )

        # Calculate enhanced demand response
        enhanced_response = self.agent.calculate_enhanced_demand_response(
            forecast_demand=[80.0, 85.0, 90.0],
            forecast_generation=[160.0, 155.0, 150.0],
            forecast_prices=[100.0, 105.0, 110.0],
            secured_energy_mw=15.0,  # Energy secured through requests
        )

        assert "demand_adjustment_mw" in enhanced_response
        assert "request_based_adjustment_mw" in enhanced_response
        assert "traditional_response_mw" in enhanced_response
        assert "total_response_mw" in enhanced_response
        assert "confidence" in enhanced_response

        # Total response should be sum of components
        total = enhanced_response["total_response_mw"]
        traditional = enhanced_response["traditional_response_mw"]
        request_based = enhanced_response["request_based_adjustment_mw"]

        assert abs(total - (traditional + request_based)) < 0.01

    def test_demand_response_with_energy_security(self):
        """Test demand response when energy has been secured through requests."""
        # When energy is secured, demand response should be less aggressive
        enhanced_response_with_security = self.agent.calculate_enhanced_demand_response(
            forecast_demand=[85.0, 90.0, 95.0],
            forecast_generation=[150.0, 145.0, 140.0],
            forecast_prices=[110.0, 115.0, 120.0],
            secured_energy_mw=20.0,  # Significant energy secured
        )

        enhanced_response_without_security = self.agent.calculate_enhanced_demand_response(
            forecast_demand=[85.0, 90.0, 95.0],
            forecast_generation=[150.0, 145.0, 140.0],
            forecast_prices=[110.0, 115.0, 120.0],
            secured_energy_mw=0.0,  # No energy secured
        )

        # Response should be less aggressive when energy is secured
        with_security_reduction = abs(enhanced_response_with_security["total_response_mw"])
        without_security_reduction = abs(enhanced_response_without_security["total_response_mw"])

        assert with_security_reduction < without_security_reduction

    def test_primary_output_demand_response(self):
        """Test that primary output is demand response incorporating request logic."""
        # Test the main method that produces the primary output
        assert hasattr(self.agent, "generate_primary_demand_response")

        # Setup conditions
        self.agent.update_grid_conditions(
            frequency_hz=59.85, voltage_kv=0.95, local_load_mw=190.0, local_generation_mw=170.0, electricity_price=105.0
        )

        # Generate primary demand response output
        primary_output = self.agent.generate_primary_demand_response(
            forecast_demand=[80.0, 85.0, 90.0, 95.0],
            forecast_generation=[165.0, 160.0, 155.0, 150.0],
            forecast_prices=[110.0, 115.0, 120.0, 125.0],
            swarm_bus=None,  # No swarm coordination for this test
        )

        # Verify primary output structure
        assert "demand_response_mw" in primary_output
        assert "energy_requests_sent" in primary_output
        assert "energy_secured_mw" in primary_output
        assert "coordination_signals" in primary_output
        assert "response_confidence" in primary_output
        assert "action_priority" in primary_output

        # Primary output should be demand response
        assert isinstance(primary_output["demand_response_mw"], int | float)
        assert primary_output["response_confidence"] >= 0.0
        assert primary_output["action_priority"] >= 0.0


class TestEnergyRequestSystemIntegration:
    """Test complete energy request system integration."""

    def setup_method(self):
        """Set up complex test environment with multiple agents."""
        # Create multiple demand agents
        self.demand_agents = []
        for i in range(3):
            load = Load(
                asset_id=f"load_{i}",
                name=f"Load {i}",
                node_id=f"load_node_{i}",
                capacity_mw=80.0 + i * 20,
                baseline_demand_mw=60.0 + i * 15,
                dr_capability_mw=15.0 + i * 5,
            )
            load.set_status(AssetStatus.ONLINE)
            agent = DemandAgent(load=load, agent_id=f"demand_agent_{i}")
            self.demand_agents.append(agent)

        # Create battery agents
        self.battery_agents = []
        for i in range(2):
            battery = Battery(
                asset_id=f"battery_{i}",
                name=f"Battery {i}",
                node_id=f"battery_node_{i}",
                capacity_mw=40.0 + i * 10,
                energy_capacity_mwh=160.0 + i * 40,
                initial_soc_percent=50.0 + i * 20,
            )
            battery.set_status(AssetStatus.ONLINE)
            agent = BatteryAgent(battery=battery, agent_id=f"battery_agent_{i}")
            self.battery_agents.append(agent)

        # Create swarm bus
        self.swarm_bus = SwarmBus(grid_width=15, grid_height=15)

        # Register all agents
        for i, agent in enumerate(self.demand_agents):
            position = GridPosition(x=2 + i * 3, y=5)
            self.swarm_bus.register_agent(agent, position)

        for i, agent in enumerate(self.battery_agents):
            position = GridPosition(x=6 + i * 3, y=8)
            self.swarm_bus.register_agent(agent, position)

    def test_multi_agent_energy_request_coordination(self):
        """Test energy request coordination with multiple agents."""
        # Setup grid stress scenario
        for agent in self.demand_agents:
            agent.update_grid_conditions(
                frequency_hz=59.8,
                voltage_kv=0.94,
                local_load_mw=250.0,
                local_generation_mw=210.0,
                electricity_price=125.0,
            )

        # Execute coordination for all demand agents
        results = []
        for agent in self.demand_agents:
            result = agent.generate_primary_demand_response(
                forecast_demand=[70.0, 75.0, 80.0, 85.0],
                forecast_generation=[200.0, 195.0, 190.0, 185.0],
                forecast_prices=[130.0, 135.0, 140.0, 145.0],
                swarm_bus=self.swarm_bus,
            )
            results.append(result)

        # Verify all agents produced primary outputs
        assert len(results) == 3
        for result in results:
            assert "demand_response_mw" in result
            assert isinstance(result["demand_response_mw"], int | float)

        # Verify coordination occurred (some agents should have sent requests)
        total_requests = sum(result["energy_requests_sent"] for result in results)
        assert total_requests > 0

    def test_system_pheromone_field_evolution(self):
        """Test pheromone field evolution during energy request coordination."""
        # Initial pheromone field should be empty
        # initial_totals = self.swarm_bus.get_system_stats()["pheromone_totals"]

        # Execute energy request coordination
        for agent in self.demand_agents:
            agent.update_grid_conditions(
                frequency_hz=59.75,
                voltage_kv=0.92,
                local_load_mw=280.0,
                local_generation_mw=230.0,
                electricity_price=140.0,
            )

            agent.generate_primary_demand_response(
                forecast_demand=[75.0, 80.0, 85.0, 90.0],
                forecast_generation=[225.0, 220.0, 215.0, 210.0],
                forecast_prices=[145.0, 150.0, 155.0, 160.0],
                swarm_bus=self.swarm_bus,
            )

        # Update swarm bus time step to allow pheromone diffusion
        self.swarm_bus.update_time_step()

        # Pheromone field should now contain energy request pheromones
        final_totals = self.swarm_bus.get_system_stats()["pheromone_totals"]

        # Check for energy request pheromones
        energy_request_pheromones = [
            final_totals.get("energy_request_high", 0.0),
            final_totals.get("energy_request_normal", 0.0),
        ]

        assert sum(energy_request_pheromones) > 0.0

    def test_demand_response_system_performance(self):
        """Test overall system performance with energy request logic."""
        # Create challenging scenario
        grid_conditions = {
            "frequency_hz": 59.7,
            "voltage_kv": 0.90,
            "local_load_mw": 320.0,
            "local_generation_mw": 260.0,
            "electricity_price": 180.0,
        }

        # Update all agents
        for agent in self.demand_agents + self.battery_agents:
            agent.update_grid_conditions(**grid_conditions)

        # Execute demand response with energy requests
        total_demand_response = 0.0
        total_energy_secured = 0.0

        for agent in self.demand_agents:
            result = agent.generate_primary_demand_response(
                forecast_demand=[80.0, 85.0, 90.0, 95.0],
                forecast_generation=[250.0, 240.0, 230.0, 220.0],
                forecast_prices=[185.0, 190.0, 195.0, 200.0],
                swarm_bus=self.swarm_bus,
            )

            total_demand_response += result["demand_response_mw"]
            total_energy_secured += result["energy_secured_mw"]

        # System should provide meaningful demand response
        assert abs(total_demand_response) > 0.0

        # System should attempt to secure energy through requests
        assert total_energy_secured >= 0.0

        # All agents should produce valid responses
        for agent in self.demand_agents:
            assert agent.get_coordination_signal() is not None
            assert agent.get_pheromone_strength() >= 0.0
