"""Tests for rule-based controller implementation."""

from unittest.mock import Mock

from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.assets.wind import WindTurbine
from psireg.sim.engine import GridEngine, GridState
from psireg.utils.enums import AssetType


class TestRuleBasedController:
    """Test rule-based controller implementation."""

    def test_rule_controller_creation(self):
        """Test that rule-based controller can be created."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        assert controller is not None

    def test_rule_controller_initialization(self):
        """Test rule-based controller initialization."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)
        grid_engine.assets = {
            "battery_1": Mock(spec=Battery),
            "load_1": Mock(spec=Load),
            "solar_1": Mock(spec=SolarPanel),
            "wind_1": Mock(spec=WindTurbine),
        }

        result = controller.initialize(grid_engine)
        assert result is True
        assert controller.grid_engine == grid_engine

    def test_rule_controller_frequency_regulation(self):
        """Test frequency regulation rules."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)

        # Mock battery asset
        battery = Mock(spec=Battery)
        battery.asset_id = "battery_1"
        battery.asset_type = AssetType.BATTERY
        battery.capacity_mw = 50.0
        battery.current_soc_percent = 50.0
        battery.get_max_charge_power.return_value = 50.0
        battery.get_max_discharge_power.return_value = 50.0

        grid_engine.assets = {"battery_1": battery}
        controller.initialize(grid_engine)

        # Test low frequency (need discharge)
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 59.8  # Low frequency
        grid_state.total_generation_mw = 90.0
        grid_state.total_load_mw = 100.0
        grid_state.power_balance_mw = -10.0

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        assert "battery_1" in actions
        assert actions["battery_1"]["power_setpoint_mw"] < 0  # Discharge

    def test_rule_controller_high_frequency_regulation(self):
        """Test high frequency regulation rules."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)

        # Mock battery asset
        battery = Mock(spec=Battery)
        battery.asset_id = "battery_1"
        battery.asset_type = AssetType.BATTERY
        battery.capacity_mw = 50.0
        battery.current_soc_percent = 50.0
        battery.get_max_charge_power.return_value = 50.0
        battery.get_max_discharge_power.return_value = 50.0

        grid_engine.assets = {"battery_1": battery}
        controller.initialize(grid_engine)

        # Test high frequency (need charge)
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 60.2  # High frequency
        grid_state.total_generation_mw = 110.0
        grid_state.total_load_mw = 100.0
        grid_state.power_balance_mw = 10.0

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        assert "battery_1" in actions
        assert actions["battery_1"]["power_setpoint_mw"] > 0  # Charge

    def test_rule_controller_demand_response(self):
        """Test demand response rules."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)

        # Mock load asset
        load = Mock(spec=Load)
        load.asset_id = "load_1"
        load.asset_type = AssetType.LOAD
        load.baseline_demand_mw = 75.0
        load.dr_capability_mw = 20.0
        load.current_demand_mw = 75.0

        grid_engine.assets = {"load_1": load}
        controller.initialize(grid_engine)

        # Test grid stress (need demand reduction)
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 59.7  # Low frequency indicates stress
        grid_state.total_generation_mw = 90.0
        grid_state.total_load_mw = 100.0
        grid_state.power_balance_mw = -10.0

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        assert "load_1" in actions
        assert actions["load_1"]["dr_signal_mw"] < 0  # Reduce demand

    def test_rule_controller_renewable_curtailment(self):
        """Test renewable curtailment rules."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)

        # Mock solar asset
        solar = Mock(spec=SolarPanel)
        solar.asset_id = "solar_1"
        solar.asset_type = AssetType.SOLAR
        solar.current_output_mw = 80.0
        solar.capacity_mw = 100.0
        solar.curtailment_factor = 0.0

        grid_engine.assets = {"solar_1": solar}
        controller.initialize(grid_engine)

        # Test over-generation (need curtailment)
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 60.3  # High frequency indicates over-generation
        grid_state.total_generation_mw = 120.0
        grid_state.total_load_mw = 100.0
        grid_state.power_balance_mw = 20.0

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        assert "solar_1" in actions
        assert actions["solar_1"]["curtailment_factor"] > 0  # Apply curtailment

    def test_rule_controller_soc_management(self):
        """Test battery SOC management rules."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)

        # Mock battery with low SOC
        battery = Mock(spec=Battery)
        battery.asset_id = "battery_1"
        battery.asset_type = AssetType.BATTERY
        battery.capacity_mw = 50.0
        battery.current_soc_percent = 15.0  # Low SOC
        battery.get_max_charge_power.return_value = 50.0
        battery.get_max_discharge_power.return_value = 50.0

        grid_engine.assets = {"battery_1": battery}
        controller.initialize(grid_engine)

        # Test with normal grid conditions but low SOC
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 60.0  # Normal frequency
        grid_state.total_generation_mw = 100.0
        grid_state.total_load_mw = 100.0
        grid_state.power_balance_mw = 0.0

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        # Should prioritize charging when SOC is low
        assert "battery_1" in actions
        assert actions["battery_1"]["power_setpoint_mw"] > 0  # Charge

    def test_rule_controller_priority_handling(self):
        """Test rule priority handling for conflicting objectives."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)

        # Mock battery with high SOC and low frequency
        battery = Mock(spec=Battery)
        battery.asset_id = "battery_1"
        battery.asset_type = AssetType.BATTERY
        battery.capacity_mw = 50.0
        battery.current_soc_percent = 95.0  # High SOC (want to discharge)
        battery.get_max_charge_power.return_value = 50.0
        battery.get_max_discharge_power.return_value = 50.0

        grid_engine.assets = {"battery_1": battery}
        controller.initialize(grid_engine)

        # Test with low frequency (need discharge) and high SOC (also want discharge)
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 59.8  # Low frequency (priority: discharge)
        grid_state.total_generation_mw = 90.0
        grid_state.total_load_mw = 100.0
        grid_state.power_balance_mw = -10.0

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        # Frequency regulation should take priority
        assert "battery_1" in actions
        assert actions["battery_1"]["power_setpoint_mw"] < 0  # Discharge

    def test_rule_controller_performance_metrics(self):
        """Test rule controller performance metrics calculation."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)
        grid_engine.assets = {}

        controller.initialize(grid_engine)

        # Simulate some control actions
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 60.1
        grid_state.total_generation_mw = 100.0
        grid_state.total_load_mw = 95.0
        grid_state.power_balance_mw = 5.0

        controller.update(grid_state, 1.0)
        metrics = controller.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert "frequency_deviation_hz" in metrics
        assert "power_balance_mw" in metrics
        assert "control_actions_count" in metrics
        assert "response_time_s" in metrics

    def test_rule_controller_reset(self):
        """Test rule controller reset functionality."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)
        grid_engine.assets = {}

        controller.initialize(grid_engine)

        # Simulate some operation
        grid_state = Mock(spec=GridState)
        controller.update(grid_state, 1.0)

        # Reset and verify state is cleared
        controller.reset()
        metrics = controller.get_performance_metrics()

        # Should reset internal counters and metrics
        assert metrics["control_actions_count"] == 0

    def test_rule_controller_edge_cases(self):
        """Test rule controller edge cases."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)
        grid_engine.assets = {}

        controller.initialize(grid_engine)

        # Test with no controllable assets
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 59.5  # Significant deviation

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        # Should handle gracefully with no assets
        assert isinstance(actions, dict)
        assert len(actions) == 0

    def test_rule_controller_multiple_assets(self):
        """Test rule controller with multiple assets of same type."""
        from psireg.controllers.rule import RuleBasedController

        controller = RuleBasedController()
        grid_engine = Mock(spec=GridEngine)

        # Multiple batteries
        battery1 = Mock(spec=Battery)
        battery1.asset_id = "battery_1"
        battery1.asset_type = AssetType.BATTERY
        battery1.capacity_mw = 50.0
        battery1.current_soc_percent = 30.0
        battery1.get_max_charge_power.return_value = 50.0
        battery1.get_max_discharge_power.return_value = 50.0

        battery2 = Mock(spec=Battery)
        battery2.asset_id = "battery_2"
        battery2.asset_type = AssetType.BATTERY
        battery2.capacity_mw = 30.0
        battery2.current_soc_percent = 70.0
        battery2.get_max_charge_power.return_value = 30.0
        battery2.get_max_discharge_power.return_value = 30.0

        grid_engine.assets = {"battery_1": battery1, "battery_2": battery2}
        controller.initialize(grid_engine)

        # Test frequency regulation with multiple batteries
        grid_state = Mock(spec=GridState)
        grid_state.frequency_hz = 59.8  # Low frequency
        grid_state.total_generation_mw = 90.0
        grid_state.total_load_mw = 100.0
        grid_state.power_balance_mw = -10.0

        controller.update(grid_state, 1.0)
        actions = controller.get_control_actions()

        # Both batteries should receive discharge commands
        assert "battery_1" in actions
        assert "battery_2" in actions
        assert actions["battery_1"]["power_setpoint_mw"] < 0
        assert actions["battery_2"]["power_setpoint_mw"] < 0


class TestRuleControllerLogic:
    """Test specific rule controller logic components."""

    def test_frequency_droop_calculation(self):
        """Test frequency droop calculation logic."""
        # This will test specific droop calculations once implemented
        pass

    def test_demand_response_scheduling(self):
        """Test demand response scheduling logic."""
        # This will test DR scheduling once implemented
        pass

    def test_renewable_curtailment_strategy(self):
        """Test renewable curtailment strategy."""
        # This will test curtailment strategy once implemented
        pass

    def test_battery_soc_limits(self):
        """Test battery SOC limit handling."""
        # This will test SOC limits once implemented
        pass


class TestRuleControllerIntegration:
    """Test rule controller integration scenarios."""

    def test_rule_controller_storm_scenario(self):
        """Test rule controller during storm scenario."""
        # This will test storm scenario integration once implemented
        pass

    def test_rule_controller_peak_demand(self):
        """Test rule controller during peak demand."""
        # This will test peak demand handling once implemented
        pass

    def test_rule_controller_renewable_surge(self):
        """Test rule controller during renewable surge."""
        # This will test renewable surge handling once implemented
        pass
