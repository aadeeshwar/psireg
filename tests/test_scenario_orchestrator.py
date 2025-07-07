"""Tests for Scenario Orchestrator system.

This module tests the scenario orchestrator functionality including:
- Scenario definition and loading
- Simulation execution
- Weather condition integration
- Asset and agent coordination
- Results collection and reporting
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestScenarioOrchestrator:
    """Test Scenario Orchestrator functionality."""

    def test_scenario_orchestrator_creation(self):
        """Test scenario orchestrator creation."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()
        assert orchestrator is not None
        assert hasattr(orchestrator, "run_scenario")
        assert hasattr(orchestrator, "list_scenarios")

    def test_scenario_orchestrator_with_config(self):
        """Test scenario orchestrator with configuration."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        config = {
            "simulation": {"timestep_minutes": 15, "horizon_hours": 24},
            "grid": {"frequency_hz": 60.0, "voltage_kv": 230.0},
        }

        orchestrator = ScenarioOrchestrator(config=config)
        assert orchestrator is not None
        assert orchestrator.config is not None

    def test_list_available_scenarios(self):
        """Test listing available scenarios."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()
        scenarios = orchestrator.list_scenarios()

        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        assert "storm_day" in scenarios
        assert "peak_demand" in scenarios
        assert "normal" in scenarios

    def test_get_scenario_info(self):
        """Test getting scenario information."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Test storm_day scenario info
        info = orchestrator.get_scenario_info("storm_day")
        assert info is not None
        assert "name" in info
        assert "description" in info
        assert "duration_hours" in info
        assert "weather_conditions" in info
        assert info["name"] == "storm_day"

    def test_invalid_scenario_handling(self):
        """Test handling of invalid scenarios."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        with pytest.raises(ValueError, match="Unknown scenario"):
            orchestrator.get_scenario_info("invalid_scenario")

    def test_scenario_configuration_loading(self):
        """Test loading scenario configuration."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Test loading storm_day scenario
        config = orchestrator._load_scenario_config("storm_day")
        assert config is not None
        assert "weather_conditions" in config
        assert "grid_events" in config
        assert "assets" in config


class TestScenarioExecution:
    """Test scenario execution functionality."""

    def test_run_scenario_basic(self):
        """Test basic scenario execution."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Mock the simulation execution to avoid complex grid engine setup
        with patch.object(orchestrator, "_execute_simulation") as mock_exec:
            with patch.object(orchestrator, "_setup_grid_engine"):
                with patch.object(orchestrator, "_setup_metrics_collector"):
                    with patch.object(orchestrator, "_setup_swarm_agents"):
                        mock_exec.return_value = {"simulation_data": "test"}

                        result = orchestrator.run_scenario("storm_day")
                        assert result is not None
                        assert "status" in result
                        assert result["status"] == "success"
                        assert "execution_time_seconds" in result

    def test_run_scenario_with_duration(self):
        """Test scenario execution with custom duration."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Mock the simulation execution to avoid complex grid engine setup
        with patch.object(orchestrator, "_execute_simulation") as mock_exec:
            with patch.object(orchestrator, "_setup_grid_engine"):
                with patch.object(orchestrator, "_setup_metrics_collector"):
                    with patch.object(orchestrator, "_setup_swarm_agents"):
                        mock_exec.return_value = {"simulation_data": "test"}

                        result = orchestrator.run_scenario("storm_day", duration_hours=48)
                        assert result is not None
                        assert result["status"] == "success"
                        assert result["duration_hours"] == 48

    def test_run_scenario_with_output_dir(self):
        """Test scenario execution with output directory."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the simulation execution and output generation
            with patch.object(orchestrator, "_execute_simulation") as mock_exec:
                with patch.object(orchestrator, "_setup_grid_engine"):
                    with patch.object(orchestrator, "_setup_metrics_collector"):
                        with patch.object(orchestrator, "_setup_swarm_agents"):
                            with patch.object(orchestrator, "_generate_output_files") as mock_output:
                                mock_exec.return_value = {"simulation_data": "test"}
                                mock_output.return_value = {"results.json": "/path/to/results.json"}

                                result = orchestrator.run_scenario("storm_day", output_dir=temp_dir)
                                assert result is not None
                                assert result["status"] == "success"
                                assert "output_files" in result

    def test_run_scenario_with_metrics(self):
        """Test scenario execution with metrics collection."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Mock the simulation execution and metrics collection
        with patch.object(orchestrator, "_execute_simulation") as mock_exec:
            with patch.object(orchestrator, "_setup_grid_engine"):
                with patch.object(orchestrator, "_setup_metrics_collector"):
                    with patch.object(orchestrator, "_setup_swarm_agents"):
                        with patch.object(orchestrator, "_collect_final_metrics") as mock_metrics:
                            mock_exec.return_value = {"simulation_data": "test"}
                            mock_metrics.return_value = {"performance": {"test": "data"}}

                            # Mock the metrics collector so it exists
                            orchestrator.metrics_collector = MagicMock()

                            result = orchestrator.run_scenario("storm_day", enable_metrics=True)
                            assert result is not None
                            assert result["status"] == "success"
                            assert "metrics" in result
                            assert "performance" in result["metrics"]

    def test_scenario_interruption_handling(self):
        """Test handling of scenario interruption."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Mock the simulation execution to raise KeyboardInterrupt
        with patch.object(orchestrator, "_execute_simulation") as mock_exec:
            with patch.object(orchestrator, "_setup_grid_engine"):
                with patch.object(orchestrator, "_setup_metrics_collector"):
                    with patch.object(orchestrator, "_setup_swarm_agents"):
                        mock_exec.side_effect = KeyboardInterrupt()

                        result = orchestrator.run_scenario("storm_day")
                        assert result is not None
                        assert result["status"] == "interrupted"

    def test_scenario_error_handling(self):
        """Test handling of scenario execution errors."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        with patch("psireg.cli.orchestrator.GridEngine") as mock_engine:
            mock_engine.side_effect = Exception("Grid engine error")

            result = orchestrator.run_scenario("storm_day")
            assert result is not None
            assert result["status"] == "error"
            assert "error_message" in result


class TestScenarioConfiguration:
    """Test scenario configuration functionality."""

    def test_scenario_config_validation(self):
        """Test scenario configuration validation."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Test valid config
        valid_config = {
            "name": "test_scenario",
            "description": "Test scenario",
            "duration_hours": 24,
            "weather_conditions": {"condition": "CLEAR"},
            "grid_events": [],
            "assets": [],
        }

        assert orchestrator._validate_scenario_config(valid_config) is True

    def test_scenario_config_invalid(self):
        """Test invalid scenario configuration."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Test invalid config (missing required fields)
        invalid_config = {
            "name": "test_scenario",
            # Missing required fields
        }

        assert orchestrator._validate_scenario_config(invalid_config) is False

    def test_scenario_config_weather_conditions(self):
        """Test scenario configuration with weather conditions."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        config = orchestrator._load_scenario_config("storm_day")

        assert "weather_conditions" in config
        weather = config["weather_conditions"]
        assert isinstance(weather, dict)
        assert "condition" in weather
        from psireg.utils.enums import WeatherCondition

        # The storm day scenario starts with CLOUDY conditions, not STORMY
        assert weather["condition"] in [WeatherCondition.CLOUDY, WeatherCondition.STORMY]

    def test_scenario_config_grid_events(self):
        """Test scenario configuration with grid events."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        config = orchestrator._load_scenario_config("storm_day")

        assert "grid_events" in config
        events = config["grid_events"]
        assert isinstance(events, list)
        assert len(events) > 0

    def test_scenario_config_assets(self):
        """Test scenario configuration with assets."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        config = orchestrator._load_scenario_config("storm_day")

        assert "assets" in config
        assets = config["assets"]
        assert isinstance(assets, dict)
        assert "renewable" in assets
        assert "storage" in assets
        assert "load" in assets


class TestScenarioIntegration:
    """Test scenario integration with other components."""

    def test_grid_engine_integration(self):
        """Test integration with GridEngine."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        with patch("psireg.cli.orchestrator.GridEngine") as mock_engine:
            mock_engine_instance = MagicMock()
            mock_engine.return_value = mock_engine_instance

            # Test grid engine setup
            orchestrator._setup_grid_engine("storm_day")
            mock_engine.assert_called_once()

    def test_metrics_collector_integration(self):
        """Test integration with metrics collector."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        with patch("psireg.cli.orchestrator.MetricsCollector") as mock_collector:
            mock_collector_instance = MagicMock()
            mock_collector.return_value = mock_collector_instance

            # Test metrics setup
            orchestrator._setup_metrics_collector()
            mock_collector.assert_called_once()

    def test_swarm_agents_integration(self):
        """Test integration with swarm agents."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        with patch("psireg.cli.orchestrator.SwarmBus") as mock_swarm:
            with patch.object(orchestrator, "_setup_grid_engine"):
                mock_swarm_instance = MagicMock()
                mock_swarm.return_value = mock_swarm_instance

                # Setup grid engine first so get_all_assets works
                orchestrator.grid_engine = MagicMock()
                orchestrator.grid_engine.get_all_assets.return_value = []

                # Test swarm setup
                orchestrator._setup_swarm_agents("storm_day")
                mock_swarm.assert_called_once()

    def test_weather_system_integration(self):
        """Test integration with weather system."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Setup grid engine first so get_all_assets works
        orchestrator.grid_engine = MagicMock()
        orchestrator.grid_engine.get_all_assets.return_value = []

        # Test weather condition application
        weather_config = {"condition": "STORMY", "wind_speed": 25.0}
        orchestrator._apply_weather_conditions(weather_config)

        # Should not raise any errors
        assert True

    def test_output_generation_integration(self):
        """Test integration with output generation."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test output generation
            results = {
                "status": "success",
                "metrics": {"total_steps": 100},
                "grid_data": {"frequency": [60.0, 59.9, 60.1]},
            }

            output_files = orchestrator._generate_output_files(results, temp_dir, "json")

            assert isinstance(output_files, dict)
            assert "summary" in output_files
            assert "metrics" in output_files


class TestScenarioValidation:
    """Test scenario validation functionality."""

    def test_scenario_execution_validation(self):
        """Test scenario execution validation."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Test pre-execution validation
        assert orchestrator._validate_scenario_execution("storm_day") is True

    def test_scenario_results_validation(self):
        """Test scenario results validation."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Test valid results
        valid_results = {"status": "success", "duration": 3600, "metrics": {"total_steps": 100}, "grid_data": []}

        assert orchestrator._validate_scenario_results(valid_results) is True

    def test_scenario_output_validation(self):
        """Test scenario output validation."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test output directory validation
            assert orchestrator._validate_output_directory(temp_dir) is True

            # Test invalid output directory
            assert orchestrator._validate_output_directory("/invalid/path") is False

    def test_scenario_configuration_integrity(self):
        """Test scenario configuration integrity."""
        from psireg.cli.orchestrator import ScenarioOrchestrator

        orchestrator = ScenarioOrchestrator()

        # Test all available scenarios have valid configurations
        scenarios = orchestrator.list_scenarios()

        for scenario in scenarios:
            config = orchestrator._load_scenario_config(scenario)
            assert orchestrator._validate_scenario_config(config) is True
