"""Tests for CLI main module.

This module tests the main CLI interface functionality including:
- Command-line argument parsing
- Scenario orchestrator integration
- Configuration management
- Error handling and validation
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestCLIInterface:
    """Test CLI interface functionality."""

    def test_cli_module_imports(self):
        """Test that CLI module can be imported."""
        from psireg.cli.main import app, create_cli_app

        assert app is not None
        assert create_cli_app is not None

    def test_cli_app_creation(self):
        """Test CLI app creation."""
        from psireg.cli.main import create_cli_app

        app = create_cli_app()
        assert app is not None
        assert hasattr(app, "command")

    def test_simulate_command_registration(self):
        """Test that simulate command is registered."""
        from psireg.cli.main import app

        # Get registered commands
        commands = app.registered_commands
        command_names = [cmd.name or cmd.callback.__name__ for cmd in commands]
        assert "simulate" in command_names

    def test_simulate_command_with_scenario(self):
        """Test simulate command with scenario argument."""
        from psireg.cli.main import simulate_command

        # Mock the scenario orchestrator
        with patch("psireg.cli.main.ScenarioOrchestrator") as mock_orchestrator:
            mock_instance = MagicMock()
            mock_orchestrator.return_value = mock_instance
            mock_instance.run_scenario.return_value = {"status": "success"}

            # Test command execution
            result = simulate_command("storm_day")
            assert result is not None
            mock_orchestrator.assert_called_once()
            mock_instance.run_scenario.assert_called_once_with(
                scenario_name="storm_day",
                duration_hours=None,
                output_dir="output",
                output_format="json",
                enable_metrics=True,
                verbose=False,
            )

    def test_simulate_command_with_invalid_scenario(self):
        """Test simulate command with invalid scenario."""
        from psireg.cli.main import simulate_command

        with patch("psireg.cli.main.ScenarioOrchestrator") as mock_orchestrator:
            mock_instance = MagicMock()
            mock_orchestrator.return_value = mock_instance
            mock_instance.run_scenario.side_effect = ValueError("Invalid scenario")

            result = simulate_command("invalid_scenario")
            assert result["status"] == "error"
            assert "Invalid scenario" in result["error_message"]

    def test_simulate_command_with_config_file(self):
        """Test simulate command with configuration file."""
        from psireg.cli.main import simulate_command

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
version: "0.1.0"
environment: "testing"
simulation:
  timestep_minutes: 15
  horizon_hours: 24
"""
            )
            config_file = f.name

        try:
            with patch("psireg.cli.main.ScenarioOrchestrator") as mock_orchestrator:
                mock_instance = MagicMock()
                mock_orchestrator.return_value = mock_instance
                mock_instance.run_scenario.return_value = {"status": "success"}

                result = simulate_command("storm_day", config_file=config_file)
                assert result is not None
                mock_orchestrator.assert_called_once()

        finally:
            os.unlink(config_file)

    def test_simulate_command_with_output_options(self):
        """Test simulate command with output options."""
        from psireg.cli.main import simulate_command

        with patch("psireg.cli.main.ScenarioOrchestrator") as mock_orchestrator:
            mock_instance = MagicMock()
            mock_orchestrator.return_value = mock_instance
            mock_instance.run_scenario.return_value = {"status": "success"}

            result = simulate_command("storm_day", output_dir="test_output", output_format="json", verbose=True)
            assert result is not None
            mock_orchestrator.assert_called_once()

    def test_cli_help_system(self):
        """Test CLI help system."""
        from psireg.cli.main import app

        # Test that help is available
        assert hasattr(app, "info")
        assert app.info.name == "psi"
        assert "PSIREG" in app.info.help

    def test_cli_version_command(self):
        """Test CLI version command."""
        from psireg.cli.main import version_command

        version = version_command()
        assert version is not None
        assert isinstance(version, str)

    def test_cli_list_scenarios_command(self):
        """Test CLI list scenarios command."""
        from psireg.cli.main import list_scenarios_command

        with patch("psireg.cli.main.ScenarioOrchestrator") as mock_orchestrator:
            mock_instance = MagicMock()
            mock_orchestrator.return_value = mock_instance
            mock_instance.list_scenarios.return_value = ["storm_day", "peak_demand", "normal"]

            scenarios = list_scenarios_command()
            assert scenarios is not None
            assert isinstance(scenarios, list)
            assert "storm_day" in scenarios


class TestCLIConfiguration:
    """Test CLI configuration handling."""

    def test_config_file_loading(self):
        """Test configuration file loading."""
        from psireg.cli.main import load_cli_config

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
version: "0.1.0"
environment: "testing"
simulation:
  timestep_minutes: 15
  horizon_hours: 24
grid:
  frequency_hz: 60.0
  voltage_kv: 230.0
"""
            )
            config_file = f.name

        try:
            config = load_cli_config(config_file)
            assert config is not None
            assert config.version == "0.1.0"
            assert config.environment == "testing"
            assert config.simulation.timestep_minutes == 15
            assert config.grid.frequency_hz == 60.0
        finally:
            os.unlink(config_file)

    def test_config_validation(self):
        """Test configuration validation."""
        from psireg.cli.main import load_cli_config

        # Create invalid config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
version: "0.1.0"
environment: "testing"
simulation:
  timestep_minutes: -15  # Invalid negative value
"""
            )
            config_file = f.name

        try:
            with pytest.raises(ValueError):  # Should raise validation error
                load_cli_config(config_file)
        finally:
            os.unlink(config_file)

    def test_default_config_creation(self):
        """Test default configuration creation."""
        from psireg.cli.main import create_default_config

        config = create_default_config()
        assert config is not None
        assert config.version == "0.1.0"
        assert config.simulation.timestep_minutes == 15
        assert config.grid.frequency_hz == 60.0


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_missing_scenario_argument(self):
        """Test handling of missing scenario argument."""
        from psireg.cli.main import simulate_command

        with pytest.raises(ValueError):
            simulate_command("")  # Empty scenario

    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        from psireg.cli.main import simulate_command

        with pytest.raises(FileNotFoundError):
            simulate_command("storm_day", config_file="non_existent_file.yaml")

    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        from psireg.cli.main import simulate_command

        with patch("psireg.cli.main.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")

            with pytest.raises(PermissionError):
                simulate_command("storm_day", output_dir="/tmp/forbidden")

    def test_keyboard_interrupt_handling(self):
        """Test handling of keyboard interrupt."""
        from psireg.cli.main import simulate_command

        with patch("psireg.cli.main.ScenarioOrchestrator") as mock_orchestrator:
            mock_instance = MagicMock()
            mock_orchestrator.return_value = mock_instance
            mock_instance.run_scenario.side_effect = KeyboardInterrupt()

            with pytest.raises(KeyboardInterrupt):
                simulate_command("storm_day")


class TestCLIIntegration:
    """Test CLI integration with other components."""

    def test_metrics_collection_integration(self):
        """Test integration with metrics collection."""
        from psireg.cli.main import simulate_command

        with patch("psireg.cli.main.ScenarioOrchestrator") as mock_orchestrator:
            mock_instance = MagicMock()
            mock_orchestrator.return_value = mock_instance
            mock_instance.run_scenario.return_value = {
                "status": "success",
                "metrics": {"total_steps": 100, "completion_time": 3600},
            }

            result = simulate_command("storm_day", enable_metrics=True)
            assert result is not None
            assert "metrics" in result

    def test_logging_integration(self):
        """Test integration with logging system."""
        from psireg.cli.main import simulate_command

        with patch("psireg.cli.main.ScenarioOrchestrator") as mock_orchestrator:
            mock_instance = MagicMock()
            mock_orchestrator.return_value = mock_instance
            mock_instance.run_scenario.return_value = {"status": "success"}

            # Test with verbose logging
            result = simulate_command("storm_day", verbose=True)
            assert result is not None

    def test_output_format_integration(self):
        """Test integration with different output formats."""
        from psireg.cli.main import simulate_command

        with patch("psireg.cli.main.ScenarioOrchestrator") as mock_orchestrator:
            mock_instance = MagicMock()
            mock_orchestrator.return_value = mock_instance
            mock_instance.run_scenario.return_value = {"status": "success"}

            # Test JSON output
            result = simulate_command("storm_day", output_format="json")
            assert result is not None

            # Test CSV output
            result = simulate_command("storm_day", output_format="csv")
            assert result is not None
