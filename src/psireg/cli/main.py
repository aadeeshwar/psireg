"""Main CLI interface for PSIREG renewable energy grid system.

This module provides the main command-line interface using Typer framework
for the Predictive Swarm Intelligence for Renewable Energy Grids system.

Usage:
    psi simulate --scenario storm_day
    psi list-scenarios
    psi version

Primary Output:
    One-command runs for scenario orchestration
"""

import os
import sys
from pathlib import Path
from typing import Any

try:
    import typer
    from rich.console import Console
    from rich.table import Table

    _TYPER_AVAILABLE = True
except ImportError:
    _TYPER_AVAILABLE = False
    typer = None
    Console = None
    Table = None

from psireg.config.loaders import YamlConfigLoader
from psireg.config.schema import PSIREGConfig
from psireg.utils.logger import logger

# Import CLI components
from .orchestrator import ScenarioOrchestrator

# Create console for rich output
console = Console() if Console else None

# Create main Typer app
if _TYPER_AVAILABLE:
    app = typer.Typer(
        name="psi",
        help="PSIREG - Predictive Swarm Intelligence for Renewable Energy Grids",
        add_completion=False,
        rich_markup_mode="rich",
    )
else:
    app = None


def create_cli_app() -> typer.Typer:
    """Create and configure the CLI application.

    Returns:
        Configured Typer application
    """
    if not _TYPER_AVAILABLE:
        raise ImportError("Typer is required for CLI functionality. Install with: pip install typer rich")

    return app


@app.command()
def simulate(
    scenario: str = typer.Argument(..., help="Scenario name to simulate (e.g., storm_day)"),
    config_file: str | None = typer.Option(None, "--config", "-c", help="Configuration file path"),
    output_dir: str | None = typer.Option("output", "--output", "-o", help="Output directory"),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format (json/csv)"),
    duration_hours: int | None = typer.Option(None, "--duration", "-d", help="Simulation duration in hours"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    enable_metrics: bool = typer.Option(True, "--metrics/--no-metrics", help="Enable metrics collection"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show configuration without running"),
) -> dict[str, Any]:
    """Simulate a renewable energy grid scenario.

    This is the primary command for scenario orchestration, providing one-command runs
    for comprehensive grid simulations with weather conditions, asset coordination,
    and emergency response testing.

    Args:
        scenario: Scenario name to simulate
        config_file: Optional configuration file path
        output_dir: Output directory for results
        output_format: Output format (json/csv)
        duration_hours: Optional simulation duration override
        verbose: Enable verbose logging
        enable_metrics: Enable metrics collection
        dry_run: Show configuration without execution

    Returns:
        Simulation results dictionary
    """
    return simulate_command(
        scenario=scenario,
        config_file=config_file,
        output_dir=output_dir,
        output_format=output_format,
        duration_hours=duration_hours,
        verbose=verbose,
        enable_metrics=enable_metrics,
        dry_run=dry_run,
    )


def simulate_command(
    scenario: str,
    config_file: str | None = None,
    output_dir: str = "output",
    output_format: str = "json",
    duration_hours: int | None = None,
    verbose: bool = False,
    enable_metrics: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute scenario simulation command.

    Args:
        scenario: Scenario name to simulate
        config_file: Optional configuration file path
        output_dir: Output directory for results
        output_format: Output format (json/csv)
        duration_hours: Optional simulation duration override
        verbose: Enable verbose logging
        enable_metrics: Enable metrics collection
        dry_run: Show configuration without execution

    Returns:
        Simulation results dictionary

    Raises:
        ValueError: If scenario is invalid
        FileNotFoundError: If config file not found
        PermissionError: If output directory not writable
    """
    # Validate inputs
    if not scenario or scenario.strip() == "":
        raise ValueError("Scenario name cannot be empty")

    # Setup logging
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Load configuration
    config = None
    if config_file:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        config = load_cli_config(config_file)
    else:
        config = create_default_config()

    # Create output directory
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot create output directory {output_dir}: {e}") from e

    # Show configuration for dry run
    if dry_run:
        if console:
            console.print(f"[bold blue]Scenario:[/bold blue] {scenario}")
            console.print(f"[bold blue]Duration:[/bold blue] {duration_hours or 'default'} hours")
            console.print(f"[bold blue]Output:[/bold blue] {output_dir}")
            console.print(f"[bold blue]Format:[/bold blue] {output_format}")
            console.print(f"[bold blue]Metrics:[/bold blue] {'enabled' if enable_metrics else 'disabled'}")
        else:
            print(f"Scenario: {scenario}")
            print(f"Duration: {duration_hours or 'default'} hours")
            print(f"Output: {output_dir}")
            print(f"Format: {output_format}")
            print(f"Metrics: {'enabled' if enable_metrics else 'disabled'}")
        return {"status": "dry_run", "scenario": scenario}

    # Display startup message
    if console:
        console.print(f"[bold green]🚀 Starting PSIREG simulation:[/bold green] {scenario}")
        console.print(f"[dim]Output directory: {output_dir}[/dim]")
    else:
        print(f"🚀 Starting PSIREG simulation: {scenario}")
        print(f"Output directory: {output_dir}")

    # Create and configure orchestrator
    try:
        orchestrator = ScenarioOrchestrator(config=config.model_dump() if config else None)

        # Execute scenario
        result = orchestrator.run_scenario(
            scenario_name=scenario,
            duration_hours=duration_hours,
            output_dir=str(output_path),
            output_format=output_format,
            enable_metrics=enable_metrics,
            verbose=verbose,
        )

        # Display results
        if console:
            if result.get("status") == "success":
                console.print("[bold green]✅ Simulation completed successfully![/bold green]")
                console.print(f"[dim]Duration: {result.get('duration', 'unknown')} seconds[/dim]")
                if "metrics" in result:
                    console.print(f"[dim]Metrics collected: {len(result['metrics'])} data points[/dim]")
            else:
                console.print(
                    f"[bold red]❌ Simulation failed: {result.get('error_message', 'Unknown error')}[/bold red]"
                )
        else:
            if result.get("status") == "success":
                print("✅ Simulation completed successfully!")
                print(f"Duration: {result.get('duration', 'unknown')} seconds")
                if "metrics" in result:
                    print(f"Metrics collected: {len(result['metrics'])} data points")
            else:
                print(f"❌ Simulation failed: {result.get('error_message', 'Unknown error')}")

        return result

    except KeyboardInterrupt:
        if console:
            console.print("[bold yellow]⚠️ Simulation interrupted by user[/bold yellow]")
        else:
            print("⚠️ Simulation interrupted by user")
        raise
    except Exception as e:
        if console:
            console.print(f"[bold red]❌ Simulation error: {e}[/bold red]")
        else:
            print(f"❌ Simulation error: {e}")
        return {"status": "error", "error_message": str(e)}


@app.command("list-scenarios")
def list_scenarios() -> list[str]:
    """List available simulation scenarios.

    Returns:
        List of available scenario names
    """
    return list_scenarios_command()


def list_scenarios_command() -> list[str]:
    """Execute list scenarios command.

    Returns:
        List of available scenario names
    """
    try:
        orchestrator = ScenarioOrchestrator()
        scenarios = orchestrator.list_scenarios()

        if console:
            table = Table(title="Available PSIREG Scenarios")
            table.add_column("Scenario", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Duration", style="green")

            for scenario_name in scenarios:
                try:
                    info = orchestrator.get_scenario_info(scenario_name)
                    table.add_row(
                        scenario_name,
                        info.get("description", "No description"),
                        f"{info.get('duration_hours', 'unknown')} hours",
                    )
                except Exception:
                    table.add_row(scenario_name, "Error loading info", "unknown")

            console.print(table)
        else:
            print("Available PSIREG Scenarios:")
            for scenario_name in scenarios:
                try:
                    info = orchestrator.get_scenario_info(scenario_name)
                    duration = info.get("duration_hours", "unknown")
                    description = info.get("description", "No description")
                    print(f"  {scenario_name:15} - {description} ({duration} hours)")
                except Exception:
                    print(f"  {scenario_name:15} - Error loading info")

        return scenarios

    except Exception as e:
        if console:
            console.print(f"[bold red]❌ Error listing scenarios: {e}[/bold red]")
        else:
            print(f"❌ Error listing scenarios: {e}")
        return []


@app.command()
def version() -> str:
    """Show PSIREG version information.

    Returns:
        Version string
    """
    return version_command()


def version_command() -> str:
    """Execute version command.

    Returns:
        Version string
    """
    from psireg import __version__

    version_info = f"PSIREG v{__version__}"

    if console:
        console.print(f"[bold cyan]{version_info}[/bold cyan]")
        console.print("[dim]Predictive Swarm Intelligence for Renewable Energy Grids[/dim]")
    else:
        print(version_info)
        print("Predictive Swarm Intelligence for Renewable Energy Grids")

    return version_info


def load_cli_config(config_file: str) -> PSIREGConfig:
    """Load CLI configuration from file.

    Args:
        config_file: Path to configuration file

    Returns:
        Loaded configuration

    Raises:
        FileNotFoundError: If config file not found
        Exception: If config loading fails
    """
    loader = YamlConfigLoader()
    return loader.load_config(config_file)


def create_default_config() -> PSIREGConfig:
    """Create default CLI configuration.

    Returns:
        Default configuration
    """
    return PSIREGConfig()


# Main entry point for console script
def main() -> None:
    """Main entry point for the CLI application."""
    if not _TYPER_AVAILABLE:
        print("❌ Error: Typer is required for CLI functionality")
        print("Install with: pip install typer rich")
        sys.exit(1)

    app()


if __name__ == "__main__":
    main()
