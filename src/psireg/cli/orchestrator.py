"""Scenario Orchestrator for PSIREG renewable energy grid system.

This module provides scenario orchestration functionality for running
comprehensive renewable energy grid simulations with predefined scenarios,
weather conditions, and emergency response testing.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from psireg.config.schema import GridConfig, SimulationConfig
from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.assets.thermal import CoalPlant, NaturalGasPlant
from psireg.sim.assets.wind import WindTurbine
from psireg.sim.engine import GridEngine, NetworkNode, TransmissionLine
from psireg.sim.metrics import MetricsCollector
from psireg.swarm.agents.battery_agent import BatteryAgent
from psireg.swarm.agents.demand_agent import DemandAgent
from psireg.swarm.agents.solar_agent import SolarAgent
from psireg.swarm.agents.wind_agent import WindAgent
from psireg.swarm.pheromone import SwarmBus
from psireg.utils.enums import AssetStatus, AssetType, WeatherCondition
from psireg.utils.logger import logger

# Import scenario implementations
from .scenarios.storm_day import StormDayScenario


class ScenarioOrchestrator:
    """Orchestrates comprehensive renewable energy grid scenarios.

    The ScenarioOrchestrator manages the execution of predefined scenarios
    including weather conditions, asset coordination, emergency response,
    and metrics collection for comprehensive grid simulation testing.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize scenario orchestrator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.grid_engine: GridEngine | None = None
        self.metrics_collector: MetricsCollector | None = None
        self.swarm_bus: SwarmBus | None = None
        self.agents: list[Any] = []

        # Available scenarios
        self.scenarios = {
            "storm_day": StormDayScenario(),
            "peak_demand": self._create_peak_demand_scenario(),
            "normal": self._create_normal_scenario(),
            "renewable_surge": self._create_renewable_surge_scenario(),
            "grid_outage": self._create_grid_outage_scenario(),
        }

        logger.info("ScenarioOrchestrator initialized with %d scenarios", len(self.scenarios))

    def list_scenarios(self) -> list[str]:
        """List available scenarios.

        Returns:
            List of available scenario names
        """
        return list(self.scenarios.keys())

    def get_scenario_info(self, scenario_name: str) -> dict[str, Any]:
        """Get information about a specific scenario.

        Args:
            scenario_name: Name of the scenario

        Returns:
            Scenario information dictionary

        Raises:
            ValueError: If scenario not found
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]
        return {
            "name": scenario.name,
            "description": scenario.description,
            "duration_hours": scenario.duration_hours,
            "weather_conditions": scenario.get_weather_conditions(),
            "features": getattr(scenario, "features", []),
        }

    def run_scenario(
        self,
        scenario_name: str,
        duration_hours: int | None = None,
        output_dir: str | None = None,
        output_format: str = "json",
        enable_metrics: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run a complete scenario simulation.

        Args:
            scenario_name: Name of the scenario to run
            duration_hours: Optional duration override
            output_dir: Optional output directory
            output_format: Output format (json/csv)
            enable_metrics: Enable metrics collection
            verbose: Enable verbose logging

        Returns:
            Simulation results dictionary
        """
        start_time = time.time()

        try:
            # Validate scenario
            if not self._validate_scenario_execution(scenario_name):
                return {"status": "error", "error_message": f"Invalid scenario: {scenario_name}"}

            scenario = self.scenarios[scenario_name]
            actual_duration = duration_hours or scenario.duration_hours

            logger.info("Starting scenario '%s' for %d hours", scenario_name, actual_duration)

            # Setup components
            self._setup_grid_engine(scenario_name)
            if enable_metrics:
                self._setup_metrics_collector()
            self._setup_swarm_agents(scenario_name)

            # Execute simulation
            results = self._execute_simulation(scenario, actual_duration, verbose)

            # Collect final metrics
            if enable_metrics and self.metrics_collector:
                results["metrics"] = self._collect_final_metrics()

            # Generate output files
            if output_dir:
                output_files = self._generate_output_files(results, output_dir, output_format)
                results["output_files"] = output_files

            # Calculate execution time
            execution_time = time.time() - start_time
            results.update(
                {
                    "status": "success",
                    "scenario_name": scenario_name,
                    "duration_hours": actual_duration,
                    "execution_time_seconds": execution_time,
                    "completion_time": datetime.now().isoformat(),
                }
            )

            logger.info("Scenario '%s' completed successfully in %.2f seconds", scenario_name, execution_time)
            return results

        except KeyboardInterrupt:
            logger.warning("Scenario execution interrupted by user")
            return {
                "status": "interrupted",
                "scenario_name": scenario_name,
                "execution_time_seconds": time.time() - start_time,
            }
        except Exception as e:
            logger.error("Scenario execution failed: %s", e)
            return {
                "status": "error",
                "scenario_name": scenario_name,
                "error_message": str(e),
                "execution_time_seconds": time.time() - start_time,
            }

    def _setup_grid_engine(self, scenario_name: str) -> None:
        """Setup grid engine for scenario.

        Args:
            scenario_name: Name of the scenario
        """
        # Create simulation and grid configurations
        sim_config = SimulationConfig(
            timestep_minutes=self.config.get("simulation", {}).get("timestep_minutes", 15),
            horizon_hours=self.config.get("simulation", {}).get("horizon_hours", 24),
        )

        grid_config = GridConfig(
            frequency_hz=self.config.get("grid", {}).get("frequency_hz", 60.0),
            voltage_kv=self.config.get("grid", {}).get("voltage_kv", 230.0),
        )

        # Create grid engine
        self.grid_engine = GridEngine(sim_config, grid_config)

        # Setup grid topology
        self._setup_grid_topology()

        # Add assets based on scenario
        self._setup_scenario_assets(scenario_name)

        logger.info("Grid engine setup completed for scenario '%s'", scenario_name)

    def _setup_grid_topology(self) -> None:
        """Setup basic grid topology."""
        # Create network nodes
        nodes = [
            NetworkNode(node_id="gen_node_1", name="Generation Hub 1", voltage_kv=230.0),
            NetworkNode(node_id="gen_node_2", name="Generation Hub 2", voltage_kv=230.0),
            NetworkNode(node_id="load_node_1", name="Load Center 1", voltage_kv=138.0),
            NetworkNode(node_id="load_node_2", name="Load Center 2", voltage_kv=138.0),
            NetworkNode(node_id="storage_node", name="Storage Hub", voltage_kv=138.0),
        ]

        for node in nodes:
            self.grid_engine.add_node(node)

        # Create transmission lines
        lines = [
            TransmissionLine(
                line_id="line_1",
                name="Main Transmission 1",
                from_node="gen_node_1",
                to_node="load_node_1",
                capacity_mw=500.0,
                length_km=150.0,
                resistance=0.05,
            ),
            TransmissionLine(
                line_id="line_2",
                name="Main Transmission 2",
                from_node="gen_node_2",
                to_node="load_node_2",
                capacity_mw=500.0,
                length_km=120.0,
                resistance=0.04,
            ),
            TransmissionLine(
                line_id="line_3",
                name="Interconnect",
                from_node="load_node_1",
                to_node="load_node_2",
                capacity_mw=300.0,
                length_km=80.0,
                resistance=0.03,
            ),
            TransmissionLine(
                line_id="line_4",
                name="Storage Link",
                from_node="storage_node",
                to_node="load_node_1",
                capacity_mw=200.0,
                length_km=50.0,
                resistance=0.02,
            ),
        ]

        for line in lines:
            self.grid_engine.add_transmission_line(line)

    def _setup_scenario_assets(self, scenario_name: str) -> None:
        """Setup assets for specific scenario.

        Args:
            scenario_name: Name of the scenario
        """
        scenario = self.scenarios[scenario_name]
        asset_config = scenario.get_asset_configuration()

        # Add renewable assets
        renewable_config = asset_config.get("renewable", {})
        self._add_renewable_assets(renewable_config)

        # Add storage assets
        storage_config = asset_config.get("storage", {})
        self._add_storage_assets(storage_config)

        # Add thermal assets
        thermal_config = asset_config.get("thermal", {})
        self._add_thermal_assets(thermal_config)

        # Add load assets
        load_config = asset_config.get("load", {})
        self._add_load_assets(load_config)

    def _add_renewable_assets(self, config: dict[str, Any]) -> None:
        """Add renewable energy assets."""
        # Wind farms
        wind_config = config.get("wind_farms", {})
        wind_count = wind_config.get("count", 3)

        for i in range(wind_count):
            wind = WindTurbine(
                asset_id=f"wind_{i+1:03d}",
                name=f"Wind Farm {i+1}",
                node_id="gen_node_1" if i < wind_count // 2 else "gen_node_2",
                capacity_mw=wind_config.get("capacity_mw", 50.0),
                rotor_diameter_m=wind_config.get("rotor_diameter_m", 120.0),
                hub_height_m=wind_config.get("hub_height_m", 100.0),
            )
            wind.set_status(AssetStatus.ONLINE)
            self.grid_engine.add_asset(wind)

        # Solar farms
        solar_config = config.get("solar_farms", {})
        solar_count = solar_config.get("count", 4)

        for i in range(solar_count):
            solar = SolarPanel(
                asset_id=f"solar_{i+1:03d}",
                name=f"Solar Farm {i+1}",
                node_id="gen_node_1" if i < solar_count // 2 else "gen_node_2",
                capacity_mw=solar_config.get("capacity_mw", 30.0),
                panel_efficiency=solar_config.get("efficiency", 0.20),
                panel_area_m2=solar_config.get("area_m2", 150000.0),
            )
            solar.set_status(AssetStatus.ONLINE)
            self.grid_engine.add_asset(solar)

    def _add_storage_assets(self, config: dict[str, Any]) -> None:
        """Add energy storage assets."""
        battery_config = config.get("batteries", {})
        battery_count = battery_config.get("count", 2)

        for i in range(battery_count):
            battery = Battery(
                asset_id=f"battery_{i+1:03d}",
                name=f"Battery Storage {i+1}",
                node_id="storage_node",
                capacity_mw=battery_config.get("capacity_mw", 100.0),
                energy_capacity_mwh=battery_config.get("energy_capacity_mwh", 400.0),
                initial_soc_percent=battery_config.get("initial_soc", 50.0),
            )
            battery.set_status(AssetStatus.ONLINE)
            self.grid_engine.add_asset(battery)

    def _add_thermal_assets(self, config: dict[str, Any]) -> None:
        """Add thermal generation assets."""
        # Natural gas plants
        gas_config = config.get("natural_gas", {})
        for i in range(gas_config.get("count", 2)):
            gas_plant = NaturalGasPlant(
                asset_id=f"gas_{i+1:03d}",
                name=f"Natural Gas Plant {i+1}",
                node_id="gen_node_2",
                capacity_mw=gas_config.get("capacity_mw", 200.0),
                min_output_mw=gas_config.get("min_output_mw", 50.0),
                ramp_rate_mw_per_min=gas_config.get("ramp_rate", 10.0),
            )
            gas_plant.set_status(AssetStatus.ONLINE)
            self.grid_engine.add_asset(gas_plant)

        # Coal plants (if configured)
        coal_config = config.get("coal", {})
        for i in range(coal_config.get("count", 0)):
            coal_plant = CoalPlant(
                asset_id=f"coal_{i+1:03d}",
                name=f"Coal Plant {i+1}",
                node_id="gen_node_1",
                capacity_mw=coal_config.get("capacity_mw", 300.0),
                min_output_mw=coal_config.get("min_output_mw", 100.0),
                ramp_rate_mw_per_min=coal_config.get("ramp_rate", 5.0),
            )
            coal_plant.set_status(AssetStatus.ONLINE)
            self.grid_engine.add_asset(coal_plant)

    def _add_load_assets(self, config: dict[str, Any]) -> None:
        """Add load assets."""
        load_config = config.get("demand_response", {})
        load_count = load_config.get("count", 5)

        for i in range(load_count):
            load = Load(
                asset_id=f"load_{i+1:03d}",
                name=f"Load Center {i+1}",
                node_id="load_node_1" if i < load_count // 2 else "load_node_2",
                capacity_mw=load_config.get("capacity_mw", 150.0),
                baseline_demand_mw=load_config.get("baseline_mw", 100.0),
                dr_capability_mw=load_config.get("dr_capability_mw", 30.0),
            )
            load.set_status(AssetStatus.ONLINE)
            self.grid_engine.add_asset(load)

    def _setup_metrics_collector(self) -> None:
        """Setup metrics collection."""
        log_directory = Path("logs/metrics")
        log_directory.mkdir(parents=True, exist_ok=True)

        self.metrics_collector = MetricsCollector(log_directory=log_directory, enable_mae_calculation=True)

        # Register all built-in hooks
        self.metrics_collector.register_default_hooks()

        logger.info("Metrics collector setup completed")

    def _setup_swarm_agents(self, scenario_name: str) -> None:
        """Setup swarm intelligence agents.

        Args:
            scenario_name: Name of the scenario
        """
        # Create swarm bus
        self.swarm_bus = SwarmBus(
            grid_width=20,
            grid_height=20,
            pheromone_decay=0.95,
            pheromone_diffusion=0.1,
            time_step_s=15.0 * 60.0,  # 15 minutes
            communication_range=5.0,
        )

        # Create agents for assets
        self.agents = []

        for asset in self.grid_engine.get_all_assets():
            agent = None

            if asset.asset_type == AssetType.BATTERY:
                agent = BatteryAgent(asset)
            elif asset.asset_type == AssetType.WIND:
                agent = WindAgent(asset)
            elif asset.asset_type == AssetType.SOLAR:
                agent = SolarAgent(asset)
            elif asset.asset_type == AssetType.LOAD:
                agent = DemandAgent(asset)

            if agent:
                # Register agent with swarm bus
                position = self._calculate_agent_position(len(self.agents))
                self.swarm_bus.register_agent(agent, position)
                self.agents.append(agent)

        logger.info("Swarm agents setup completed with %d agents", len(self.agents))

    def _calculate_agent_position(self, agent_index: int) -> Any:
        """Calculate position for agent in swarm grid."""
        from psireg.swarm.pheromone import GridPosition

        grid_size = 20
        x = (agent_index * 3) % grid_size
        y = (agent_index * 3) // grid_size
        return GridPosition(x=x, y=y)

    def _execute_simulation(self, scenario: Any, duration_hours: int, verbose: bool) -> dict[str, Any]:
        """Execute the main simulation.

        Args:
            scenario: Scenario instance
            duration_hours: Simulation duration in hours
            verbose: Enable verbose output

        Returns:
            Simulation results
        """
        logger.info("Executing simulation for %d hours", duration_hours)

        # Get weather timeline
        weather_timeline = scenario.get_weather_timeline()
        grid_events = scenario.get_grid_events()

        # Simulation results
        results = {"grid_data": [], "weather_data": [], "agent_data": [], "event_log": []}

        # Execute simulation steps
        total_steps = duration_hours * 4  # 15-minute steps

        for step in range(total_steps):
            current_hour = step // 4

            # Apply weather conditions
            if current_hour < len(weather_timeline):
                weather = weather_timeline[current_hour]
                self._apply_weather_conditions(weather)
                results["weather_data"].append({"step": step, "hour": current_hour, **weather})

            # Process grid events
            for event in grid_events:
                if event.get("hour") == current_hour:
                    self._process_grid_event(event)
                    results["event_log"].append({"step": step, "hour": current_hour, **event})

            # Update swarm agents
            if self.swarm_bus:
                coordination_results = self.swarm_bus.coordinate_agents(update_pheromones=True)
                results["agent_data"].append({"step": step, "coordination": coordination_results})

            # Execute grid engine step
            if self.grid_engine:
                self.grid_engine.step()
                grid_state = self.grid_engine.get_state()
                results["grid_data"].append(
                    {
                        "step": step,
                        "timestamp": grid_state.timestamp.isoformat(),
                        "frequency_hz": grid_state.frequency_hz,
                        "total_load_mw": grid_state.total_load_mw,
                        "total_generation_mw": grid_state.total_generation_mw,
                        "total_renewable_mw": grid_state.total_renewable_mw,
                    }
                )

            # Collect metrics
            if self.metrics_collector:
                self.metrics_collector.collect_metrics(self.grid_engine)

            if verbose and step % 24 == 0:  # Log every 6 hours
                logger.info("Simulation progress: %.1f%% (step %d/%d)", (step / total_steps) * 100, step, total_steps)

        logger.info("Simulation execution completed")
        return results

    def _apply_weather_conditions(self, weather: dict[str, Any]) -> None:
        """Apply weather conditions to assets.

        Args:
            weather: Weather conditions dictionary
        """
        condition = weather.get("condition", WeatherCondition.CLEAR)
        wind_speed = weather.get("wind_speed_ms", 10.0)
        irradiance = weather.get("irradiance_w_m2", 800.0)
        temperature = weather.get("temperature_c", 25.0)
        air_density = weather.get("air_density_kg_m3", 1.225)

        for asset in self.grid_engine.get_all_assets():
            if hasattr(asset, "set_weather_condition"):
                asset.set_weather_condition(condition)

            if asset.asset_type == AssetType.WIND:
                asset.set_wind_speed(wind_speed)
                asset.set_air_density(air_density)
            elif asset.asset_type == AssetType.SOLAR:
                asset.set_irradiance(irradiance)
                asset.set_temperature(temperature)

    def _process_grid_event(self, event: dict[str, Any]) -> None:
        """Process a grid event.

        Args:
            event: Grid event dictionary
        """
        event_type = event.get("type")

        if event_type == "transmission_line_outage":
            line_id = event.get("line_id")
            duration = event.get("duration_hours", 1)
            logger.info("Processing transmission outage: %s for %d hours", line_id, duration)

        elif event_type == "generation_trip":
            asset_id = event.get("asset_id")
            logger.info("Processing generation trip: %s", asset_id)
            asset = self.grid_engine.get_asset(asset_id)
            if asset:
                asset.set_status(AssetStatus.OFFLINE)

        elif event_type == "emergency_response":
            trigger = event.get("trigger")
            logger.info("Processing emergency response: %s", trigger)
            self._activate_emergency_response()

    def _activate_emergency_response(self) -> None:
        """Activate emergency response procedures."""
        if self.swarm_bus:
            from psireg.swarm.pheromone import PheromoneType

            # Deposit emergency pheromones
            for agent in self.agents:
                self.swarm_bus.deposit_pheromone(agent.agent_id, PheromoneType.EMERGENCY_RESPONSE, 0.8)

    def _collect_final_metrics(self) -> dict[str, Any]:
        """Collect final metrics from simulation.

        Returns:
            Final metrics dictionary
        """
        if not self.metrics_collector:
            return {}

        # Export metrics
        summary = self.metrics_collector.export_summary()
        performance = self.metrics_collector.get_performance_stats()

        return {
            "summary": summary,
            "performance": performance,
            "total_hooks": len(self.metrics_collector.hooks),
            "collection_count": sum(hook.collection_count for hook in self.metrics_collector.hooks.values()),
        }

    def _generate_output_files(self, results: dict[str, Any], output_dir: str, output_format: str) -> dict[str, str]:
        """Generate output files.

        Args:
            results: Simulation results
            output_dir: Output directory
            output_format: Output format

        Returns:
            Dictionary of generated file paths
        """
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_files = {}

        # Generate summary file
        summary_file = output_path / f"simulation_summary_{timestamp}.json"
        summary_data = {
            "scenario": results.get("scenario_name"),
            "status": results.get("status"),
            "duration_hours": results.get("duration_hours"),
            "execution_time": results.get("execution_time_seconds"),
            "total_steps": len(results.get("grid_data", [])),
            "events_processed": len(results.get("event_log", [])),
        }

        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        output_files["summary"] = str(summary_file)

        # Generate metrics file
        if "metrics" in results:
            metrics_file = output_path / f"metrics_{timestamp}.json"
            with open(metrics_file, "w") as f:
                json.dump(results["metrics"], f, indent=2)
            output_files["metrics"] = str(metrics_file)

        # Generate grid data file
        if output_format == "json":
            grid_file = output_path / f"grid_data_{timestamp}.json"
            with open(grid_file, "w") as f:
                json.dump(results.get("grid_data", []), f, indent=2)
            output_files["grid_data"] = str(grid_file)
        else:  # CSV format
            import csv

            grid_file = output_path / f"grid_data_{timestamp}.csv"
            grid_data = results.get("grid_data", [])
            if grid_data:
                with open(grid_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=grid_data[0].keys())
                    writer.writeheader()
                    writer.writerows(grid_data)
                output_files["grid_data"] = str(grid_file)

        return output_files

    def _validate_scenario_execution(self, scenario_name: str) -> bool:
        """Validate scenario can be executed.

        Args:
            scenario_name: Name of the scenario

        Returns:
            True if scenario is valid
        """
        return scenario_name in self.scenarios

    def _validate_scenario_config(self, config: dict[str, Any]) -> bool:
        """Validate scenario configuration.

        Args:
            config: Scenario configuration

        Returns:
            True if configuration is valid
        """
        required_fields = ["name", "description", "duration_hours", "weather_conditions", "grid_events", "assets"]
        return all(field in config for field in required_fields)

    def _validate_scenario_results(self, results: dict[str, Any]) -> bool:
        """Validate scenario results.

        Args:
            results: Scenario results

        Returns:
            True if results are valid
        """
        required_fields = ["status", "duration", "metrics", "grid_data"]
        return all(field in results for field in required_fields)

    def _validate_output_directory(self, output_dir: str) -> bool:
        """Validate output directory.

        Args:
            output_dir: Output directory path

        Returns:
            True if directory is valid and writable
        """
        try:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            return True
        except (PermissionError, OSError):
            return False

    def _load_scenario_config(self, scenario_name: str) -> dict[str, Any]:
        """Load scenario configuration.

        Args:
            scenario_name: Name of the scenario

        Returns:
            Scenario configuration dictionary
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]
        return {
            "name": scenario.name,
            "description": scenario.description,
            "duration_hours": scenario.duration_hours,
            "weather_conditions": scenario.get_weather_conditions(),
            "grid_events": scenario.get_grid_events(),
            "assets": scenario.get_asset_configuration(),
        }

    # Simple scenario implementations for basic scenarios
    def _create_peak_demand_scenario(self):
        """Create peak demand scenario."""
        from .scenarios.base import DefaultScenario

        class PeakDemandScenario(DefaultScenario):
            def __init__(self):
                super().__init__()
                self.name = "peak_demand"
                self.description = "High demand scenario with peak load conditions"
                self.duration_hours = 12

        return PeakDemandScenario()

    def _create_normal_scenario(self):
        """Create normal operating scenario."""
        from .scenarios.base import DefaultScenario

        class NormalScenario(DefaultScenario):
            def __init__(self):
                super().__init__()
                self.name = "normal"
                self.description = "Normal grid operations with typical load patterns"
                self.duration_hours = 24

        return NormalScenario()

    def _create_renewable_surge_scenario(self):
        """Create renewable energy surge scenario."""
        from .scenarios.base import DefaultScenario

        class RenewableSurgeScenario(DefaultScenario):
            def __init__(self):
                super().__init__()
                self.name = "renewable_surge"
                self.description = "High renewable energy output scenario"
                self.duration_hours = 8

        return RenewableSurgeScenario()

    def _create_grid_outage_scenario(self):
        """Create grid outage scenario."""
        from .scenarios.base import DefaultScenario

        class GridOutageScenario(DefaultScenario):
            def __init__(self):
                super().__init__()
                self.name = "grid_outage"
                self.description = "Major grid outage and recovery scenario"
                self.duration_hours = 6

        return GridOutageScenario()
