#!/usr/bin/env python3
"""Comprehensive demo of renewable agents (Solar/Wind) swarm coordination.

This demo showcases how SolarAgent and WindAgent use PPO forecasting combined
with pheromone gradient coordination to produce demand response signals for
grid load balancing.

Key features demonstrated:
1. PPO forecasting for renewable generation
2. Pheromone-based swarm coordination
3. Demand response signal generation
4. Multi-agent coordination scenarios
5. Grid support priority management

Run this demo to see how the swarm intelligence system coordinates renewable
assets to produce optimal demand response signals.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import PSIREG modules
from psireg.swarm import SwarmBus, PheromoneField, PheromoneType, GridPosition
from psireg.swarm.agents import SolarAgent, WindAgent
from psireg.sim.assets import SolarPanel, WindTurbine
from psireg.utils.enums import WeatherCondition
from psireg.rl.predictive_layer import PredictiveLayer

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False
    logger.warning("Matplotlib not available - skipping visualization")


@dataclass
class GridConditions:
    """Current grid conditions for simulation."""
    frequency_hz: float = 60.0
    voltage_kv: float = 138.0
    local_load_mw: float = 100.0
    local_generation_mw: float = 80.0
    electricity_price: float = 60.0
    wind_speed_ms: float = 12.0
    air_density_kg_m3: float = 1.225
    irradiance_w_m2: float = 800.0
    ambient_temp_c: float = 25.0
    weather_condition: WeatherCondition = WeatherCondition.CLEAR


@dataclass
class DemandResponseResult:
    """Result of demand response calculation."""
    agent_id: str
    signal_mw: float
    curtailment_factor: float
    confidence: float
    reason: str
    frequency_component: float
    economic_component: float
    coordination_component: float
    generation_mw: float
    capacity_mw: float


class RenewableAgentsDemo:
    """Demo class showcasing renewable agents coordination."""
    
    def __init__(self):
        """Initialize demo with renewable agents."""
        self.swarm_bus = SwarmBus(grid_width=20, grid_height=20)
        self.pheromone_field = PheromoneField(grid_width=10, grid_height=10)
        self.agents: List[SolarAgent | WindAgent] = []
        self.grid_conditions = GridConditions()
        self.time_step = 0
        self.demo_results: List[Dict[str, Any]] = []
        
        # Create demo assets and agents
        self._create_demo_agents()
        
        logger.info("Renewable Agents Demo initialized")
    
    def _create_demo_agents(self):
        """Create demo solar and wind agents."""
        # Create solar agents
        for i in range(3):
            solar_panel = SolarPanel(
                asset_id=f"solar_{i}",
                name=f"Solar Panel {i}",
                node_id=f"solar_node_{i}",
                capacity_mw=5.0,
                panel_efficiency=0.2,
                panel_area_m2=25000,
            )
            
            solar_agent = SolarAgent(
                solar_panel=solar_panel,
                agent_id=f"solar_agent_{i}",
                communication_range=3.0,
                coordination_weight=0.3
            )
            
            self.agents.append(solar_agent)
            
        # Create wind agents
        for i in range(3):
            wind_turbine = WindTurbine(
                asset_id=f"wind_{i}",
                name=f"Wind Turbine {i}",
                node_id=f"wind_node_{i}",
                capacity_mw=3.0,
                rotor_diameter_m=100.0,
                hub_height_m=80.0,
                cut_in_speed_ms=3.0,
                cut_out_speed_ms=25.0,
                rated_speed_ms=12.0,
            )
            
            wind_agent = WindAgent(
                wind_turbine=wind_turbine,
                agent_id=f"wind_agent_{i}",
                communication_range=3.0,
                coordination_weight=0.3
            )
            
            self.agents.append(wind_agent)
        
        logger.info(f"Created {len(self.agents)} renewable agents")
    
    def setup_ppo_predictors(self):
        """Set up PPO predictors for the agents."""
        logger.info("Setting up PPO predictors...")
        
        # In a real implementation, you would load trained PPO models
        # For this demo, we'll simulate PPO predictors
        for agent in self.agents:
            # Create a mock PPO predictor
            mock_predictor = MockPPOPredictor(agent_type=type(agent).__name__)
            agent.set_ppo_predictor(mock_predictor)
            
        logger.info("PPO predictors configured for all agents")
    
    def run_coordination_demo(self, time_steps: int = 48):
        """Run the main coordination demo.
        
        Args:
            time_steps: Number of time steps to simulate (default: 48 hours)
        """
        logger.info(f"Starting coordination demo for {time_steps} time steps...")
        
        # Set up PPO predictors
        self.setup_ppo_predictors()
        
        # Run simulation
        for step in range(time_steps):
            self.time_step = step
            hour_of_day = step % 24
            
            # Update grid conditions for this time step
            self._update_grid_conditions(hour_of_day)
            
            # Update agent conditions
            self._update_agent_conditions()
            
            # Calculate demand response signals
            demand_responses = self._calculate_demand_response()
            
            # Update pheromone field and coordination
            self._update_swarm_coordination(demand_responses)
            
            # Store results
            self._store_results(step, demand_responses)
            
            if step % 6 == 0:  # Log every 6 hours
                logger.info(f"Step {step}: {len(demand_responses)} agents produced demand response")
        
        logger.info("Coordination demo completed")
        return self.demo_results
    
    def _update_grid_conditions(self, hour_of_day: int):
        """Update grid conditions based on time of day."""
        # Daily patterns
        solar_factor = max(0.1, np.sin(np.pi * hour_of_day / 12))  # Peak at noon
        wind_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)  # Variable wind
        load_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Peak evening
        
        # Update conditions
        self.grid_conditions.irradiance_w_m2 = 1000.0 * solar_factor
        self.grid_conditions.wind_speed_ms = 10.0 * wind_factor
        self.grid_conditions.local_load_mw = 100.0 * load_factor
        
        # Add some frequency variation
        self.grid_conditions.frequency_hz = 60.0 + 0.1 * np.sin(2 * np.pi * hour_of_day / 24)
        
        # Price variations
        self.grid_conditions.electricity_price = 60.0 + 40.0 * load_factor
    
    def _update_agent_conditions(self):
        """Update all agent conditions."""
        for agent in self.agents:
            if isinstance(agent, SolarAgent):
                agent.solar_panel.set_irradiance(self.grid_conditions.irradiance_w_m2)
                agent.solar_panel.set_temperature(self.grid_conditions.ambient_temp_c)
                agent.solar_panel.set_weather_condition(self.grid_conditions.weather_condition)
            elif isinstance(agent, WindAgent):
                agent.wind_turbine.set_wind_speed(self.grid_conditions.wind_speed_ms)
                agent.wind_turbine.set_air_density(self.grid_conditions.air_density_kg_m3)
                agent.wind_turbine.set_weather_condition(self.grid_conditions.weather_condition)
            
            # Update grid conditions for all agents
            agent.update_grid_conditions(
                frequency_hz=self.grid_conditions.frequency_hz,
                voltage_kv=self.grid_conditions.voltage_kv,
                local_load_mw=self.grid_conditions.local_load_mw,
                local_generation_mw=self.grid_conditions.local_generation_mw,
                electricity_price=self.grid_conditions.electricity_price,
            )
    
    def _calculate_demand_response(self) -> List[DemandResponseResult]:
        """Calculate demand response signals from all agents."""
        demand_responses = []
        
        # Create forecasts for agents
        forecast_hours = 24
        forecast_times = list(range(forecast_hours))
        
        for agent in self.agents:
            if isinstance(agent, SolarAgent):
                # Solar forecasts
                forecast_irradiance = [
                    max(0, 1000 * np.sin(np.pi * (self.time_step + h) / 12))
                    for h in forecast_times
                ]
                forecast_prices = [
                    60.0 + 40.0 * (0.7 + 0.3 * np.sin(2 * np.pi * h / 24))
                    for h in forecast_times
                ]
                forecast_grid_stress = [
                    abs(0.1 * np.sin(2 * np.pi * h / 24))
                    for h in forecast_times
                ]
                
                # Calculate demand response
                dr_signal = agent.calculate_demand_response_signal(
                    forecast_irradiance=forecast_irradiance,
                    forecast_prices=forecast_prices,
                    forecast_grid_stress=forecast_grid_stress,
                    time_horizon_hours=forecast_hours,
                )
                
                # Create result
                result = DemandResponseResult(
                    agent_id=agent.agent_id,
                    signal_mw=dr_signal["signal_mw"],
                    curtailment_factor=dr_signal["curtailment_factor"],
                    confidence=dr_signal["confidence"],
                    reason=dr_signal["reason"],
                    frequency_component=dr_signal["frequency_component"],
                    economic_component=dr_signal["economic_component"],
                    coordination_component=dr_signal["coordination_component"],
                    generation_mw=agent.solar_panel.current_output_mw,
                    capacity_mw=agent.solar_panel.capacity_mw,
                )
                
            elif isinstance(agent, WindAgent):
                # Wind forecasts
                forecast_wind_speed = [
                    10.0 * (1.0 + 0.3 * np.sin(2 * np.pi * (self.time_step + h) / 24))
                    for h in forecast_times
                ]
                forecast_prices = [
                    60.0 + 40.0 * (0.7 + 0.3 * np.sin(2 * np.pi * h / 24))
                    for h in forecast_times
                ]
                forecast_grid_stress = [
                    abs(0.1 * np.sin(2 * np.pi * h / 24))
                    for h in forecast_times
                ]
                
                # Calculate demand response
                dr_signal = agent.calculate_demand_response_signal(
                    forecast_wind_speed=forecast_wind_speed,
                    forecast_prices=forecast_prices,
                    forecast_grid_stress=forecast_grid_stress,
                    time_horizon_hours=forecast_hours,
                )
                
                # Create result
                result = DemandResponseResult(
                    agent_id=agent.agent_id,
                    signal_mw=dr_signal["signal_mw"],
                    curtailment_factor=dr_signal["curtailment_factor"],
                    confidence=dr_signal["confidence"],
                    reason=dr_signal["reason"],
                    frequency_component=dr_signal["frequency_component"],
                    economic_component=dr_signal["economic_component"],
                    coordination_component=dr_signal["coordination_component"],
                    generation_mw=agent.wind_turbine.current_output_mw,
                    capacity_mw=agent.wind_turbine.capacity_mw,
                )
            
            demand_responses.append(result)
        
        return demand_responses
    
    def _update_swarm_coordination(self, demand_responses: List[DemandResponseResult]):
        """Update swarm coordination based on demand response signals."""
        # Calculate coordination signals for each agent
        coordination_signals = {}
        
        for result in demand_responses:
            # Convert demand response to coordination signal
            coord_signal = result.signal_mw / 10.0  # Normalize
            coordination_signals[result.agent_id] = coord_signal
        
        # Update agent coordination
        for agent in self.agents:
            # Get neighbor signals (agents within communication range)
            neighbor_signals = []
            for other_agent in self.agents:
                if other_agent.agent_id != agent.agent_id:
                    # Simple distance calculation (in a real implementation, use actual positions)
                    if other_agent.agent_id in coordination_signals:
                        neighbor_signals.append(coordination_signals[other_agent.agent_id])
            
            # Update agent with neighbor signals
            agent.update_swarm_signals(neighbor_signals)
    
    def _store_results(self, step: int, demand_responses: List[DemandResponseResult]):
        """Store results for analysis."""
        step_results = {
            "step": step,
            "hour_of_day": step % 24,
            "grid_conditions": {
                "frequency_hz": self.grid_conditions.frequency_hz,
                "voltage_kv": self.grid_conditions.voltage_kv,
                "local_load_mw": self.grid_conditions.local_load_mw,
                "local_generation_mw": self.grid_conditions.local_generation_mw,
                "electricity_price": self.grid_conditions.electricity_price,
                "irradiance_w_m2": self.grid_conditions.irradiance_w_m2,
                "wind_speed_ms": self.grid_conditions.wind_speed_ms,
            },
            "demand_responses": [
                {
                    "agent_id": dr.agent_id,
                    "signal_mw": dr.signal_mw,
                    "curtailment_factor": dr.curtailment_factor,
                    "confidence": dr.confidence,
                    "reason": dr.reason,
                    "generation_mw": dr.generation_mw,
                    "capacity_mw": dr.capacity_mw,
                }
                for dr in demand_responses
            ],
            "total_demand_response_mw": sum(dr.signal_mw for dr in demand_responses),
            "total_generation_mw": sum(dr.generation_mw for dr in demand_responses),
            "total_capacity_mw": sum(dr.capacity_mw for dr in demand_responses),
        }
        
        self.demo_results.append(step_results)
    
    def print_summary(self):
        """Print summary of demo results."""
        if not self.demo_results:
            logger.warning("No demo results available")
            return
        
        print("\n" + "="*80)
        print("RENEWABLE AGENTS SWARM COORDINATION DEMO SUMMARY")
        print("="*80)
        
        # Agent summary
        solar_agents = [a for a in self.agents if isinstance(a, SolarAgent)]
        wind_agents = [a for a in self.agents if isinstance(a, WindAgent)]
        
        print(f"Total Agents: {len(self.agents)}")
        print(f"  Solar Agents: {len(solar_agents)}")
        print(f"  Wind Agents: {len(wind_agents)}")
        print(f"Total Capacity: {sum(a.capacity_mw for a in [ag.solar_panel for ag in solar_agents] + [ag.wind_turbine for ag in wind_agents]):.1f} MW")
        
        # Performance summary
        total_dr_signals = [result["total_demand_response_mw"] for result in self.demo_results]
        total_generation = [result["total_generation_mw"] for result in self.demo_results]
        
        print(f"\nPerformance Summary:")
        print(f"  Average Demand Response: {np.mean(total_dr_signals):.2f} MW")
        print(f"  Max Demand Response: {np.max(total_dr_signals):.2f} MW")
        print(f"  Average Generation: {np.mean(total_generation):.2f} MW")
        print(f"  Max Generation: {np.max(total_generation):.2f} MW")
        
        # Show sample demand response reasons
        print(f"\nSample Demand Response Reasons:")
        for i, result in enumerate(self.demo_results[::12]):  # Every 12 hours
            print(f"  Hour {result['hour_of_day']}: {result['demand_responses'][0]['reason']}")
            if i >= 3:  # Show first 4 samples
                break
        
        print("\n" + "="*80)
        print("Demo completed successfully!")
        print("Key achievements:")
        print("✓ PPO forecasting integration")
        print("✓ Pheromone-based coordination")
        print("✓ Demand response signal generation")
        print("✓ Multi-agent renewable coordination")
        print("="*80)
    
    def create_visualizations(self):
        """Create visualizations of demo results."""
        if not HAVE_MATPLOTLIB:
            logger.warning("Matplotlib not available - skipping visualizations")
            return
        
        if not self.demo_results:
            logger.warning("No demo results available for visualization")
            return
        
        # Extract data for plotting
        hours = [r["hour_of_day"] for r in self.demo_results]
        demand_response = [r["total_demand_response_mw"] for r in self.demo_results]
        generation = [r["total_generation_mw"] for r in self.demo_results]
        electricity_price = [r["grid_conditions"]["electricity_price"] for r in self.demo_results]
        frequency = [r["grid_conditions"]["frequency_hz"] for r in self.demo_results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Renewable Agents Swarm Coordination Demo Results', fontsize=16)
        
        # Plot 1: Demand Response vs Generation
        axes[0, 0].plot(hours, demand_response, 'b-', label='Demand Response', linewidth=2)
        axes[0, 0].plot(hours, generation, 'g-', label='Generation', linewidth=2)
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Power (MW)')
        axes[0, 0].set_title('Demand Response vs Generation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Electricity Price
        axes[0, 1].plot(hours, electricity_price, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Price ($/MWh)')
        axes[0, 1].set_title('Electricity Price')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Grid Frequency
        axes[1, 0].plot(hours, frequency, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        axes[1, 0].set_title('Grid Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Agent-specific demand response
        solar_dr = []
        wind_dr = []
        for result in self.demo_results:
            solar_signals = [dr["signal_mw"] for dr in result["demand_responses"] if "solar" in dr["agent_id"]]
            wind_signals = [dr["signal_mw"] for dr in result["demand_responses"] if "wind" in dr["agent_id"]]
            solar_dr.append(sum(solar_signals))
            wind_dr.append(sum(wind_signals))
        
        axes[1, 1].plot(hours, solar_dr, 'orange', label='Solar Agents', linewidth=2)
        axes[1, 1].plot(hours, wind_dr, 'cyan', label='Wind Agents', linewidth=2)
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Demand Response (MW)')
        axes[1, 1].set_title('Agent-Specific Demand Response')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("examples/output")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "renewable_agents_demo.png", dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_dir / 'renewable_agents_demo.png'}")
        
        plt.show()


class MockPPOPredictor:
    """Mock PPO predictor for demo purposes."""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Mock prediction based on observation."""
        # Simple rule-based prediction for demo
        if self.agent_type == "SolarAgent":
            # Solar prediction based on irradiance proxy
            irradiance_proxy = observation[2] if len(observation) > 2 else 0.5
            return np.array([irradiance_proxy * 0.8 + 0.1])
        elif self.agent_type == "WindAgent":
            # Wind prediction based on wind speed proxy
            wind_proxy = observation[2] if len(observation) > 2 else 0.5
            return np.array([wind_proxy * 0.9 + 0.05])
        else:
            return np.array([0.5])


def main():
    """Main demo function."""
    logger.info("Starting Renewable Agents Swarm Coordination Demo")
    
    # Create and run demo
    demo = RenewableAgentsDemo()
    
    # Run coordination demo
    results = demo.run_coordination_demo(time_steps=48)  # 48 hours
    
    # Print summary
    demo.print_summary()
    
    # Create visualizations
    demo.create_visualizations()
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main() 