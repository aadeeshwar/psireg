"""Grid simulation engine for PSIREG renewable energy system.

This module provides the core simulation engine that evolves grid state with minimal physics,
including network topology, power flow balance, frequency/voltage tracking, and asset scheduling.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator

from psireg.config.schema import GridConfig, SimulationConfig
from psireg.sim.assets.base import Asset
from psireg.utils.enums import AssetType
from psireg.utils.logger import logger  # type: ignore
from psireg.utils.types import MW, Hz, Timestamp, kV


class GridState(BaseModel):
    """Represents the current state of the electrical grid.

    This class captures the essential grid parameters at a point in time,
    including power flows, frequency, voltage, and balance.
    """

    timestamp: Timestamp = Field(..., description="Timestamp of the grid state")
    frequency_hz: Hz = Field(..., gt=0.0, description="Grid frequency in Hz")
    total_generation_mw: MW = Field(..., ge=0.0, description="Total generation in MW")
    total_load_mw: MW = Field(..., ge=0.0, description="Total load in MW")
    total_storage_mw: MW = Field(..., description="Total storage charging (+) or discharging (-) in MW")
    grid_losses_mw: MW = Field(default=0.0, ge=0.0, description="Grid transmission losses in MW")

    @field_validator("frequency_hz")
    @classmethod
    def validate_frequency(cls, v: float) -> float:
        """Validate frequency is positive."""
        if v <= 0:
            raise ValueError("Frequency must be positive")
        return v

    @property
    def power_balance_mw(self) -> MW:
        """Calculate power balance: generation - load - storage - losses.

        Returns:
            Power balance in MW (positive = excess generation, negative = deficit)
        """
        return self.total_generation_mw - self.total_load_mw - self.total_storage_mw - self.grid_losses_mw

    @property
    def is_balanced(self) -> bool:
        """Check if grid is in power balance (within tolerance).

        Returns:
            True if grid is balanced within 1 MW tolerance
        """
        return abs(self.power_balance_mw) <= 1.0  # 1 MW tolerance

    def __str__(self) -> str:
        """String representation of grid state."""
        return (
            f"GridState(f={self.frequency_hz:.2f}Hz, "
            f"gen={self.total_generation_mw:.1f}MW, "
            f"load={self.total_load_mw:.1f}MW, "
            f"balance={self.power_balance_mw:.1f}MW)"
        )


class NetworkNode(BaseModel):
    """Represents a node in the electrical network topology.

    Network nodes are connection points where assets are connected
    and power flows are aggregated.
    """

    node_id: str = Field(..., min_length=1, description="Unique identifier for the node")
    name: str = Field(..., min_length=1, description="Human-readable name of the node")
    voltage_kv: kV = Field(..., gt=0.0, description="Nominal voltage level in kV")
    latitude: float = Field(default=0.0, ge=-90.0, le=90.0, description="Latitude in degrees")
    longitude: float = Field(default=0.0, ge=-180.0, le=180.0, description="Longitude in degrees")
    current_frequency_hz: Hz = Field(default=60.0, gt=0.0, description="Current frequency at node in Hz")
    current_voltage_kv: kV = Field(default=0.0, description="Current voltage at node in kV")

    def model_post_init(self, __context: Any) -> None:
        """Initialize current voltage to nominal voltage if not set."""
        if self.current_voltage_kv == 0.0:
            self.current_voltage_kv = self.voltage_kv

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        """Validate node ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Node ID cannot be empty")
        return v.strip()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate node name is not empty."""
        if not v or not v.strip():
            raise ValueError("Node name cannot be empty")
        return v.strip()

    def __str__(self) -> str:
        """String representation of network node."""
        return f"Node({self.node_id}: {self.name}, {self.voltage_kv:.1f}kV)"


class TransmissionLine(BaseModel):
    """Represents a transmission line connecting two network nodes.

    Transmission lines carry power between nodes and have capacity limits
    and resistance characteristics.
    """

    line_id: str = Field(..., min_length=1, description="Unique identifier for the line")
    name: str = Field(..., min_length=1, description="Human-readable name of the line")
    from_node: str = Field(..., min_length=1, description="ID of the source node")
    to_node: str = Field(..., min_length=1, description="ID of the destination node")
    capacity_mw: MW = Field(..., gt=0.0, description="Maximum power capacity in MW")
    length_km: float = Field(..., gt=0.0, description="Length of the line in km")
    resistance: float = Field(..., gt=0.0, description="Electrical resistance in ohms")
    current_flow_mw: MW = Field(default=0.0, description="Current power flow in MW")

    @field_validator("line_id")
    @classmethod
    def validate_line_id(cls, v: str) -> str:
        """Validate line ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Line ID cannot be empty")
        return v.strip()

    @field_validator("from_node")
    @classmethod
    def validate_from_node(cls, v: str) -> str:
        """Validate from_node is not empty."""
        if not v or not v.strip():
            raise ValueError("From node cannot be empty")
        return v.strip()

    @field_validator("to_node")
    @classmethod
    def validate_to_node(cls, v: str) -> str:
        """Validate to_node is not empty."""
        if not v or not v.strip():
            raise ValueError("To node cannot be empty")
        return v.strip()

    @field_validator("capacity_mw")
    @classmethod
    def validate_capacity(cls, v: float) -> float:
        """Validate capacity is positive."""
        if v <= 0:
            raise ValueError("Capacity must be positive")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that from_node and to_node are different."""
        if self.from_node == self.to_node:
            raise ValueError("From node and to node must be different")

    def is_overloaded(self) -> bool:
        """Check if line is over capacity.

        Returns:
            True if current flow exceeds capacity
        """
        return abs(self.current_flow_mw) > self.capacity_mw

    def utilization_percent(self) -> float:
        """Calculate line utilization as percentage of capacity.

        Returns:
            Utilization percentage (0-100+)
        """
        return (abs(self.current_flow_mw) / self.capacity_mw) * 100.0

    def __str__(self) -> str:
        """String representation of transmission line."""
        return f"Line({self.line_id}: {self.from_node}->{self.to_node}, {self.capacity_mw:.1f}MW)"


class GridEngine:
    """Main simulation engine for the electrical grid.

    This class manages the simulation of the electrical grid, including:
    - Network topology (nodes and transmission lines)
    - Power flow balance (generation - load ± storage)
    - Frequency and voltage tracking
    - Asset scheduling and tick() calls
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        grid_config: GridConfig,
        start_time: datetime | None = None,
    ):
        """Initialize the grid simulation engine.

        Args:
            simulation_config: Configuration for simulation parameters
            grid_config: Configuration for grid system parameters
            start_time: Optional start time for simulation (defaults to epoch)
        """
        self.simulation_config = simulation_config
        self.grid_config = grid_config
        self.current_time = start_time or datetime.fromtimestamp(0)

        # Asset management
        self.assets: dict[str, Asset] = {}

        # Network topology
        self.nodes: dict[str, NetworkNode] = {}
        self.transmission_lines: dict[str, TransmissionLine] = {}

        # Grid state tracking
        self._current_frequency = grid_config.frequency_hz
        self._total_generation = 0.0
        self._total_load = 0.0
        self._total_storage = 0.0
        self._grid_losses = 0.0

        logger.info(f"GridEngine initialized with {simulation_config.mode} mode")

    def add_asset(self, asset: Asset) -> None:
        """Add an asset to the simulation.

        Args:
            asset: Asset to add to the simulation

        Raises:
            ValueError: If asset with same ID already exists or max assets exceeded
        """
        if asset.asset_id in self.assets:
            raise ValueError(f"Asset with ID '{asset.asset_id}' already exists")

        if len(self.assets) >= self.simulation_config.max_assets:
            raise ValueError(f"Maximum number of assets ({self.simulation_config.max_assets}) exceeded")

        self.assets[asset.asset_id] = asset
        logger.debug(f"Added asset: {asset}")

    def add_node(self, node: NetworkNode) -> None:
        """Add a network node to the simulation.

        Args:
            node: Network node to add

        Raises:
            ValueError: If node with same ID already exists
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID '{node.node_id}' already exists")

        # Initialize current voltage to nominal if not set
        if node.current_voltage_kv == 0.0:
            node.current_voltage_kv = node.voltage_kv

        self.nodes[node.node_id] = node
        logger.debug(f"Added node: {node}")

    def add_transmission_line(self, line: TransmissionLine) -> None:
        """Add a transmission line to the simulation.

        Args:
            line: Transmission line to add

        Raises:
            ValueError: If line with same ID already exists or nodes don't exist
        """
        if line.line_id in self.transmission_lines:
            raise ValueError(f"Transmission line with ID '{line.line_id}' already exists")

        # Validate that both nodes exist
        if line.from_node not in self.nodes:
            raise ValueError(f"Node '{line.from_node}' not found")
        if line.to_node not in self.nodes:
            raise ValueError(f"Node '{line.to_node}' not found")

        # Validate that from_node and to_node are different
        if line.from_node == line.to_node:
            raise ValueError("From node and to node must be different")

        self.transmission_lines[line.line_id] = line
        logger.debug(f"Added transmission line: {line}")

    def step(self, dt: timedelta) -> None:
        """Advance the simulation by one time step.

        This method:
        1. Advances simulation time
        2. Calls tick() on all assets
        3. Calculates power flows
        4. Updates frequency and voltage
        5. Checks grid stability

        Args:
            dt: Time delta to advance the simulation

        Raises:
            ValueError: If dt is negative
        """
        if dt.total_seconds() < 0:
            raise ValueError("Time delta must be non-negative")

        # Advance simulation time
        self.current_time += dt

        # Call tick() on all assets
        dt_seconds = dt.total_seconds()
        for asset in self.assets.values():
            try:
                asset.tick(dt_seconds)
            except Exception as e:
                logger.warning(f"Error in asset {asset.asset_id} tick: {e}")

        # Calculate power flows
        self._calculate_power_flows()

        # Update grid frequency based on power balance
        self._update_frequency()

        # Update node voltages
        self._update_node_voltages()

        # Update transmission line flows
        self._update_transmission_flows()

        # Check grid stability
        self._check_grid_stability()

        logger.debug(f"Simulation step completed at {self.current_time}")

    def get_state(self) -> GridState:
        """Get the current state of the grid.

        Returns:
            Current grid state with all key parameters
        """
        return GridState(
            timestamp=self.current_time,
            frequency_hz=self._current_frequency,
            total_generation_mw=self._total_generation,
            total_load_mw=self._total_load,
            total_storage_mw=self._total_storage,
            grid_losses_mw=self._grid_losses,
        )

    def reset(self, start_time: datetime | None = None) -> None:
        """Reset the simulation to initial state.

        Args:
            start_time: Optional new start time (defaults to epoch)
        """
        self.current_time = start_time or datetime.fromtimestamp(0)
        self.assets.clear()
        self.nodes.clear()
        self.transmission_lines.clear()

        # Reset grid state
        self._current_frequency = self.grid_config.frequency_hz
        self._total_generation = 0.0
        self._total_load = 0.0
        self._total_storage = 0.0
        self._grid_losses = 0.0

        logger.info("GridEngine reset to initial state")

    def _calculate_power_flows(self) -> None:
        """Calculate total power flows from all assets."""
        self._total_generation = 0.0
        self._total_load = 0.0
        self._total_storage = 0.0

        for asset in self.assets.values():
            # Get power output (use current_output_mw directly for mocked assets)
            if hasattr(asset, "get_power_output"):
                power_output = asset.get_power_output()
            else:
                power_output = asset.current_output_mw

            if asset.asset_type == AssetType.LOAD:
                # Loads have negative power output
                self._total_load += abs(power_output)
            elif asset.asset_type in {AssetType.BATTERY, AssetType.HYDRO}:
                # Storage can charge (positive) or discharge (negative)
                self._total_storage += power_output
            else:
                # Generation assets (solar, wind, etc.)
                self._total_generation += max(0.0, power_output)

        # Calculate grid losses (simplified model: 2% of total generation)
        self._grid_losses = self._total_generation * 0.02

    def _update_frequency(self) -> None:
        """Update grid frequency based on power balance."""
        # Get current power balance
        power_balance = self._total_generation - self._total_load - self._total_storage - self._grid_losses

        # Simple frequency droop model: 1% frequency change per 100 MW imbalance
        frequency_deviation = (power_balance / 100.0) * 0.01 * self.grid_config.frequency_hz

        # Apply frequency deviation
        self._current_frequency = self.grid_config.frequency_hz + frequency_deviation

        # Clamp frequency to reasonable bounds (±5% of nominal)
        min_freq = self.grid_config.frequency_hz * 0.95
        max_freq = self.grid_config.frequency_hz * 1.05
        self._current_frequency = max(min_freq, min(max_freq, self._current_frequency))

    def _update_node_voltages(self) -> None:
        """Update voltage at each network node."""
        for node in self.nodes.values():
            # Update node frequency to match grid frequency
            node.current_frequency_hz = self._current_frequency

            # Simple voltage regulation (maintain nominal voltage for now)
            # In a more sophisticated model, this would depend on reactive power flows
            node.current_voltage_kv = node.voltage_kv

    def _update_transmission_flows(self) -> None:
        """Update power flows on transmission lines."""
        # Calculate net power injection at each node
        node_injections: dict[str, float] = defaultdict(float)

        for asset in self.assets.values():
            if asset.node_id in self.nodes:
                node_injections[asset.node_id] += asset.get_power_output()
            else:
                logger.warning(f"Asset {asset.asset_id} references unknown node {asset.node_id}")

        # Simple power flow calculation for transmission lines
        for line in self.transmission_lines.values():
            from_injection = node_injections.get(line.from_node, 0.0)
            to_injection = node_injections.get(line.to_node, 0.0)

            # Flow from high injection to low injection
            # This is a simplified model - real power flow depends on impedances
            net_flow = (from_injection - to_injection) / 2.0

            # Apply capacity constraints
            if abs(net_flow) > line.capacity_mw:
                net_flow = line.capacity_mw * (1.0 if net_flow > 0 else -1.0)

            line.current_flow_mw = net_flow

    def _check_grid_stability(self) -> None:
        """Check grid stability and apply corrective actions if needed."""
        # Check frequency stability
        frequency_deviation = abs(self._current_frequency - self.grid_config.frequency_hz)
        if frequency_deviation > self.grid_config.stability_threshold:
            logger.warning(f"Grid frequency deviation: {frequency_deviation:.3f} Hz")

            # Apply simple frequency regulation (move towards nominal)
            correction = (self.grid_config.frequency_hz - self._current_frequency) * 0.1
            self._current_frequency += correction

    def get_asset_by_id(self, asset_id: str) -> Asset | None:
        """Get asset by ID.

        Args:
            asset_id: Unique identifier of the asset

        Returns:
            Asset if found, None otherwise
        """
        return self.assets.get(asset_id)

    def get_assets_by_type(self, asset_type: AssetType) -> list[Asset]:
        """Get all assets of a specific type.

        Args:
            asset_type: Type of assets to retrieve

        Returns:
            List of assets of the specified type
        """
        return [asset for asset in self.assets.values() if asset.asset_type == asset_type]

    def get_assets_by_node(self, node_id: str) -> list[Asset]:
        """Get all assets connected to a specific node.

        Args:
            node_id: ID of the network node

        Returns:
            List of assets connected to the node
        """
        return [asset for asset in self.assets.values() if asset.node_id == node_id]

    def get_grid_summary(self) -> dict[str, Any]:
        """Get a summary of the current grid state.

        Returns:
            Dictionary with grid summary information
        """
        state = self.get_state()

        return {
            "simulation_time": self.current_time.isoformat(),
            "assets": {
                "total": len(self.assets),
                "by_type": {asset_type.value: len(self.get_assets_by_type(asset_type)) for asset_type in AssetType},
                "online": len([a for a in self.assets.values() if a.is_online()]),
            },
            "network": {
                "nodes": len(self.nodes),
                "transmission_lines": len(self.transmission_lines),
            },
            "power_flows": {
                "generation_mw": state.total_generation_mw,
                "load_mw": state.total_load_mw,
                "storage_mw": state.total_storage_mw,
                "losses_mw": state.grid_losses_mw,
                "balance_mw": state.power_balance_mw,
            },
            "grid_parameters": {
                "frequency_hz": state.frequency_hz,
                "is_balanced": state.is_balanced,
            },
        }

    def __str__(self) -> str:
        """String representation of the grid engine."""
        return (
            f"GridEngine(assets={len(self.assets)}, nodes={len(self.nodes)}, "
            f"lines={len(self.transmission_lines)}, time={self.current_time})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the grid engine."""
        return (
            f"GridEngine(assets={len(self.assets)}, nodes={len(self.nodes)}, "
            f"lines={len(self.transmission_lines)}, mode={self.simulation_config.mode}, "
            f"time={self.current_time})"
        )


__all__ = ["GridEngine", "GridState", "NetworkNode", "TransmissionLine"]
