"""Pheromone Field Infrastructure for PSIREG Swarm Intelligence.

This module provides the core pheromone field infrastructure for distributed
grid coordination, including:
- Grid-based spatial pheromone storage
- Temporal decay and diffusion mechanisms
- Agent registration and communication
- Swarm coordination bus

The pheromone field serves as the primary communication medium for swarm
intelligence, enabling agents to coordinate through chemical-like signals
that decay over time and diffuse through space.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PheromoneType(Enum):
    """Types of pheromones for different coordination signals."""

    DEMAND_REDUCTION = "demand_reduction"
    FREQUENCY_SUPPORT = "frequency_support"
    ECONOMIC_SIGNAL = "economic_signal"
    COORDINATION = "coordination"
    EMERGENCY_RESPONSE = "emergency_response"
    RENEWABLE_CURTAILMENT = "renewable_curtailment"


@dataclass
class GridPosition:
    """Position within the pheromone grid."""

    x: int
    y: int

    def __eq__(self, other: object) -> bool:
        """Check equality with another GridPosition."""
        if not isinstance(other, GridPosition):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash((self.x, self.y))


class PheromoneField:
    """Grid-based pheromone field with decay and diffusion.

    This class manages a 2D grid of pheromone concentrations that:
    - Decay exponentially over time
    - Diffuse to neighboring cells
    - Support multiple pheromone types
    - Provide spatial queries and operations
    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        decay_rate: float = 0.99,
        diffusion_rate: float = 0.05,
        time_step_s: float = 1.0,
        max_pheromone_strength: float = 1.0,
    ):
        """Initialize pheromone field.

        Args:
            grid_width: Width of the pheromone grid
            grid_height: Height of the pheromone grid
            decay_rate: Pheromone decay rate per time step (0-1)
            diffusion_rate: Pheromone diffusion rate per time step (0-1)
            time_step_s: Time step duration in seconds
            max_pheromone_strength: Maximum pheromone strength (clamping)
        """
        # Validate parameters
        if grid_width <= 0 or grid_height <= 0:
            raise ValueError("Grid dimensions must be positive")
        if not (0.0 <= decay_rate <= 1.0):
            raise ValueError("Decay rate must be between 0 and 1")
        if not (0.0 <= diffusion_rate <= 1.0):
            raise ValueError("Diffusion rate must be between 0 and 1")
        if time_step_s <= 0:
            raise ValueError("Time step must be positive")

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.decay_rate = decay_rate
        self.diffusion_rate = diffusion_rate
        self.time_step_s = time_step_s
        self.max_pheromone_strength = max_pheromone_strength

        # Initialize pheromone grid: [width, height, num_pheromone_types]
        self.pheromone_grid = np.zeros((grid_width, grid_height, len(PheromoneType)), dtype=np.float64)

        # Time tracking
        self.current_time = 0.0

        # Create pheromone type to index mapping
        self._pheromone_type_indices = {pheromone_type: i for i, pheromone_type in enumerate(PheromoneType)}

        logger.debug(
            f"Initialized PheromoneField: {grid_width}x{grid_height}, "
            f"decay={decay_rate}, diffusion={diffusion_rate}"
        )

    def deposit_pheromone(
        self,
        position: GridPosition,
        pheromone_type: PheromoneType,
        strength: float,
    ) -> None:
        """Deposit pheromone at specified position.

        Args:
            position: Grid position to deposit pheromone
            pheromone_type: Type of pheromone to deposit
            strength: Strength of pheromone to deposit
        """
        if strength < 0:
            raise ValueError("Pheromone strength must be non-negative")

        if not self._validate_position(position):
            raise ValueError("Position out of bounds")

        pheromone_idx = self._pheromone_type_indices[pheromone_type]

        # Add to existing pheromone, clamping to maximum
        current_strength = self.pheromone_grid[position.x, position.y, pheromone_idx]
        new_strength = min(current_strength + strength, self.max_pheromone_strength)
        self.pheromone_grid[position.x, position.y, pheromone_idx] = new_strength

        logger.debug(f"Deposited {pheromone_type.value} pheromone: " f"{strength:.3f} at ({position.x}, {position.y})")

    def get_pheromone_strength(
        self,
        position: GridPosition,
        pheromone_type: PheromoneType,
    ) -> float:
        """Get pheromone strength at specified position.

        Args:
            position: Grid position to query
            pheromone_type: Type of pheromone to query

        Returns:
            Pheromone strength at position
        """
        if not self._validate_position(position):
            raise ValueError("Position out of bounds")

        pheromone_idx = self._pheromone_type_indices[pheromone_type]
        return float(self.pheromone_grid[position.x, position.y, pheromone_idx])

    def get_neighborhood_pheromones(
        self,
        center: GridPosition,
        pheromone_type: PheromoneType,
        radius: int = 1,
    ) -> list[tuple[GridPosition, float]]:
        """Get pheromone strengths in neighborhood around center position.

        Args:
            center: Center position for neighborhood
            pheromone_type: Type of pheromone to query
            radius: Neighborhood radius (Manhattan distance)

        Returns:
            List of (position, strength) tuples for neighborhood
        """
        if not self._validate_position(center):
            raise ValueError("Center position out of bounds")

        neighborhood = []
        pheromone_idx = self._pheromone_type_indices[pheromone_type]

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = center.x + dx
                y = center.y + dy

                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    position = GridPosition(x=x, y=y)
                    strength = float(self.pheromone_grid[x, y, pheromone_idx])
                    neighborhood.append((position, strength))

        return neighborhood

    def update_time_step(self) -> None:
        """Update pheromone field for one time step (decay and diffusion)."""
        # Apply decay
        self.pheromone_grid *= self.decay_rate

        # Apply diffusion
        if self.diffusion_rate > 0:
            self._apply_diffusion()

        # Update time
        self.current_time += self.time_step_s

        logger.debug(f"Updated pheromone field: t={self.current_time:.1f}s")

    def _apply_diffusion(self) -> None:
        """Apply diffusion to pheromone field."""
        # Apply diffusion to each pheromone type
        for pheromone_idx in range(len(PheromoneType)):
            pheromone_layer = self.pheromone_grid[:, :, pheromone_idx]

            # Calculate diffusion amount
            diffusion_amount = pheromone_layer * self.diffusion_rate

            # Apply diffusion using convolution-like operation
            diffused_layer = np.zeros_like(pheromone_layer)

            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    # Spread pheromone to neighbors
                    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

                    for nx, ny in neighbors:
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            diffused_layer[nx, ny] += diffusion_amount[x, y] / 4.0

            # Update pheromone layer (remove diffused amount, add received amount)
            self.pheromone_grid[:, :, pheromone_idx] = pheromone_layer - diffusion_amount + diffused_layer

    def get_total_pheromone(self, pheromone_type: PheromoneType) -> float:
        """Get total pheromone strength across entire grid.

        Args:
            pheromone_type: Type of pheromone to sum

        Returns:
            Total pheromone strength
        """
        pheromone_idx = self._pheromone_type_indices[pheromone_type]
        return float(np.sum(self.pheromone_grid[:, :, pheromone_idx]))

    def get_max_pheromone_position(
        self,
        pheromone_type: PheromoneType,
    ) -> tuple[GridPosition, float]:
        """Find position with maximum pheromone strength.

        Args:
            pheromone_type: Type of pheromone to find maximum for

        Returns:
            Tuple of (position, strength) for maximum pheromone
        """
        pheromone_idx = self._pheromone_type_indices[pheromone_type]
        pheromone_layer = self.pheromone_grid[:, :, pheromone_idx]

        # Find position of maximum value
        max_idx = np.unravel_index(np.argmax(pheromone_layer), pheromone_layer.shape)
        max_position = GridPosition(x=int(max_idx[0]), y=int(max_idx[1]))
        max_strength = float(pheromone_layer[max_idx])

        return max_position, max_strength

    def clear_pheromones(self, pheromone_type: PheromoneType | None = None) -> None:
        """Clear pheromones from field.

        Args:
            pheromone_type: Specific pheromone type to clear (None for all)
        """
        if pheromone_type is None:
            # Clear all pheromones
            self.pheromone_grid.fill(0.0)
        else:
            # Clear specific pheromone type
            pheromone_idx = self._pheromone_type_indices[pheromone_type]
            self.pheromone_grid[:, :, pheromone_idx] = 0.0

    def reset(self) -> None:
        """Reset pheromone field to initial state."""
        self.pheromone_grid.fill(0.0)
        self.current_time = 0.0
        logger.debug("Reset pheromone field")

    def _validate_position(self, position: GridPosition) -> bool:
        """Validate that position is within grid bounds.

        Args:
            position: Position to validate

        Returns:
            True if position is valid, False otherwise
        """
        return 0 <= position.x < self.grid_width and 0 <= position.y < self.grid_height


class SwarmBus:
    """Central coordination bus for swarm intelligence.

    This class manages the overall swarm coordination system, including:
    - Agent registration and spatial management
    - Pheromone field integration
    - Neighbor discovery and communication
    - Time synchronization
    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        pheromone_decay: float = 0.99,
        pheromone_diffusion: float = 0.05,
        time_step_s: float = 1.0,
        communication_range: float = 5.0,
        max_agents: int = 1000,
    ):
        """Initialize swarm bus.

        Args:
            grid_width: Width of the spatial grid
            grid_height: Height of the spatial grid
            pheromone_decay: Pheromone decay rate per time step
            pheromone_diffusion: Pheromone diffusion rate per time step
            time_step_s: Time step duration in seconds
            communication_range: Default communication range for agents
            max_agents: Maximum number of agents supported
        """
        # Validate parameters
        if grid_width <= 0 or grid_height <= 0:
            raise ValueError("Grid dimensions must be positive")
        if communication_range <= 0:
            raise ValueError("Communication range must be positive")
        if max_agents <= 0:
            raise ValueError("Max agents must be positive")

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.pheromone_decay = pheromone_decay
        self.pheromone_diffusion = pheromone_diffusion
        self.time_step_s = time_step_s
        self.communication_range = communication_range
        self.max_agents = max_agents

        # Initialize pheromone field
        self.pheromone_field = PheromoneField(
            grid_width=grid_width,
            grid_height=grid_height,
            decay_rate=pheromone_decay,
            diffusion_rate=pheromone_diffusion,
            time_step_s=time_step_s,
        )

        # Agent management
        self.registered_agents: dict[str, Any] = {}
        self.agent_positions: dict[str, GridPosition] = {}

        # Time tracking
        self.current_time = 0.0

        logger.info(f"Initialized SwarmBus: {grid_width}x{grid_height}, " f"max_agents={max_agents}")

    def register_agent(self, agent: Any, position: GridPosition) -> bool:
        """Register an agent in the swarm.

        Args:
            agent: Agent object to register
            position: Initial position of agent

        Returns:
            True if registration successful, False otherwise
        """
        # Check capacity
        if len(self.registered_agents) >= self.max_agents:
            logger.warning(f"Cannot register agent {agent.agent_id}: " f"maximum capacity ({self.max_agents}) reached")
            return False

        # Check for duplicate ID
        if agent.agent_id in self.registered_agents:
            logger.warning(f"Cannot register agent {agent.agent_id}: " f"agent ID already exists")
            return False

        # Validate position
        if not self._validate_position(position):
            raise ValueError("Agent position out of bounds")

        # Register agent
        self.registered_agents[agent.agent_id] = agent
        self.agent_positions[agent.agent_id] = position

        logger.debug(f"Registered agent {agent.agent_id} at ({position.x}, {position.y})")
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the swarm.

        Args:
            agent_id: ID of agent to unregister

        Returns:
            True if unregistration successful, False otherwise
        """
        if agent_id not in self.registered_agents:
            logger.warning(f"Cannot unregister agent {agent_id}: agent not found")
            return False

        del self.registered_agents[agent_id]
        del self.agent_positions[agent_id]

        logger.debug(f"Unregistered agent {agent_id}")
        return True

    def move_agent(self, agent_id: str, new_position: GridPosition) -> bool:
        """Move an agent to a new position.

        Args:
            agent_id: ID of agent to move
            new_position: New position for agent

        Returns:
            True if move successful, False otherwise
        """
        if agent_id not in self.registered_agents:
            logger.warning(f"Cannot move agent {agent_id}: agent not found")
            return False

        if not self._validate_position(new_position):
            raise ValueError("Agent position out of bounds")

        old_position = self.agent_positions[agent_id]
        self.agent_positions[agent_id] = new_position

        logger.debug(
            f"Moved agent {agent_id} from ({old_position.x}, {old_position.y}) "
            f"to ({new_position.x}, {new_position.y})"
        )
        return True

    def get_neighbors(
        self,
        agent_id: str,
        radius: float | None = None,
    ) -> list[Any]:
        """Get neighboring agents within communication range.

        Args:
            agent_id: ID of agent to find neighbors for
            radius: Communication radius (uses agent's default if None)

        Returns:
            List of neighboring agent objects
        """
        if agent_id not in self.registered_agents:
            logger.warning(f"Cannot get neighbors for agent {agent_id}: agent not found")
            return []

        agent = self.registered_agents[agent_id]
        agent_position = self.agent_positions[agent_id]

        # Use provided radius or agent's communication range
        if radius is None:
            radius = getattr(agent, "communication_range", self.communication_range)

        neighbors = []
        for other_id, other_agent in self.registered_agents.items():
            if other_id == agent_id:
                continue

            other_position = self.agent_positions[other_id]
            distance = self._calculate_distance(agent_position, other_position)

            if distance <= radius:
                neighbors.append(other_agent)

        return neighbors

    def get_agent_positions_in_radius(
        self,
        center: GridPosition,
        radius: float,
    ) -> list[tuple[GridPosition, str]]:
        """Get agent positions within radius of center position.

        Args:
            center: Center position for search
            radius: Search radius

        Returns:
            List of (position, agent_id) tuples within radius
        """
        nearby_agents = []

        for agent_id, position in self.agent_positions.items():
            distance = self._calculate_distance(center, position)
            if distance <= radius:
                nearby_agents.append((position, agent_id))

        return nearby_agents

    def deposit_pheromone(
        self,
        agent_id: str,
        pheromone_type: PheromoneType,
        strength: float,
    ) -> bool:
        """Deposit pheromone at agent's current position.

        Args:
            agent_id: ID of agent depositing pheromone
            pheromone_type: Type of pheromone to deposit
            strength: Strength of pheromone to deposit

        Returns:
            True if deposit successful, False otherwise
        """
        if agent_id not in self.registered_agents:
            logger.warning(f"Cannot deposit pheromone for agent {agent_id}: " f"agent not found")
            return False

        position = self.agent_positions[agent_id]
        self.pheromone_field.deposit_pheromone(position, pheromone_type, strength)

        logger.debug(f"Agent {agent_id} deposited {pheromone_type.value} " f"pheromone: {strength:.3f}")
        return True

    def get_pheromone_at_agent(
        self,
        agent_id: str,
        pheromone_type: PheromoneType,
    ) -> float:
        """Get pheromone strength at agent's current position.

        Args:
            agent_id: ID of agent
            pheromone_type: Type of pheromone to query

        Returns:
            Pheromone strength at agent's position (0.0 if agent not found)
        """
        if agent_id not in self.registered_agents:
            return 0.0

        position = self.agent_positions[agent_id]
        return self.pheromone_field.get_pheromone_strength(position, pheromone_type)

    def get_neighborhood_pheromones(
        self,
        agent_id: str,
        pheromone_type: PheromoneType,
        radius: int = 1,
    ) -> list[tuple[GridPosition, float]]:
        """Get pheromone field in agent's neighborhood.

        Args:
            agent_id: ID of agent
            pheromone_type: Type of pheromone to query
            radius: Neighborhood radius

        Returns:
            List of (position, strength) tuples for neighborhood
        """
        if agent_id not in self.registered_agents:
            return []

        position = self.agent_positions[agent_id]
        return self.pheromone_field.get_neighborhood_pheromones(position, pheromone_type, radius)

    def coordinate_agents(self, update_pheromones: bool = False) -> dict[str, list[Any]]:
        """Coordinate all registered agents.

        Args:
            update_pheromones: Whether to update pheromone field with agent signals

        Returns:
            Dictionary mapping agent_id to list of neighboring agents
        """
        coordination_results = {}

        for agent_id, agent in self.registered_agents.items():
            # Get neighbors
            neighbors = self.get_neighbors(agent_id)
            coordination_results[agent_id] = neighbors

            # Update agent with neighbor signals (if agent supports it)
            if hasattr(agent, "update_swarm_signals"):
                neighbor_signals = []
                for neighbor in neighbors:
                    if hasattr(neighbor, "get_coordination_signal"):
                        signal = neighbor.get_coordination_signal()
                        neighbor_signals.append(signal)

                agent.update_swarm_signals(neighbor_signals)

            # Update pheromone field (if requested)
            if update_pheromones and hasattr(agent, "get_pheromone_strength"):
                pheromone_strength = agent.get_pheromone_strength()
                if pheromone_strength > 0:
                    self.deposit_pheromone(agent_id, PheromoneType.COORDINATION, pheromone_strength)

        return coordination_results

    def update_time_step(self) -> None:
        """Update system for one time step."""
        # Update pheromone field
        self.pheromone_field.update_time_step()

        # Update system time
        self.current_time += self.time_step_s

        logger.debug(f"Updated SwarmBus time step: t={self.current_time:.1f}s")

    def get_agent_info(self, agent_id: str) -> dict[str, Any] | None:
        """Get information about a registered agent.

        Args:
            agent_id: ID of agent to get info for

        Returns:
            Dictionary with agent information, or None if not found
        """
        if agent_id not in self.registered_agents:
            return None

        agent = self.registered_agents[agent_id]
        position = self.agent_positions[agent_id]

        info = {
            "agent_id": agent_id,
            "position": position,
            "communication_range": getattr(agent, "communication_range", self.communication_range),
        }

        return info

    def get_system_stats(self) -> dict[str, Any]:
        """Get system statistics.

        Returns:
            Dictionary with system statistics
        """
        # Calculate pheromone totals
        pheromone_totals = {}
        for pheromone_type in PheromoneType:
            total = self.pheromone_field.get_total_pheromone(pheromone_type)
            pheromone_totals[pheromone_type.value] = total

        stats = {
            "total_agents": len(self.registered_agents),
            "grid_size": (self.grid_width, self.grid_height),
            "current_time": self.current_time,
            "pheromone_totals": pheromone_totals,
        }

        return stats

    def reset(self) -> None:
        """Reset the entire swarm system."""
        # Clear agents
        self.registered_agents.clear()
        self.agent_positions.clear()

        # Reset pheromone field
        self.pheromone_field.reset()

        # Reset time
        self.current_time = 0.0

        logger.info("Reset SwarmBus system")

    def _validate_position(self, position: GridPosition) -> bool:
        """Validate that position is within grid bounds.

        Args:
            position: Position to validate

        Returns:
            True if position is valid, False otherwise
        """
        return 0 <= position.x < self.grid_width and 0 <= position.y < self.grid_height

    def _calculate_distance(self, pos1: GridPosition, pos2: GridPosition) -> float:
        """Calculate Euclidean distance between two positions.

        Args:
            pos1: First position
            pos2: Second position

        Returns:
            Euclidean distance between positions
        """
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return math.sqrt(dx * dx + dy * dy)
