"""Tests for PheromoneField infrastructure.

This module contains comprehensive tests for pheromone field functionality including:
- Grid-based pheromone storage and retrieval
- Temporal decay mechanisms
- Spatial diffusion algorithms
- Agent position and neighborhood management
- Performance and edge case handling
"""

import pytest
from psireg.swarm.pheromone import GridPosition, PheromoneField, PheromoneType


class TestPheromoneFieldCreation:
    """Test PheromoneField creation and initialization."""

    def test_pheromone_field_creation(self):
        """Test pheromone field creation with basic parameters."""
        field = PheromoneField(
            grid_width=10,
            grid_height=10,
            decay_rate=0.95,
            diffusion_rate=0.1,
            time_step_s=1.0,
        )

        assert field.grid_width == 10
        assert field.grid_height == 10
        assert field.decay_rate == 0.95
        assert field.diffusion_rate == 0.1
        assert field.time_step_s == 1.0
        assert field.current_time == 0.0
        assert field.pheromone_grid.shape == (10, 10, len(PheromoneType))

    def test_pheromone_field_default_values(self):
        """Test pheromone field creation with default values."""
        field = PheromoneField(grid_width=5, grid_height=5)

        assert field.decay_rate == 0.99
        assert field.diffusion_rate == 0.05
        assert field.time_step_s == 1.0
        assert field.max_pheromone_strength == 1.0

    def test_pheromone_field_validation(self):
        """Test pheromone field parameter validation."""
        # Test negative dimensions
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            PheromoneField(grid_width=-1, grid_height=5)

        # Test invalid decay rate
        with pytest.raises(ValueError, match="Decay rate must be between 0 and 1"):
            PheromoneField(grid_width=5, grid_height=5, decay_rate=1.5)

        # Test invalid diffusion rate
        with pytest.raises(ValueError, match="Diffusion rate must be between 0 and 1"):
            PheromoneField(grid_width=5, grid_height=5, diffusion_rate=-0.1)

        # Test invalid time step
        with pytest.raises(ValueError, match="Time step must be positive"):
            PheromoneField(grid_width=5, grid_height=5, time_step_s=-1.0)


class TestPheromoneGridOperations:
    """Test pheromone grid storage and retrieval operations."""

    def test_deposit_pheromone(self):
        """Test depositing pheromone at specific positions."""
        field = PheromoneField(grid_width=5, grid_height=5)

        # Deposit pheromone at center
        position = GridPosition(x=2, y=2)
        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.8)

        # Check pheromone was deposited
        strength = field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert strength == 0.8

    def test_deposit_pheromone_multiple_types(self):
        """Test depositing multiple pheromone types at same position."""
        field = PheromoneField(grid_width=5, grid_height=5)
        position = GridPosition(x=1, y=1)

        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.6)
        field.deposit_pheromone(position, PheromoneType.FREQUENCY_SUPPORT, 0.4)

        assert field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION) == 0.6
        assert field.get_pheromone_strength(position, PheromoneType.FREQUENCY_SUPPORT) == 0.4

    def test_deposit_pheromone_accumulation(self):
        """Test pheromone accumulation at same position."""
        field = PheromoneField(grid_width=5, grid_height=5)
        position = GridPosition(x=2, y=2)

        # Deposit twice at same position
        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.3)
        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.4)

        # Should accumulate but not exceed max
        strength = field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert strength == 0.7

    def test_deposit_pheromone_max_clamp(self):
        """Test pheromone strength clamping at maximum."""
        field = PheromoneField(grid_width=5, grid_height=5, max_pheromone_strength=1.0)
        position = GridPosition(x=2, y=2)

        # Deposit beyond maximum
        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.8)
        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.5)

        # Should be clamped to maximum
        strength = field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert strength == 1.0

    def test_get_pheromone_strength_bounds_check(self):
        """Test bounds checking for pheromone strength retrieval."""
        field = PheromoneField(grid_width=5, grid_height=5)

        # Test out of bounds positions
        with pytest.raises(ValueError, match="Position out of bounds"):
            field.get_pheromone_strength(GridPosition(x=-1, y=2), PheromoneType.DEMAND_REDUCTION)

        with pytest.raises(ValueError, match="Position out of bounds"):
            field.get_pheromone_strength(GridPosition(x=5, y=2), PheromoneType.DEMAND_REDUCTION)

    def test_get_neighborhood_pheromones(self):
        """Test getting pheromone strengths in neighborhood."""
        field = PheromoneField(grid_width=5, grid_height=5)
        center = GridPosition(x=2, y=2)

        # Deposit pheromones in neighborhood
        field.deposit_pheromone(GridPosition(x=1, y=1), PheromoneType.DEMAND_REDUCTION, 0.5)
        field.deposit_pheromone(GridPosition(x=2, y=1), PheromoneType.DEMAND_REDUCTION, 0.6)
        field.deposit_pheromone(GridPosition(x=3, y=1), PheromoneType.DEMAND_REDUCTION, 0.7)
        field.deposit_pheromone(GridPosition(x=2, y=2), PheromoneType.DEMAND_REDUCTION, 0.8)

        # Get neighborhood pheromones
        neighborhood = field.get_neighborhood_pheromones(center, PheromoneType.DEMAND_REDUCTION, radius=1)

        assert len(neighborhood) == 9  # 3x3 neighborhood
        assert any(strength == 0.8 for pos, strength in neighborhood)  # Center position
        assert any(strength == 0.6 for pos, strength in neighborhood)  # Neighbor position


class TestPheromoneDecay:
    """Test pheromone decay mechanisms."""

    def test_decay_single_step(self):
        """Test pheromone decay after single time step."""
        field = PheromoneField(grid_width=5, grid_height=5, decay_rate=0.9, diffusion_rate=0.0)
        position = GridPosition(x=2, y=2)

        # Deposit pheromone
        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 1.0)

        # Advance time step
        field.update_time_step()

        # Check decay
        strength = field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert abs(strength - 0.9) < 1e-6

    def test_decay_multiple_steps(self):
        """Test pheromone decay over multiple time steps."""
        field = PheromoneField(grid_width=5, grid_height=5, decay_rate=0.8, diffusion_rate=0.0)
        position = GridPosition(x=2, y=2)

        # Deposit pheromone
        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 1.0)

        # Advance multiple time steps
        for _ in range(3):
            field.update_time_step()

        # Check exponential decay: 1.0 * 0.8^3 = 0.512
        strength = field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert abs(strength - 0.512) < 1e-6

    def test_decay_all_pheromone_types(self):
        """Test that decay affects all pheromone types."""
        field = PheromoneField(grid_width=5, grid_height=5, decay_rate=0.9, diffusion_rate=0.0)
        position = GridPosition(x=2, y=2)

        # Deposit multiple types
        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.8)
        field.deposit_pheromone(position, PheromoneType.FREQUENCY_SUPPORT, 0.6)
        field.deposit_pheromone(position, PheromoneType.ECONOMIC_SIGNAL, 0.4)

        # Advance time step
        field.update_time_step()

        # Check all types decayed
        assert abs(field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION) - 0.72) < 1e-6
        assert abs(field.get_pheromone_strength(position, PheromoneType.FREQUENCY_SUPPORT) - 0.54) < 1e-6
        assert abs(field.get_pheromone_strength(position, PheromoneType.ECONOMIC_SIGNAL) - 0.36) < 1e-6


class TestPheromoneeDiffusion:
    """Test pheromone diffusion mechanisms."""

    def test_diffusion_single_step(self):
        """Test pheromone diffusion after single time step."""
        field = PheromoneField(grid_width=5, grid_height=5, diffusion_rate=0.2, decay_rate=1.0)
        center = GridPosition(x=2, y=2)

        # Deposit pheromone at center
        field.deposit_pheromone(center, PheromoneType.DEMAND_REDUCTION, 1.0)

        # Advance time step for diffusion
        field.update_time_step()

        # Check that neighbors have some pheromone
        neighbors = field.get_neighborhood_pheromones(center, PheromoneType.DEMAND_REDUCTION, radius=1)
        neighbor_strengths = [strength for pos, strength in neighbors if pos != center]

        # Some neighbors should have received pheromone
        assert any(strength > 0.0 for strength in neighbor_strengths)

        # Center should have less pheromone (diffused out)
        center_strength = field.get_pheromone_strength(center, PheromoneType.DEMAND_REDUCTION)
        assert center_strength < 1.0

    def test_diffusion_conservation(self):
        """Test that diffusion conserves total pheromone (ignoring decay)."""
        field = PheromoneField(grid_width=5, grid_height=5, diffusion_rate=0.1, decay_rate=1.0)
        center = GridPosition(x=2, y=2)

        # Deposit pheromone
        initial_total = 1.0
        field.deposit_pheromone(center, PheromoneType.DEMAND_REDUCTION, initial_total)

        # Calculate total before diffusion
        total_before = field.get_total_pheromone(PheromoneType.DEMAND_REDUCTION)

        # Advance time step for diffusion
        field.update_time_step()

        # Calculate total after diffusion
        total_after = field.get_total_pheromone(PheromoneType.DEMAND_REDUCTION)

        # Should be approximately conserved (within small numerical error)
        assert abs(total_before - total_after) < 1e-6

    def test_diffusion_boundary_conditions(self):
        """Test diffusion behavior at grid boundaries."""
        field = PheromoneField(grid_width=3, grid_height=3, diffusion_rate=0.3, decay_rate=1.0)
        corner = GridPosition(x=0, y=0)

        # Deposit pheromone at corner
        field.deposit_pheromone(corner, PheromoneType.DEMAND_REDUCTION, 1.0)

        # Advance time step
        field.update_time_step()

        # Check that diffusion handled boundary correctly
        corner_strength = field.get_pheromone_strength(corner, PheromoneType.DEMAND_REDUCTION)
        assert corner_strength < 1.0  # Should have diffused

        # Check adjacent cells
        right_strength = field.get_pheromone_strength(GridPosition(x=1, y=0), PheromoneType.DEMAND_REDUCTION)
        down_strength = field.get_pheromone_strength(GridPosition(x=0, y=1), PheromoneType.DEMAND_REDUCTION)

        assert right_strength > 0.0
        assert down_strength > 0.0


class TestPheromoneFieldUtilities:
    """Test utility functions of PheromoneField."""

    def test_get_total_pheromone(self):
        """Test getting total pheromone across grid."""
        field = PheromoneField(grid_width=3, grid_height=3)

        # Deposit pheromones at different positions
        field.deposit_pheromone(GridPosition(x=0, y=0), PheromoneType.DEMAND_REDUCTION, 0.3)
        field.deposit_pheromone(GridPosition(x=1, y=1), PheromoneType.DEMAND_REDUCTION, 0.5)
        field.deposit_pheromone(GridPosition(x=2, y=2), PheromoneType.DEMAND_REDUCTION, 0.2)

        total = field.get_total_pheromone(PheromoneType.DEMAND_REDUCTION)
        assert abs(total - 1.0) < 1e-6

    def test_get_max_pheromone_position(self):
        """Test finding position with maximum pheromone."""
        field = PheromoneField(grid_width=3, grid_height=3)

        # Deposit pheromones at different positions
        field.deposit_pheromone(GridPosition(x=0, y=0), PheromoneType.DEMAND_REDUCTION, 0.3)
        field.deposit_pheromone(GridPosition(x=1, y=1), PheromoneType.DEMAND_REDUCTION, 0.8)
        field.deposit_pheromone(GridPosition(x=2, y=2), PheromoneType.DEMAND_REDUCTION, 0.5)

        max_pos, max_strength = field.get_max_pheromone_position(PheromoneType.DEMAND_REDUCTION)
        assert max_pos == GridPosition(x=1, y=1)
        assert max_strength == 0.8

    def test_clear_pheromones(self):
        """Test clearing pheromones from grid."""
        field = PheromoneField(grid_width=3, grid_height=3)

        # Deposit pheromones
        field.deposit_pheromone(GridPosition(x=1, y=1), PheromoneType.DEMAND_REDUCTION, 0.5)
        field.deposit_pheromone(GridPosition(x=1, y=1), PheromoneType.FREQUENCY_SUPPORT, 0.3)

        # Clear specific type
        field.clear_pheromones(PheromoneType.DEMAND_REDUCTION)

        assert field.get_pheromone_strength(GridPosition(x=1, y=1), PheromoneType.DEMAND_REDUCTION) == 0.0
        assert field.get_pheromone_strength(GridPosition(x=1, y=1), PheromoneType.FREQUENCY_SUPPORT) == 0.3

        # Clear all types
        field.clear_pheromones()
        assert field.get_pheromone_strength(GridPosition(x=1, y=1), PheromoneType.FREQUENCY_SUPPORT) == 0.0

    def test_reset_field(self):
        """Test resetting field to initial state."""
        field = PheromoneField(grid_width=3, grid_height=3)

        # Deposit pheromones and advance time
        field.deposit_pheromone(GridPosition(x=1, y=1), PheromoneType.DEMAND_REDUCTION, 0.5)
        field.update_time_step()
        field.update_time_step()

        # Reset field
        field.reset()

        assert field.current_time == 0.0
        assert field.get_total_pheromone(PheromoneType.DEMAND_REDUCTION) == 0.0


class TestPheromoneFieldPerformance:
    """Test performance aspects of PheromoneField."""

    def test_large_grid_performance(self):
        """Test performance with large grid sizes."""
        field = PheromoneField(grid_width=100, grid_height=100)

        # Deposit pheromones across grid
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                field.deposit_pheromone(GridPosition(x=i, y=j), PheromoneType.DEMAND_REDUCTION, 0.5)

        # Multiple time steps should complete in reasonable time
        for _ in range(10):
            field.update_time_step()

        # Basic functionality should still work
        total = field.get_total_pheromone(PheromoneType.DEMAND_REDUCTION)
        assert total > 0.0

    def test_memory_usage(self):
        """Test memory usage with different grid sizes."""
        # Small grid
        small_field = PheromoneField(grid_width=5, grid_height=5)
        small_size = small_field.pheromone_grid.nbytes

        # Larger grid
        large_field = PheromoneField(grid_width=50, grid_height=50)
        large_size = large_field.pheromone_grid.nbytes

        # Memory should scale appropriately
        expected_ratio = (50 * 50) / (5 * 5)
        actual_ratio = large_size / small_size
        assert abs(actual_ratio - expected_ratio) < 0.1


class TestPheromoneFieldEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_pheromone_deposit(self):
        """Test depositing zero pheromone."""
        field = PheromoneField(grid_width=5, grid_height=5)
        position = GridPosition(x=2, y=2)

        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.0)
        strength = field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert strength == 0.0

    def test_negative_pheromone_deposit(self):
        """Test handling negative pheromone values."""
        field = PheromoneField(grid_width=5, grid_height=5)
        position = GridPosition(x=2, y=2)

        with pytest.raises(ValueError, match="Pheromone strength must be non-negative"):
            field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, -0.5)

    def test_single_cell_grid(self):
        """Test functionality with single cell grid."""
        field = PheromoneField(grid_width=1, grid_height=1, diffusion_rate=0.0)
        position = GridPosition(x=0, y=0)

        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 0.5)

        # Should handle diffusion without error
        field.update_time_step()

        # Pheromone should only decay (no diffusion possible)
        strength = field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert strength == 0.5 * field.decay_rate

    def test_no_decay_no_diffusion(self):
        """Test field with no decay and no diffusion."""
        field = PheromoneField(grid_width=5, grid_height=5, decay_rate=1.0, diffusion_rate=0.0)
        position = GridPosition(x=2, y=2)

        initial_strength = 0.7
        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, initial_strength)

        # Advance multiple time steps
        for _ in range(5):
            field.update_time_step()

        # Strength should remain unchanged
        strength = field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert abs(strength - initial_strength) < 1e-6

    def test_very_high_decay_rate(self):
        """Test field with very high decay rate."""
        field = PheromoneField(grid_width=5, grid_height=5, decay_rate=0.1, diffusion_rate=0.0)
        position = GridPosition(x=2, y=2)

        field.deposit_pheromone(position, PheromoneType.DEMAND_REDUCTION, 1.0)

        # After several steps, should be nearly zero
        for _ in range(10):
            field.update_time_step()

        strength = field.get_pheromone_strength(position, PheromoneType.DEMAND_REDUCTION)
        assert strength < 0.01

    def test_concurrent_operations(self):
        """Test concurrent deposits and updates."""
        field = PheromoneField(grid_width=5, grid_height=5)

        # Deposit pheromones at multiple positions
        positions = [
            GridPosition(x=1, y=1),
            GridPosition(x=2, y=2),
            GridPosition(x=3, y=3),
        ]

        for pos in positions:
            field.deposit_pheromone(pos, PheromoneType.DEMAND_REDUCTION, 0.5)
            field.deposit_pheromone(pos, PheromoneType.FREQUENCY_SUPPORT, 0.3)

        # Update time step
        field.update_time_step()

        # All positions should have pheromones
        for pos in positions:
            demand_strength = field.get_pheromone_strength(pos, PheromoneType.DEMAND_REDUCTION)
            freq_strength = field.get_pheromone_strength(pos, PheromoneType.FREQUENCY_SUPPORT)
            assert demand_strength > 0.0
            assert freq_strength > 0.0
