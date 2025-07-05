"""Base asset classes for PSIREG simulation system.

This module provides the abstract base class for all grid assets and defines
common asset types and interfaces.
"""

from abc import ABC
from typing import Any

from pydantic import BaseModel, Field, field_validator

from psireg.utils.enums import AssetStatus, AssetType
from psireg.utils.types import MW


class Asset(BaseModel, ABC):
    """Base class for all grid assets.

    This abstract base class defines the common interface and properties
    for all types of grid assets (generation, load, storage, etc.).
    """

    asset_id: str = Field(..., min_length=1, description="Unique identifier for the asset")
    asset_type: AssetType = Field(..., description="Type of the asset")
    name: str = Field(..., min_length=1, description="Human-readable name of the asset")
    node_id: str = Field(..., min_length=1, description="ID of the network node where asset is connected")
    capacity_mw: MW = Field(..., gt=0.0, description="Maximum capacity of the asset in MW")
    status: AssetStatus = Field(default=AssetStatus.OFFLINE, description="Current operational status")
    current_output_mw: MW = Field(default=0.0, description="Current power output in MW (negative for loads)")

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, v: str) -> str:
        """Validate asset ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Asset ID cannot be empty")
        return v.strip()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate asset name is not empty."""
        if not v or not v.strip():
            raise ValueError("Asset name cannot be empty")
        return v.strip()

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        """Validate node ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Node ID cannot be empty")
        return v.strip()

    @field_validator("capacity_mw")
    @classmethod
    def validate_capacity(cls, v: float) -> float:
        """Validate capacity is positive."""
        if v <= 0:
            raise ValueError("Capacity must be positive")
        return v

    def set_status(self, status: AssetStatus) -> None:
        """Set the operational status of the asset.

        Args:
            status: New operational status
        """
        self.status = status

    def tick(self, dt_seconds: float) -> float | None:
        """Update asset state for one simulation timestep.

        This method is called by the simulation engine for each timestep.
        Subclasses should implement their specific logic here.

        Args:
            dt_seconds: Time delta in seconds since last tick

        Returns:
            Optional power output change in MW
        """
        # Default implementation does nothing
        return None

    def get_power_output(self) -> MW:
        """Get current power output.

        Returns:
            Current power output in MW (negative for loads)
        """
        return self.current_output_mw

    def set_power_output(self, power_mw: MW) -> None:
        """Set current power output.

        Args:
            power_mw: Power output in MW (negative for loads)
        """
        # Clamp power output to capacity limits for generation assets
        if power_mw > 0 and not self.is_load():
            power_mw = min(power_mw, self.capacity_mw)
        elif power_mw < 0 and self.is_load():
            # For loads, limit negative power to capacity
            power_mw = max(power_mw, -self.capacity_mw)

        self.current_output_mw = power_mw

    def get_state(self) -> dict[str, Any]:
        """Get comprehensive state information for the asset.

        This method provides a uniform interface for accessing asset state
        across all asset types.

        Returns:
            Dictionary containing complete asset state
        """
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "name": self.name,
            "node_id": self.node_id,
            "capacity_mw": self.capacity_mw,
            "status": self.status,
            "current_output_mw": self.current_output_mw,
            "is_online": self.is_online(),
            "is_renewable": self.is_renewable(),
            "is_storage": self.is_storage(),
            "is_load": self.is_load(),
            "utilization_percent": self.get_utilization_percent(),
            "efficiency": self.get_efficiency(),
        }

    def get_utilization_percent(self) -> float:
        """Calculate asset utilization as percentage of capacity.

        Returns:
            Utilization percentage (0-100)
        """
        if self.capacity_mw == 0:
            return 0.0

        # Use absolute value to handle both generation and load assets
        return (abs(self.current_output_mw) / self.capacity_mw) * 100.0

    def get_efficiency(self) -> float:
        """Get asset efficiency factor.

        Base implementation returns 1.0 (100% efficiency).
        Subclasses can override for more realistic efficiency models.

        Returns:
            Efficiency factor (0.0 to 1.0)
        """
        return 1.0

    def reset(self) -> None:
        """Reset asset to default state.

        Resets status to OFFLINE and power output to 0.
        """
        self.status = AssetStatus.OFFLINE
        self.current_output_mw = 0.0

    def is_online(self) -> bool:
        """Check if asset is online and operational.

        Returns:
            True if asset is online
        """
        return self.status == AssetStatus.ONLINE

    def is_renewable(self) -> bool:
        """Check if asset is a renewable energy source.

        Returns:
            True if asset is renewable (solar, wind, etc.)
        """
        return self.asset_type in {AssetType.SOLAR, AssetType.WIND, AssetType.HYDRO}

    def is_storage(self) -> bool:
        """Check if asset is an energy storage system.

        Returns:
            True if asset is storage (battery, etc.)
        """
        return self.asset_type == AssetType.BATTERY

    def is_load(self) -> bool:
        """Check if asset is a load.

        Returns:
            True if asset is a load
        """
        return self.asset_type == AssetType.LOAD

    def __str__(self) -> str:
        """String representation of the asset."""
        return f"{self.name} ({self.asset_type.value}, {self.capacity_mw:.1f} MW)"

    def __repr__(self) -> str:
        """Detailed string representation of the asset."""
        return (
            f"Asset(id={self.asset_id}, type={self.asset_type.value}, "
            f"name={self.name}, capacity={self.capacity_mw:.1f} MW, "
            f"status={self.status.value})"
        )


__all__ = ["Asset", "AssetType"]
