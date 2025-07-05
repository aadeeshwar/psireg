"""Base asset classes for PSIREG simulation system.

This module provides the abstract base class for all grid assets and defines
common asset types and interfaces.
"""

from abc import ABC

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
        self.current_output_mw = power_mw

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
