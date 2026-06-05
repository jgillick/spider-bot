"""
Utility functions for CAN bus motor control.
"""

from .protocol import TWO_PI


def rad_to_rev(rad: float) -> float:
    """Convert radians to revolutions (for vel_ff / protocol fields that use rev/s)."""
    return rad / TWO_PI


def make_can_id(cmd_id: int, node_id: int) -> int:
    """Create CAN arbitration ID from command and node ID."""
    return (node_id << 5) | cmd_id
