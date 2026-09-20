

"""
Working-fluid and reference-condition constants used throughout the
parallel compressor model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GasProperties:
    """
    Thermodynamic properties of the working fluid, assumed calorically
    perfect (constant cp / gamma). Defaults correspond to dry air.
    """

    cp: float = 1004.5  # Specific heat at constant pressure [J/(kg*K)]
    gamma: float = 1.4  # Ratio of specific heats [-]
    R: float = 287.05  # Specific gas constant [J/(kg*K)]


@dataclass(frozen=True)
class ReferenceConditions:
    """Standard-day reference conditions used for speed/flow correction (theta, delta)."""

    p_ref: float = 101325.0  # Reference pressure [Pa]
    T_ref: float = 288.15  # Reference temperature [K]
