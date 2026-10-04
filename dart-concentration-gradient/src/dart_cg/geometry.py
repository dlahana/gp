"""Tube geometry and probe position (SI internally)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .units import to_si


@dataclass(frozen=True)
class Tube:
    """Cylindrical tube. ``length`` and ``diameter`` in metres."""

    length: float
    diameter: float

    def __post_init__(self) -> None:
        if not (self.length > 0 and self.diameter > 0):
            raise ValueError("tube length and diameter must be positive")

    @classmethod
    def from_units(cls, length: float, diameter: float, length_unit: str = "in",
                   diameter_unit: str | None = None) -> "Tube":
        return cls(to_si(length, length_unit), to_si(diameter, diameter_unit or length_unit))

    @classmethod
    def default(cls) -> "Tube":
        """0.75 in inner diameter x 8 in long."""
        return cls.from_units(8.0, 0.75, "in")

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def area(self) -> float:
        return math.pi * self.radius**2

    @property
    def volume(self) -> float:
        return self.area * self.length


@dataclass(frozen=True)
class Probe:
    """Insect-position probe at axial position ``x`` (m from the x=0 end)."""

    x: float

    @classmethod
    def from_units(cls, x: float, unit: str = "in") -> "Probe":
        return cls(to_si(x, unit))

    def check(self, tube: Tube) -> None:
        if not 0.0 <= self.x <= tube.length:
            raise ValueError(f"probe x={self.x} m is outside the tube [0, {tube.length}] m")
