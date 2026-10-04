"""Dimensionless numbers and regime warnings."""
from __future__ import annotations

import math


def peclet(u: float, length: float, D: float) -> float:
    """Pe = u L / D."""
    return u * length / D


def diffusion_time(length: float, D: float) -> float:
    """t_D = L^2 / D  [s]."""
    return length**2 / D


def slowest_mode_time(length: float, D: float, both_ends: bool = False) -> float:
    """e-folding time of the slowest eigenmode (closed far end: 4L^2/(pi^2 D);
    sources at both ends: L^2/(pi^2 D))."""
    return (length**2 if both_ends else 4 * length**2) / (math.pi**2 * D)


def regime_warnings(u: float, length: float, D: float, dx: float | None = None) -> list[str]:
    """Human-readable warnings about the transport regime."""
    out: list[str] = []
    pe = abs(peclet(u, length, D))
    if pe > 10:
        out.append(f"Pe={pe:.3g} > 10: advection dominates; the diffusion-only picture is not valid.")
    elif pe > 0.1:
        out.append(f"Pe={pe:.3g} > 0.1: advection is not negligible compared with diffusion.")
    if dx is not None and abs(u) * dx / D > 2:
        out.append(f"cell Peclet number {abs(u) * dx / D:.3g} > 2: upwind numerical diffusion will be significant.")
    return out
