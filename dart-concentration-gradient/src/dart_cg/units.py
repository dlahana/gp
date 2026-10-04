"""Unit conversion at the edges; everything inside the library is SI.

SI base for each kind: length m, time s, pressure Pa, concentration kg/m^3,
velocity m/s, mass kg, volume m^3, temperature K.
"""
from __future__ import annotations

import numpy as np

R_GAS = 8.314462618  # J/(mol K)

_TABLES: dict[str, dict[str, float]] = {
    "length": {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "in": 0.0254, "ft": 0.3048},
    "time": {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0, "hr": 3600.0, "day": 86400.0},
    "pressure": {"Pa": 1.0, "kPa": 1e3, "bar": 1e5, "atm": 101325.0,
                 "Torr": 101325.0 / 760.0, "mmHg": 133.322387415},
    "concentration": {"kg/m3": 1.0, "g/m3": 1e-3, "mg/m3": 1e-6, "ug/m3": 1e-9, "ng/m3": 1e-12,
                      "g/L": 1.0, "mg/L": 1e-3, "ug/L": 1e-6, "ng/L": 1e-9},
    "velocity": {"m/s": 1.0, "cm/s": 1e-2, "mm/s": 1e-3, "in/s": 0.0254},
    "mass": {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "ug": 1e-9, "ng": 1e-12},
    "volume": {"m3": 1.0, "L": 1e-3, "mL": 1e-6, "uL": 1e-9},
}
_TEMPS = ("K", "C", "F")


def _kind(unit: str) -> str:
    for kind, table in _TABLES.items():
        if unit in table:
            return kind
    raise ValueError(f"unknown unit {unit!r}")


def to_si(value, unit: str):
    """Convert ``value`` given in ``unit`` to SI (arrays are fine)."""
    if unit in _TEMPS:
        v = np.asarray(value, dtype=float)
        return v if unit == "K" else (v + 273.15 if unit == "C" else (v - 32.0) * 5.0 / 9.0 + 273.15)
    return value * _TABLES[_kind(unit)][unit]


def from_si(value, unit: str):
    """Convert an SI ``value`` into ``unit``."""
    if unit in _TEMPS:
        v = np.asarray(value, dtype=float)
        return v if unit == "K" else (v - 273.15 if unit == "C" else (v - 273.15) * 9.0 / 5.0 + 32.0)
    return value / _TABLES[_kind(unit)][unit]


def convert(value, src: str, dst: str):
    """Convert between two units of the same kind."""
    if (src in _TEMPS) != (dst in _TEMPS) or (src not in _TEMPS and _kind(src) != _kind(dst)):
        raise ValueError(f"cannot convert {src!r} to {dst!r}")
    return from_si(to_si(value, src), dst)


def ppb_to_kg_m3(ppb, mw_g_mol: float, T_K: float = 298.15, P_Pa: float = 101325.0):
    """Mole-fraction ppb (ideal gas) -> mass concentration in kg/m^3."""
    return ppb * 1e-9 * P_Pa * (mw_g_mol * 1e-3) / (R_GAS * T_K)


def kg_m3_to_ppb(c, mw_g_mol: float, T_K: float = 298.15, P_Pa: float = 101325.0):
    return c * R_GAS * T_K / (P_Pa * mw_g_mol * 1e-3) * 1e9
