import math

import numpy as np
import pytest

from dart_cg import Probe, Tube
from dart_cg.dimensionless import diffusion_time, peclet, regime_warnings, slowest_mode_time
from dart_cg.units import convert, from_si, kg_m3_to_ppb, ppb_to_kg_m3, to_si


def test_length_and_default_tube():
    assert to_si(0.75, "in") == pytest.approx(0.01905)
    t = Tube.default()
    assert t.length == pytest.approx(0.2032) and t.diameter == pytest.approx(0.01905)
    assert t.area == pytest.approx(math.pi * 0.009525**2)


def test_concentration_units():
    assert to_si(1.0, "ng/L") == pytest.approx(1e-9)
    assert convert(1.0, "mg/L", "ug/m3") == pytest.approx(1e6)


def test_temperature():
    assert to_si(25.0, "C") == pytest.approx(298.15)
    assert from_si(373.15, "F") == pytest.approx(212.0)
    assert convert(32.0, "F", "C") == pytest.approx(0.0)


def test_array_and_errors():
    assert np.allclose(to_si(np.array([1.0, 2.0]), "cm"), [0.01, 0.02])
    with pytest.raises(ValueError):
        convert(1.0, "in", "Pa")
    with pytest.raises(ValueError):
        to_si(1.0, "furlong")


def test_ppb_roundtrip():
    c = ppb_to_kg_m3(50.0, 154.25)
    assert kg_m3_to_ppb(c, 154.25) == pytest.approx(50.0)


def test_tube_validation_and_probe():
    with pytest.raises(ValueError):
        Tube(-1.0, 0.01)
    with pytest.raises(ValueError):
        Probe(1.0).check(Tube.default())
    Probe.from_units(4, "in").check(Tube.default())


def test_dimensionless():
    L, D = 0.2032, 5.84e-6
    assert diffusion_time(L, D) == pytest.approx(7070, rel=1e-2)
    assert peclet(1e-3, L, D) == pytest.approx(34.8, rel=1e-2)
    assert slowest_mode_time(L, D) == pytest.approx(4 * L**2 / (math.pi**2 * D))
    assert slowest_mode_time(L, D, both_ends=True) == pytest.approx(slowest_mode_time(L, D) / 4)
    assert regime_warnings(0.0, L, D) == []
    assert "advection dominates" in regime_warnings(1e-3, L, D)[0]
    assert any("cell Peclet" in w for w in regime_warnings(1e-3, L, D, dx=2e-2))
