import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfc

from dart_cg.dimensionless import slowest_mode_time
from dart_cg.solvers.analytical import AnalyticalSolution

L, D = 0.2032, 5.84e-6
TD = L**2 / D
CONFIGS = {
    "left": dict(c_left=1.0),
    "right": dict(c_right=2.0),
    "both_equal": dict(c_left=1.0, c_right=1.0),
    "both_diff": dict(c_left=1.0, c_right=0.3),
    "left_sink": dict(c_left=0.0, c_init=1.0),
}
X = np.linspace(0, L, 41)


def make(name):
    return AnalyticalSolution(L, D, **CONFIGS[name])


@pytest.mark.parametrize("name", CONFIGS)
def test_series_matches_images(name):
    s = make(name)
    for tau in (0.004, 0.02, 0.1, 0.5, 2.0):
        t = tau * TD
        a = s.concentration(X, t, "series")
        b = s.concentration(X, t, "images")
        assert np.allclose(a, b, atol=1e-10), (name, tau, np.abs(a - b).max())


@pytest.mark.parametrize("name", CONFIGS)
def test_boundary_and_initial_conditions(name):
    s = make(name)
    t = 0.3 * TD
    if s.c_left is not None:
        assert s.concentration(0.0, t) == pytest.approx(s.c_left, abs=1e-10)
    else:
        assert s.gradient(0.0, t) == pytest.approx(0.0, abs=1e-9 / L)
    if s.c_right is not None:
        assert s.concentration(L, t) == pytest.approx(s.c_right, abs=1e-10)
    else:
        assert s.gradient(L, t) == pytest.approx(0.0, abs=1e-9 / L)
    assert np.allclose(s.concentration(X[5:-5], 0.0), s.c_init)
    assert np.allclose(s.concentration(X[5:-5], 1e-9 * TD), s.c_init, atol=1e-12)


@pytest.mark.parametrize("name", CONFIGS)
def test_steady_state(name):
    s = make(name)
    assert np.allclose(s.concentration(X, 30 * TD), s.steady_state(X), atol=1e-9)


@pytest.mark.parametrize("name", CONFIGS)
def test_satisfies_pde(name):
    s = make(name)
    x = np.linspace(0.1 * L, 0.9 * L, 9)
    for tau in (0.03, 0.2, 0.8):
        t = tau * TD
        ht, hx = 1e-4 * t, 1e-4 * L
        dcdt = (s.concentration(x, t + ht) - s.concentration(x, t - ht)) / (2 * ht)
        d2 = (s.concentration(x + hx, t) - 2 * s.concentration(x, t) + s.concentration(x - hx, t)) / hx**2
        assert np.allclose(dcdt, D * d2, rtol=1e-3, atol=1e-5 * abs(dcdt).max())


@pytest.mark.parametrize("name", CONFIGS)
def test_gradient_matches_finite_difference(name):
    s = make(name)
    x = np.linspace(0.05 * L, 0.95 * L, 11)
    for tau in (0.02, 0.5):
        t = tau * TD
        h = 1e-6 * L
        fd = (s.concentration(x + h, t) - s.concentration(x - h, t)) / (2 * h)
        assert np.allclose(s.gradient(x, t), fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("method", ["series", "images", "auto"])
def test_mass_matches_quadrature(name, method):
    s = make(name)
    xf = np.linspace(0, L, 40001)
    for tau in (0.01, 0.1, 1.0):
        t = tau * TD
        ref = np.trapezoid(s.concentration(xf, t), xf)
        assert s.mass_per_area(t, method) == pytest.approx(ref, rel=1e-6)


@pytest.mark.parametrize("name", ["left", "both_diff", "left_sink"])
def test_mass_balance_flux_integral(name):
    """d/dt integral(c) = J_left + J_right, and time-integrated flux = mass gained."""
    s = make(name)

    def J(t):
        return s.flux_into_tube(t, "left") + s.flux_into_tube(t, "right")

    t = 0.2 * TD
    h = 1e-5 * t
    dm = (s.mass_per_area(t + h) - s.mass_per_area(t - h)) / (2 * h)
    assert dm == pytest.approx(float(J(t)), rel=1e-6)
    T = 0.5 * TD
    val, _ = quad(lambda u: float(J(u * u)) * 2 * u, 0, np.sqrt(T), epsabs=0, epsrel=1e-10, limit=200)
    assert val == pytest.approx(float(s.mass_per_area(T) - s.mass_per_area(0.0)), rel=1e-7)


def test_symmetry_and_mirror():
    sym = make("both_equal")
    assert np.allclose(sym.concentration(X, 0.1 * TD), sym.concentration(L - X, 0.1 * TD))
    left = AnalyticalSolution(L, D, c_left=1.0)
    right = AnalyticalSolution(L, D, c_right=1.0)
    assert np.allclose(left.concentration(X, 0.1 * TD), right.concentration(L - X, 0.1 * TD))


def test_offset_identity():
    """Shifting every concentration by a constant shifts the solution by that constant."""
    a = AnalyticalSolution(L, D, c_left=3.0, c_right=1.0, c_init=0.0)
    b = AnalyticalSolution(L, D, c_left=2.0, c_right=0.0, c_init=-1.0)
    t = 0.07 * TD
    assert np.allclose(a.concentration(X, t), b.concentration(X, t) + 1.0, atol=1e-10)


def test_early_time_semi_infinite_limit():
    s = make("left")
    t = 1e-4 * TD
    x = np.linspace(0, 0.1 * L, 20)
    assert np.allclose(s.concentration(x, t), erfc(x / (2 * np.sqrt(D * t))), atol=1e-12)
    assert s.flux_into_tube(t) == pytest.approx(np.sqrt(D / (np.pi * t)), rel=1e-9)


def test_slowest_mode_decay_rate():
    s = make("left")
    t1, t2 = 1.0 * TD, 1.5 * TD
    e1, e2 = 1 - s.concentration(L, t1), 1 - s.concentration(L, t2)
    assert (t2 - t1) / np.log(e1 / e2) == pytest.approx(slowest_mode_time(L, D), rel=1e-4)


def test_no_sources_and_validation():
    s = AnalyticalSolution(L, D, c_init=0.5)
    assert np.allclose(s.concentration(X, 3 * TD), 0.5)
    assert np.all(s.flux_into_tube(np.array([1.0, 2.0])) == 0)
    with pytest.raises(ValueError):
        AnalyticalSolution(-1, D)
    with pytest.raises(ValueError):
        make("left").concentration(X, 1.0, method="bogus")


def test_broadcast_and_field_shapes():
    s = make("left")
    f = s.field(X, np.array([100.0, 1000.0, 5000.0]))
    assert f.shape == (3, X.size)
    assert np.allclose(f[1], s.concentration(X, 1000.0))
    assert s.probe_series(0.5 * L, np.array([10.0, 20.0])).shape == (2,)
    assert float(s.at(0.5 * L, 100.0)) == pytest.approx(float(f[0, 20]))
