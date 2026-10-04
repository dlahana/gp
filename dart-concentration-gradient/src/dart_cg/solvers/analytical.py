"""Analytical 1D diffusion in a finite tube with no-flux walls.

Solves  dc/dt = D d2c/dx2  on 0 <= x <= L with uniform initial concentration
``c_init`` and, at each end, either

* a constant-concentration (Dirichlet) source ``c_end`` (a float), or
* a closed no-flux wall (``None``).

Two equivalent representations are implemented and cross-checked in the tests:

* **eigenseries** (fast at late times, many terms at early times), and
* **method of images** with complementary error functions (fast at early times).

``method="auto"`` picks images for tau = D t / L^2 < ``tau_switch`` and the series
otherwise.  Unit solutions (zero initial state, unit source at x=0):

* ``cl`` : source at x=0, closed wall at x=L
* ``dd`` : source at x=0, zero-concentration Dirichlet at x=L

and the general solution is built by superposition (the problem is linear).
Everything is SI: metres, seconds, kg/m^3 (or any consistent concentration unit).
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.special import erfc

Method = Literal["auto", "series", "images"]
_SQRT_PI = math.sqrt(math.pi)
_MAX_TERMS = 50_000


def _ierfc(z):
    """First repeated integral of erfc: ierfc(z) = exp(-z^2)/sqrt(pi) - z erfc(z)."""
    return np.exp(-z * z) / _SQRT_PI - z * erfc(z)


class AnalyticalSolution:
    """Closed-form solution for sources at one or both ends of a closed tube.

    Parameters
    ----------
    length : tube length L [m]
    diffusivity : D [m^2/s]
    c_left, c_right : source concentration at x=0 / x=L, or ``None`` for a closed wall
    c_init : uniform initial concentration
    tau_switch : D t/L^2 below which ``method='auto'`` uses the image sum
    """

    def __init__(self, length: float, diffusivity: float, c_left: float | None = None,
                 c_right: float | None = None, c_init: float = 0.0, tau_switch: float = 0.1) -> None:
        if length <= 0 or diffusivity <= 0:
            raise ValueError("length and diffusivity must be positive")
        self.L = float(length)
        self.D = float(diffusivity)
        self.c_left = c_left
        self.c_right = c_right
        self.c_init = float(c_init)
        self.tau_switch = tau_switch
        self._kind = "dd" if (c_left is not None and c_right is not None) else "cl"

    @classmethod
    def for_tube(cls, tube, diffusivity: float, **kw) -> "AnalyticalSolution":
        return cls(tube.length, diffusivity, **kw)

    # ------------------------------------------------------------------ helpers
    def _split(self, t: np.ndarray, method: Method):
        tau = self.D * t / self.L**2
        pos = t > 0
        if method == "images":
            img = pos
        elif method == "series":
            img = np.zeros_like(pos)
        elif method == "auto":
            img = pos & (tau < self.tau_switch)
        else:
            raise ValueError(f"unknown method {method!r}")
        return img, pos & ~img, tau

    def _images(self, kind: str, x, t):
        L, s = self.L, 2.0 * np.sqrt(self.D * t)
        tau_max = float((self.D * t / L**2).max())
        K = int(math.ceil(6.0 * math.sqrt(tau_max))) + 2
        u = np.zeros_like(x)
        ux = np.zeros_like(x)
        for k in range(K):
            a = (2 * k * L + x) / s
            b = (2 * (k + 1) * L - x) / s
            ea, eb = np.exp(-a * a), np.exp(-b * b)
            if kind == "cl":
                sg = -1.0 if k % 2 else 1.0
                u += sg * (erfc(a) + erfc(b))
                ux += sg * (-2.0 / (s * _SQRT_PI)) * (ea - eb)
            else:
                u += erfc(a) - erfc(b)
                ux += (-2.0 / (s * _SQRT_PI)) * (ea + eb)
        return u, ux

    def _n_terms(self, t) -> int:
        tau_min = float((self.D * t / self.L**2).min())
        n = int(math.ceil(math.sqrt(40.0 / tau_min) / math.pi)) + 2
        if n > _MAX_TERMS:
            raise ValueError("eigenseries needs too many terms at this t; use method='images'")
        return n

    def _series(self, kind: str, x, t):
        L, D = self.L, self.D
        N = self._n_terms(t)
        if kind == "cl":
            u = np.ones_like(x)
            ux = np.zeros_like(x)
            for n in range(N):
                lam = (2 * n + 1) * math.pi / (2 * L)
                e = np.exp(-D * lam * lam * t)
                u -= (4.0 / math.pi) * np.sin(lam * x) * e / (2 * n + 1)
                ux -= (2.0 / L) * np.cos(lam * x) * e
        else:
            u = 1.0 - x / L
            ux = np.full_like(x, -1.0 / L)
            for n in range(1, N + 1):
                k = n * math.pi / L
                e = np.exp(-D * k * k * t)
                u = u - (2.0 / (n * math.pi)) * np.sin(k * x) * e
                ux = ux - (2.0 / L) * np.cos(k * x) * e
        return u, ux

    def _unit(self, x, t, method: Method, kind: str | None = None):
        kind = kind or self._kind
        u = np.zeros(x.shape)
        ux = np.zeros(x.shape)
        img, ser, _ = self._split(t, method)
        if img.any():
            u[img], ux[img] = self._images(kind, x[img], t[img])
        if ser.any():
            u[ser], ux[ser] = self._series(kind, x[ser], t[ser])
        return u, ux

    def _mass_unit(self, t, method: Method):
        """integral over [0, L] of the unit solution, per unit area."""
        L, D = self.L, self.D
        out = np.zeros(t.shape)
        img, ser, _ = self._split(t, method)
        if img.any():
            ti = t[img]
            s = 2.0 * np.sqrt(D * ti)
            K = int(math.ceil(6.0 * math.sqrt(float((D * ti / L**2).max())))) + 2
            m = np.zeros_like(ti)
            for k in range(K):
                z0, z1, z2 = 2 * k * L / s, (2 * k + 1) * L / s, (2 * k + 2) * L / s
                if self._kind == "cl":
                    m += (-1.0 if k % 2 else 1.0) * (_ierfc(z0) - _ierfc(z2))
                else:
                    m += _ierfc(z0) - 2 * _ierfc(z1) + _ierfc(z2)
            out[img] = s * m
        if ser.any():
            ts = t[ser]
            N = self._n_terms(ts)
            acc = np.zeros_like(ts)
            if self._kind == "cl":
                for n in range(N):
                    lam = (2 * n + 1) * math.pi / (2 * L)
                    acc += np.exp(-D * lam * lam * ts) / (2 * n + 1) ** 2
                out[ser] = L * (1.0 - 8.0 / math.pi**2 * acc)
            else:
                for n in range(1, N + 1, 2):
                    acc += np.exp(-D * (n * math.pi / L) ** 2 * ts) / n**2
                out[ser] = L * (0.5 - 4.0 / math.pi**2 * acc)
        return out

    @staticmethod
    def _prep(x, t):
        x, t = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
        return x, t

    # ------------------------------------------------------------------ public API
    def concentration(self, x, t, method: Method = "auto") -> np.ndarray:
        """c(x, t); ``x`` and ``t`` broadcast against each other (any x in [0, L])."""
        x, t = self._prep(x, t)
        c = np.full(x.shape, self.c_init)
        if self.c_left is not None:
            c += (self.c_left - self.c_init) * self._unit(x, t, method)[0]
        if self.c_right is not None:
            c += (self.c_right - self.c_init) * self._unit(self.L - x, t, method)[0]
        return c

    at = concentration  # query the concentration anywhere: sol.at(x, t)

    def field(self, x, t, method: Method = "auto") -> np.ndarray:
        """Concentration on a grid, shape (len(t), len(x))."""
        return self.concentration(np.asarray(x)[None, :], np.asarray(t)[:, None], method)

    def gradient(self, x, t, method: Method = "auto") -> np.ndarray:
        """dc/dx."""
        x, t = self._prep(x, t)
        g = np.zeros(x.shape)
        if self.c_left is not None:
            g += (self.c_left - self.c_init) * self._unit(x, t, method)[1]
        if self.c_right is not None:
            g -= (self.c_right - self.c_init) * self._unit(self.L - x, t, method)[1]
        return g

    def flux_into_tube(self, t, end: Literal["left", "right"] = "left", method: Method = "auto") -> np.ndarray:
        """Diffusive flux entering the tube through an end [conc * m/s] (per unit area).
        Positive = into the tube.  Zero for a closed wall; infinite at t=0 for a source."""
        t = np.asarray(t, dtype=float)
        c_end = self.c_left if end == "left" else self.c_right
        if c_end is None or c_end == self.c_init:
            return np.zeros(t.shape)
        x_end = np.zeros(t.shape) if end == "left" else np.full(t.shape, self.L)
        g = self.gradient(x_end, t, method)
        J = -self.D * g if end == "left" else self.D * g
        return np.where(t > 0, J, np.inf)

    def mass_per_area(self, t, method: Method = "auto") -> np.ndarray:
        """integral of c dx over the tube [conc * m]; multiply by tube area for mass."""
        t = np.asarray(t, dtype=float)
        m = np.full(t.shape, self.c_init * self.L)
        for c_end in (self.c_left, self.c_right):
            if c_end is not None:
                m = m + (c_end - self.c_init) * self._mass_unit(t, method)
        return m

    def steady_state(self, x) -> np.ndarray:
        """t -> infinity profile."""
        x = np.asarray(x, dtype=float)
        if self.c_left is None and self.c_right is None:
            return np.full(x.shape, self.c_init)
        if self.c_right is None:
            return np.full(x.shape, self.c_left)
        if self.c_left is None:
            return np.full(x.shape, self.c_right)
        return self.c_left + (self.c_right - self.c_left) * x / self.L

    def probe_series(self, x_probe: float, t, method: Method = "auto") -> np.ndarray:
        """Concentration vs time at one position."""
        return self.concentration(x_probe, t, method)
