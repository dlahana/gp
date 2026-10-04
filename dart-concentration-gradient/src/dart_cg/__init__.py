"""DART concentration gradient: 1D vapor transport in a diffusion tube."""
from .geometry import Probe, Tube
from .solvers.analytical import AnalyticalSolution

__all__ = ["Tube", "Probe", "AnalyticalSolution"]
__version__ = "0.1.0"
