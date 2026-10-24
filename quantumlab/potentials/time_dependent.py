"""
Explicitly time-dependent potentials V(r, t).

Solvers detect these through the ``time_dependent`` flag and call
``evaluate_at(grid, t)`` at every (sub)step instead of caching V once.
"""
from abc import abstractmethod
from typing import Callable, Optional
import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.spectral import coordinates

_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


class TimeDependentPotential(Potential):
    time_dependent = True

    @abstractmethod
    def evaluate_at(self, grid, t: float) -> np.ndarray:
        pass

    def evaluate(self, grid) -> np.ndarray:
        return self.evaluate_at(grid, 0.0)


class DipoleDrive(TimeDependentPotential):
    """
    Static potential plus a spatially uniform oscillating force along one axis
    (dipole approximation, length gauge):

        V(r, t) = V_base(r) − F(t) · r_axis,
        F(t)    = F0 · envelope(t) · sin(ω t + φ)

    For a particle of charge q in an electric field E(t), F(t) = q E(t).

    Parameters
    ----------
    base : Potential or None
        Static background potential (None means free space).
    F0 : float
        Peak force amplitude.
    omega : float
        Driving angular frequency.
    axis : {'x', 'y', 'z'}
        Direction of the force.
    phase : float
        Carrier phase φ.
    envelope : callable, optional
        envelope(t) -> float, e.g. ``sin2_envelope(T)``. Defaults to 1.
    """

    def __init__(self, base: Optional[Potential], F0: float, omega: float,
                 axis: str = 'x', phase: float = 0.0,
                 envelope: Optional[Callable[[float], float]] = None):
        if axis not in _AXIS_INDEX:
            raise ValueError("axis must be 'x', 'y' or 'z'")
        self.base = base
        self.F0 = F0
        self.omega = omega
        self.axis = axis
        self.phase = phase
        self.envelope = envelope
        self._cache_grid = None
        self._cache = None

    def force(self, t: float) -> float:
        env = 1.0 if self.envelope is None else float(self.envelope(t))
        return self.F0 * env * np.sin(self.omega * t + self.phase)

    def _static_parts(self, grid):
        if self._cache_grid is not grid:
            coords = coordinates(grid)
            idx = _AXIS_INDEX[self.axis]
            if idx >= len(coords):
                raise ValueError(f"axis '{self.axis}' not available on a {len(coords)}D grid")
            V_base = np.zeros(grid.shape) if self.base is None else self.base.evaluate(grid)
            self._cache = (V_base, coords[idx])
            self._cache_grid = grid
        return self._cache

    def evaluate_at(self, grid, t: float) -> np.ndarray:
        V_base, r_axis = self._static_parts(grid)
        return V_base - self.force(t) * r_axis


class TimeDependentCustomPotential(TimeDependentPotential):
    """User-defined V(r, t): ``func(x, t)``, ``func(X, Y, t)`` or ``func(X, Y, Z, t)``."""

    def __init__(self, func: Callable):
        if not callable(func):
            raise TypeError('func must be callable')
        self.func = func

    def evaluate_at(self, grid, t: float) -> np.ndarray:
        return np.broadcast_to(self.func(*coordinates(grid), t), grid.shape).astype(float)


def sin2_envelope(duration: float) -> Callable[[float], float]:
    """Smooth sin² pulse envelope that is non-zero only on 0 ≤ t ≤ duration."""
    def envelope(t: float) -> float:
        if t <= 0.0 or t >= duration:
            return 0.0
        return float(np.sin(np.pi * t / duration) ** 2)
    return envelope


def flat_top_envelope(duration: float, ramp: float) -> Callable[[float], float]:
    """
    Pulse envelope with sin² turn-on and turn-off of length ``ramp`` and a
    flat top in between; zero outside 0 ≤ t ≤ duration. Long flat tops give
    narrow, well-resolved harmonic lines.
    """
    if not 0 < 2 * ramp <= duration:
        raise ValueError('need 0 < 2·ramp ≤ duration')

    def envelope(t: float) -> float:
        if t <= 0.0 or t >= duration:
            return 0.0
        if t < ramp:
            return float(np.sin(0.5 * np.pi * t / ramp) ** 2)
        if t > duration - ramp:
            return float(np.sin(0.5 * np.pi * (duration - t) / ramp) ** 2)
        return 1.0
    return envelope
