import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.spectral import coordinates


class SoftCoulomb(Potential):
    """
    Soft-core Coulomb potential of a nucleus of charge Z at ``center``:

        V(r) = −Z / sqrt(|r − c|² + a²)

    The softening a removes the singularity so the potential is resolvable on
    a grid. In 1D, Z = 1 and a² = 2 give a ground-state energy of −0.5 a.u.,
    the ionisation potential of hydrogen, which makes it the standard
    single-active-electron model for strong-field physics.
    """

    def __init__(self, Z: float = 1.0, softening: float = np.sqrt(2.0), center=None):
        self.Z = Z
        self.softening = softening
        self.center = center

    def _displacements(self, grid):
        coords = coordinates(grid)
        center = np.zeros(len(coords)) if self.center is None else np.broadcast_to(
            np.asarray(self.center, dtype=float), (len(coords),))
        return [c - c0 for c, c0 in zip(coords, center)]

    def evaluate(self, grid) -> np.ndarray:
        r2 = sum(d ** 2 for d in self._displacements(grid))
        return -self.Z / np.sqrt(r2 + self.softening ** 2)

    def gradient(self, grid):
        """Analytic ∇V as a tuple of arrays (∂V/∂x[, ∂V/∂y[, ∂V/∂z]])."""
        d = self._displacements(grid)
        r2 = sum(di ** 2 for di in d)
        factor = self.Z / (r2 + self.softening ** 2) ** 1.5
        return tuple(factor * di for di in d)
