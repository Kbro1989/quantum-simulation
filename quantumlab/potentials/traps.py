import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.spectral import coordinates


class HarmonicTrap(Potential):
    """
    Anisotropic harmonic trap in 1D, 2D or 3D:

        V(r) = ½ m Σ_j ω_j² (r_j − c_j)²

    Parameters
    ----------
    omegas : float or sequence of float
        Trap frequency per axis (a scalar gives an isotropic trap).
    m : float
        Particle mass.
    center : sequence of float, optional
        Trap centre per axis. Defaults to the origin.
    """

    def __init__(self, omegas, m: float = 1.0, center=None):
        self.omegas = omegas
        self.m = m
        self.center = center

    def evaluate(self, grid) -> np.ndarray:
        coords = coordinates(grid)
        d = len(coords)
        omegas = np.broadcast_to(np.asarray(self.omegas, dtype=float), (d,))
        center = np.zeros(d) if self.center is None else np.broadcast_to(np.asarray(self.center, dtype=float), (d,))
        V = np.zeros(grid.shape)
        for r, w, c in zip(coords, omegas, center):
            V = V + 0.5 * self.m * w ** 2 * (r - c) ** 2
        return V
