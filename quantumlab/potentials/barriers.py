import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D

class GaussianBarrier(Potential):

    def __init__(self, V0: float, width: float, position: float=0.0):
        self.V0 = V0
        self.width = width
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            return self.V0 * np.exp(-((grid.X - self.position) / self.width) ** 2)
        elif isinstance(grid, Grid3D):
            return self.V0 * np.exp(-((grid.X - self.position) / self.width) ** 2)
        else:
            return self.V0 * np.exp(-((grid.x - self.position) / self.width) ** 2)

class RectangularBarrier(Potential):

    def __init__(self, V0: float, width: float, position: float=0.0):
        self.V0 = V0
        self.width = width
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, (Grid2D, Grid3D)):
            mask = (grid.X >= self.position - self.width / 2.0) & (grid.X <= self.position + self.width / 2.0)
            return np.where(mask, self.V0, 0.0)
        else:
            mask = (grid.x >= self.position - self.width / 2.0) & (grid.x <= self.position + self.width / 2.0)
            return np.where(mask, self.V0, 0.0)

class PotentialStep(Potential):

    def __init__(self, V0: float, position: float=0.0):
        self.V0 = V0
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, (Grid2D, Grid3D)):
            return np.where(grid.X >= self.position, self.V0, 0.0)
        else:
            return np.where(grid.x >= self.position, self.V0, 0.0)

class MultipleBarriers(Potential):

    def __init__(self, V0: float, width: float, positions: list):
        self.V0 = V0
        self.width = width
        self.positions = positions

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, (Grid2D, Grid3D)):
            V = np.zeros(grid.shape)
            for pos in self.positions:
                mask = (grid.X >= pos - self.width / 2.0) & (grid.X <= pos + self.width / 2.0)
                V[mask] = self.V0
            return V
        else:
            V = np.zeros(grid.shape)
            for pos in self.positions:
                mask = (grid.x >= pos - self.width / 2.0) & (grid.x <= pos + self.width / 2.0)
                V[mask] = self.V0
            return V

class ResonantTunnelingDiode(Potential):

    def __init__(self, V0: float, barrier_width: float, well_width: float, position: float=0.0):
        self.V0 = V0
        self.barrier_width = barrier_width
        self.well_width = well_width
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        half_w = self.well_width / 2.0
        w_b = self.barrier_width
        pos_left = self.position - half_w - w_b / 2.0
        pos_right = self.position + half_w + w_b / 2.0
        if isinstance(grid, (Grid2D, Grid3D)):
            mask_left = (grid.X >= pos_left - w_b / 2.0) & (grid.X <= pos_left + w_b / 2.0)
            mask_right = (grid.X >= pos_right - w_b / 2.0) & (grid.X <= pos_right + w_b / 2.0)
            V = np.zeros(grid.shape)
            V[mask_left] = self.V0
            V[mask_right] = self.V0
            return V
        else:
            mask_left = (grid.x >= pos_left - w_b / 2.0) & (grid.x <= pos_left + w_b / 2.0)
            mask_right = (grid.x >= pos_right - w_b / 2.0) & (grid.x <= pos_right + w_b / 2.0)
            V = np.zeros(grid.shape)
            V[mask_left] = self.V0
            V[mask_right] = self.V0
            return V

class Gaussian2DBarrier(Potential):
    """
    Isotropic 2D Gaussian barrier centred at (x0, y0):

        V(x, y) = V0 · exp(−((x−x0)² + (y−y0)²) / width²)

    Unlike the 1D ``GaussianBarrier``, this is a true radially-symmetric
    hill that diffracts wave packets in both transverse directions.

    Parameters
    ----------
    V0 : float
        Barrier height (energy units).
    width : float
        Gaussian width parameter (same units as x, y).
    x0 : float
        x-coordinate of the barrier centre. Default 0.0.
    y0 : float
        y-coordinate of the barrier centre. Default 0.0.
    """

    def __init__(self, V0: float, width: float, x0: float = 0.0, y0: float = 0.0):
        self.V0 = V0
        self.width = width
        self.x0 = x0
        self.y0 = y0

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            r2 = (grid.X - self.x0) ** 2 + (grid.Y - self.y0) ** 2
            return self.V0 * np.exp(-r2 / self.width ** 2)
        elif isinstance(grid, Grid3D):
            r2 = (grid.X - self.x0) ** 2 + (grid.Y - self.y0) ** 2
            return self.V0 * np.exp(-r2 / self.width ** 2)
        else:
            raise TypeError('Gaussian2DBarrier requires a 2D or 3D grid.')

class Gaussian3DBarrier(Potential):
    """
    Isotropic 3D spherical Gaussian barrier centred at (x0, y0, z0):

        V(x, y, z) = V0 · exp(−((x−x0)² + (y−y0)² + (z−z0)²) / width²)

    Parameters
    ----------
    V0 : float
        Barrier height.
    width : float
        Gaussian width (same units as x, y, z).
    x0, y0, z0 : float
        Barrier centre coordinates. Default 0.0.
    """

    def __init__(self, V0: float, width: float,
                 x0: float = 0.0, y0: float = 0.0, z0: float = 0.0):
        self.V0 = V0
        self.width = width
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

    def evaluate(self, grid) -> np.ndarray:
        if not isinstance(grid, Grid3D):
            raise TypeError('Gaussian3DBarrier requires a Grid3D.')
        r2 = ((grid.X - self.x0) ** 2 +
              (grid.Y - self.y0) ** 2 +
              (grid.Z - self.z0) ** 2)
        return self.V0 * np.exp(-r2 / self.width ** 2)
