import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.grid import Grid2D, Grid3D


def _smooth_step(u):
    """tanh step: 0 for u ≪ 0, 1 for u ≫ 0 (≈ 0.98 at u = 2). Safe for large |u|."""
    return 0.5 * (1.0 + np.tanh(u))


class ApertureScreen(Potential):
    """
    Opaque screen perpendicular to x with N equally spaced openings
    (multi-slit / multi-pinhole interferometer) in 2D or 3D.

    The wall occupies wall_position ± wall_thickness/2. Edges are smoothed
    over ``smoothing`` length units so the potential stays resolvable on the
    FFT grid. Openings are centred at y_i = (i − (N−1)/2) · separation.

    Parameters
    ----------
    V0 : float
        Wall height. Choose V0 well above the incident kinetic energy so the
        wall is effectively opaque.
    wall_position, wall_thickness : float
        Location and thickness of the screen along x.
    separation : float
        Centre-to-centre distance between neighbouring openings.
    aperture_width : float
        Slit width (or pinhole diameter).
    n_apertures : int
        Number of openings (2 = double slit).
    aperture : {'slit', 'pinhole'}
        In 3D: 'slit' openings extend along z, 'pinhole' openings are circular
        holes centred at z = 0. Ignored in 2D.
    smoothing : float
        Edge-smoothing length.
    """

    def __init__(self, V0: float, wall_position: float = 0.0, wall_thickness: float = 1.0,
                 separation: float = 4.0, aperture_width: float = 1.0, n_apertures: int = 2,
                 aperture: str = 'slit', smoothing: float = 0.25):
        if aperture not in ('slit', 'pinhole'):
            raise ValueError("aperture must be 'slit' or 'pinhole'")
        self.V0 = V0
        self.wall_position = wall_position
        self.wall_thickness = wall_thickness
        self.separation = separation
        self.aperture_width = aperture_width
        self.n_apertures = n_apertures
        self.aperture = aperture
        self.smoothing = smoothing

    @property
    def aperture_centers(self) -> np.ndarray:
        i = np.arange(self.n_apertures)
        return (i - (self.n_apertures - 1) / 2.0) * self.separation

    def evaluate(self, grid) -> np.ndarray:
        if not isinstance(grid, (Grid2D, Grid3D)):
            raise TypeError('ApertureScreen requires a Grid2D or Grid3D.')
        s = self.smoothing
        half_t = self.wall_thickness / 2.0
        dx = grid.X - self.wall_position
        wall = _smooth_step((dx + half_t) / s) * _smooth_step((half_t - dx) / s)

        half_a = self.aperture_width / 2.0
        opening = np.zeros(grid.shape)
        for yc in self.aperture_centers:
            if isinstance(grid, Grid3D) and self.aperture == 'pinhole':
                rho = np.sqrt((grid.Y - yc) ** 2 + grid.Z ** 2)
            else:
                rho = np.abs(grid.Y - yc)
            opening = opening + _smooth_step((half_a - rho) / s)
        opening = np.clip(opening, 0.0, 1.0)
        return self.V0 * wall * (1.0 - opening)


class StadiumBilliard(Potential):
    """
    Soft-walled Bunimovich stadium in 2D: the set of points within distance R
    of the segment [−L/2, L/2] on the x axis. The classical stadium billiard is
    fully chaotic; its quantum eigenstates show the well-known scarring.

        V(x, y) = V0 · S((dist(x, y) − R) / s)

    where S is a tanh step and dist the distance to the central segment.
    """

    def __init__(self, V0: float, radius: float, length: float, smoothing: float = 0.2):
        self.V0 = V0
        self.radius = radius
        self.length = length
        self.smoothing = smoothing

    def evaluate(self, grid) -> np.ndarray:
        if not isinstance(grid, Grid2D):
            raise TypeError('StadiumBilliard requires a Grid2D.')
        half = self.length / 2.0
        x_clamped = np.clip(grid.X, -half, half)
        dist = np.sqrt((grid.X - x_clamped) ** 2 + grid.Y ** 2)
        return self.V0 * _smooth_step((dist - self.radius) / self.smoothing)
