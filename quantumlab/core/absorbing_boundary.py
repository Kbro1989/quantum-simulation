import numpy as np
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D

class AbsorbingBoundaryLayer:
    """
    Polynomial absorbing boundary layer (ABL) for split-step Fourier simulations.

    Applies a smooth mask to the wavefunction near the grid edges to prevent
    FFT wraparound artifacts from unphysical periodic boundary conditions.

    The mask is 1.0 in the interior and transitions to 0.0 over a boundary
    region of width `boundary_width` using a polynomial profile of order `order`.

    Parameters
    ----------
    grid : Grid1D, Grid2D, or Grid3D
        The spatial grid.
    boundary_width : float
        Physical width of the absorbing layer at each edge.
    order : int, default 3
        Polynomial order of the mask transition. Use 3 (smoothstep) for
        well-behaved transitions; higher orders give sharper roll-offs but
        may introduce reflections. Minimum effective order is 3.
    """

    def __init__(self, grid, boundary_width: float, order: int = 3):
        self.grid = grid
        self.boundary_width = boundary_width
        self.order = order
        self.mask = self._build_mask()

    def _polynomial_profile(self, d):
        """
        Polynomial profile: 0 at d=0, 1 at d=1, with zero first derivative at both ends.
        d is normalized distance into the absorbing layer (0 = edge, 1 = interior boundary).
        """
        d = np.clip(d, 0.0, 1.0)
        if self.order <= 3:
            # Cubic smoothstep: C1-smooth at both endpoints, zero derivative at d=0 and d=1
            return d * d * (3.0 - 2.0 * d)
        elif self.order == 4:
            # Quintic smootherstep (Perlin): C2-smooth at both endpoints
            return d * d * d * (d * (d * 6.0 - 15.0) + 10.0)
        else:
            # General-order smoothstep via Bernstein polynomial
            result = np.zeros_like(d)
            for k in range(self.order):
                result += self._binomial(self.order + k - 1, k) * \
                          self._binomial(2 * self.order - 1, self.order - k - 1) * \
                          (-d) ** k
            result *= d ** self.order
            return result

    def _binomial(self, n, k):
        from math import comb
        return comb(n, k)

    def _build_mask(self):
        """Construct the absorbing mask for the grid."""
        if isinstance(self.grid, Grid1D):
            return self._build_mask_1d()
        elif isinstance(self.grid, Grid2D):
            return self._build_mask_2d()
        elif isinstance(self.grid, Grid3D):
            return self._build_mask_3d()
        else:
            raise TypeError(f'Unsupported grid type: {type(self.grid)}')

    def _build_mask_1d(self):
        x = self.grid.x
        dx = self.grid.dx
        bw = self.boundary_width
        mask = np.ones_like(x)

        left_edge = x.min()
        left_transition = (x - left_edge) / bw
        mask *= self._polynomial_profile(left_transition)

        right_edge = x.max() + dx
        right_transition = (right_edge - x) / bw
        mask *= self._polynomial_profile(right_transition)

        return mask

    def _build_mask_2d(self):
        x = self.grid.x
        y = self.grid.y
        dx = self.grid.dx
        dy = self.grid.dy
        bw = self.boundary_width

        x_min, x_max = x.min(), x.max() + dx
        y_min, y_max = y.min(), y.max() + dy

        mask_x = np.ones_like(x)
        mask_x *= self._polynomial_profile((x - x_min) / bw)
        mask_x *= self._polynomial_profile((x_max - x) / bw)

        mask_y = np.ones_like(y)
        mask_y *= self._polynomial_profile((y - y_min) / bw)
        mask_y *= self._polynomial_profile((y_max - y) / bw)

        mask = mask_x[:, np.newaxis] * mask_y[np.newaxis, :]
        return mask

    def _build_mask_3d(self):
        x, y, z = self.grid.x, self.grid.y, self.grid.z
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        bw = self.boundary_width

        x_min, x_max = x.min(), x.max() + dx
        y_min, y_max = y.min(), y.max() + dy
        z_min, z_max = z.min(), z.max() + dz

        mask_x = np.ones_like(x)
        mask_x *= self._polynomial_profile((x - x_min) / bw)
        mask_x *= self._polynomial_profile((x_max - x) / bw)

        mask_y = np.ones_like(y)
        mask_y *= self._polynomial_profile((y - y_min) / bw)
        mask_y *= self._polynomial_profile((y_max - y) / bw)

        mask_z = np.ones_like(z)
        mask_z *= self._polynomial_profile((z - z_min) / bw)
        mask_z *= self._polynomial_profile((z_max - z) / bw)

        # Separable outer product across three axes
        mask = mask_x[:, np.newaxis, np.newaxis] * mask_y[np.newaxis, :, np.newaxis] * mask_z[np.newaxis, np.newaxis, :]
        return mask

    def apply(self, wavefunction):
        """Apply the absorbing mask to a wavefunction in-place."""
        wavefunction.psi *= self.mask
        return wavefunction
