import numpy as np
from quantumlab.core.grid import Grid1D, Grid2D

class AbsorbingBoundaryLayer:
    """
    Polynomial absorbing boundary layer (ABL) for split-step Fourier simulations.

    Applies a smooth mask to the wavefunction near the grid edges to prevent
    FFT wraparound artifacts from unphysical periodic boundary conditions.

    The mask is 1.0 in the interior and transitions to 0.0 over a boundary
    region of width `boundary_width` using a polynomial profile of order `order`.

    Parameters
    ----------
    grid : Grid1D or Grid2D
        The spatial grid.
    boundary_width : float
        Physical width of the absorbing layer at each edge.
    order : int, default 3
        Polynomial order of the mask transition. Higher orders give sharper
        transitions but may introduce reflections.
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
        if self.order == 2:
            return 2.0 * d - d * d
        elif self.order == 3:
            return d * d * (3.0 - 2.0 * d)
        elif self.order == 4:
            return d * d * d * (4.0 - 3.0 * d)
        else:
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
        else:
            return self._build_mask_2d()

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

    def apply(self, wavefunction):
        """Apply the absorbing mask to a wavefunction in-place."""
        wavefunction.psi *= self.mask
        return wavefunction
