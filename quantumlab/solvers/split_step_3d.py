import numpy as np
from scipy.fft import fftn, ifftn
from quantumlab.solvers.base import Solver
from quantumlab.core.wavefunction import WaveFunction3D


class SplitStep3DSolver(Solver):
    """
    3D Split-Step Fourier Method (SSFM) solver for the time-dependent
    Schrödinger equation on a 3D spatial grid.

    The symmetric Lie-Trotter-Suzuki splitting is applied as:
        ψ(t+dt) = U_V/2 · IFFT3[ U_T · FFT3[ U_V/2 · ψ(t) ] ]

    where U_V = exp(-i·V·dt/(2ℏ)) and U_T = exp(-i·T_k·dt/ℏ),
    with T_k = ℏ²(K_x² + K_y² + K_z²) / (2m).

    This scheme is 2nd-order accurate in time (global error O(dt²)) and
    exactly unitary when no absorbing boundary is applied.

    Parameters
    ----------
    grid : Grid3D
        The 3D spatial grid.
    potential : Potential
        The potential energy operator V(x, y, z).
    dt : float
        Time step.
    hbar : float, default 1.0
        Reduced Planck constant (natural units).
    m : float, default 1.0
        Particle mass (natural units).
    workers : int, default -1
        Number of CPU threads for scipy FFT. -1 uses all available cores.
    """

    def __init__(self, grid, potential, dt: float, hbar: float = 1.0,
                 m: float = 1.0, workers: int = -1):
        super().__init__(grid, potential, dt, hbar, m)
        self.workers = workers
        self.V = self.potential.evaluate(self.grid)
        self.T_k = (self.hbar ** 2 *
                    (self.grid.K_x ** 2 + self.grid.K_y ** 2 + self.grid.K_z ** 2) /
                    (2.0 * self.m))
        self.U_V = None
        self.U_T = None
        self.update_operators()

    def update_operators(self):
        """Pre-compute the position-space and momentum-space propagators."""
        self.U_V = np.exp(-1j * self.V * self.dt / (2.0 * self.hbar))
        self.U_T = np.exp(-1j * self.T_k * self.dt / self.hbar)

    def step(self, wavefunction: WaveFunction3D) -> WaveFunction3D:
        """Advance the wavefunction by one time step using symmetric SSFM."""
        psi = wavefunction.psi.copy()
        # Half-step in position space
        psi *= self.U_V
        # Full step in momentum space
        psi_k = fftn(psi, workers=self.workers)
        psi_k *= self.U_T
        psi = ifftn(psi_k, workers=self.workers)
        # Half-step in position space
        psi *= self.U_V
        return WaveFunction3D(self.grid, psi)
