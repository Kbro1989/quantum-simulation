import numpy as np
from scipy.fft import fft2, ifft2
from quantumlab.solvers.base import Solver
from quantumlab.core.wavefunction import WaveFunction2D

class SplitStep2DSolver(Solver):
    """
    2D Split-Step Fourier Method (SSFM) solver for the time-dependent
    Schrödinger equation on a 2D spatial grid.

    The symmetric splitting operator is applied as:
        ψ(t+dt) = U_V · FFT2⁻¹[ U_T · FFT2[ U_V · ψ(t) ] ]

    where U_V = exp(-i·V·dt/(2ℏ)) and U_T = exp(-i·T_k·dt/ℏ).
    """

    def __init__(self, grid, potential, dt: float, hbar: float = 1.0, m: float = 1.0):
        super().__init__(grid, potential, dt, hbar, m)
        self.V = self.potential.evaluate(self.grid)
        self.T_k = self.hbar**2 * (self.grid.K_x**2 + self.grid.K_y**2) / (2.0 * self.m)
        self.U_V = None
        self.U_T = None
        self.update_operators()

    def update_operators(self):
        """Pre-compute the position and momentum space propagators."""
        self.U_V = np.exp(-1j * self.V * self.dt / (2.0 * self.hbar))
        self.U_T = np.exp(-1j * self.T_k * self.dt / self.hbar)

    def step(self, wavefunction: WaveFunction2D) -> WaveFunction2D:
        """Advance the wavefunction by one time step using SSFM."""
        psi = wavefunction.psi.copy()
        # Half-step in position space
        psi *= self.U_V
        # Full step in momentum space
        psi_k = fft2(psi)
        psi_k *= self.U_T
        psi = ifft2(psi_k)
        # Half-step in position space
        psi *= self.U_V
        return WaveFunction2D(self.grid, psi)
