"""
3D Split-Step Fourier solver with CuPy GPU acceleration.
Automatic fallback to CPU (SciPy) if CuPy is unavailable.
Deterministic: same grid, same potential, same dt → bit-identical evolution.
"""
import numpy as np
from quantumlab.solvers.base import Solver
from quantumlab.core.wavefunction import WaveFunction3D

try:
    import cupy as cp
    from cupy.fft import fftn as cp_fftn, ifftn as cp_ifftn
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from scipy.fft import fftn, ifftn


class SplitStep3DGPU(Solver):
    """
    GPU-accelerated 3D SSFM. Grid operators are transferred to GPU once at init.
    Wavefunctions are uploaded/downloaded per step to minimize PCI-E overhead.
    For batch runs, keep data on GPU between steps using the device context.
    """
    def __init__(self, grid, potential, dt: float, hbar: float = 1.0,
                 m: float = 1.0, device_id: int = 0):
        super().__init__(grid, potential, dt, hbar, m)
        self.device_id = device_id
        self._gpu = CUPY_AVAILABLE

        if self._gpu:
            cp.cuda.Device(device_id).use()
            self.V = cp.asarray(self.potential.evaluate(self.grid))
            self.T_k = cp.asarray(
                (self.hbar ** 2 *
                 (self.grid.K_x ** 2 + self.grid.K_y ** 2 + self.grid.K_z ** 2) /
                 (2.0 * self.m))
            )
        else:
            self.V = self.potential.evaluate(self.grid)
            self.T_k = (self.hbar ** 2 *
                        (self.grid.K_x ** 2 + self.grid.K_y ** 2 + self.grid.K_z ** 2) /
                        (2.0 * self.m))

        self.update_operators()

    def update_operators(self):
        if self._gpu:
            self.U_V = cp.exp(-1j * self.V * self.dt / (2.0 * self.hbar))
            self.U_T = cp.exp(-1j * self.T_k * self.dt / self.hbar)
        else:
            self.U_V = np.exp(-1j * self.V * self.dt / (2.0 * self.hbar))
            self.U_T = np.exp(-1j * self.T_k * self.dt / self.hbar)

    def step(self, wavefunction: WaveFunction3D) -> WaveFunction3D:
        if self._gpu:
            psi = cp.asarray(wavefunction.psi)
            psi *= self.U_V
            psi_k = cp_fftn(psi)
            psi_k *= self.U_T
            psi = cp_ifftn(psi_k)
            psi *= self.U_V
            return WaveFunction3D(self.grid, cp.asnumpy(psi))
        else:
            psi = wavefunction.psi.copy()
            psi *= self.U_V
            psi_k = fftn(psi)
            psi_k *= self.U_T
            psi = ifftn(psi_k)
            psi *= self.U_V
            return WaveFunction3D(self.grid, psi)

    @property
    def backend(self) -> str:
        return "cupy" if self._gpu else "scipy-cpu"
