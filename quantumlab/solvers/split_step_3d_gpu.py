"""
3D Split-Step Fourier solver with CuPy GPU acceleration & zero PCI-E memory bottleneck.
Automatic fallback to CPU (SciPy) if CuPy is unavailable.
Deterministic: same grid, same potential, same dt -> bit-identical evolution.
"""
from typing import Optional, Tuple, Union
import numpy as np
from quantumlab.solvers.base import Solver
from quantumlab.core.wavefunction import WaveFunction3D

try:
    import cupy as cp
    from cupy.fft import fftn as cp_fftn, ifftn as cp_ifftn
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from scipy.fft import fftn, ifftn


class SplitStep3DGPU(Solver):
    """
    GPU-accelerated 3D SSFM with double-precision complex FP64 support.
    Grid operators are transferred to GPU once at init.
    Supports in-VRAM batch stepping via `step_sequence()` or `step_gpu()` to eliminate PCI-E transfer bottlenecks.
    """
    def __init__(
        self,
        grid,
        potential,
        dt: float,
        hbar: float = 1.0,
        m: float = 1.0,
        device_id: int = 0,
        renormalize_interval: int = 0,
    ):
        super().__init__(grid, potential, dt, hbar, m)
        self.device_id = device_id
        self.renormalize_interval = renormalize_interval
        self._step_count = 0
        self._gpu = CUPY_AVAILABLE

        if self._gpu:
            try:
                num_devices = cp.cuda.runtime.getDeviceCount()
                if device_id < 0 or device_id >= num_devices:
                    raise ValueError(
                        f"Invalid GPU device_id {device_id}. Available CUDA devices: {num_devices}"
                    )
                cp.cuda.Device(device_id).use()
            except Exception as err:
                print(f"[WARN] Failed to select CUDA device {device_id}: {err}. Falling back to CPU.")
                self._gpu = False

        if self._gpu:
            self.V = cp.asarray(self.potential.evaluate(self.grid), dtype=cp.float64)
            self.T_k = cp.asarray(
                (self.hbar ** 2 *
                 (self.grid.K_x ** 2 + self.grid.K_y ** 2 + self.grid.K_z ** 2) /
                 (2.0 * self.m)),
                dtype=cp.float64
            )
        else:
            self.V = np.asarray(self.potential.evaluate(self.grid), dtype=np.float64)
            self.T_k = np.asarray(
                (self.hbar ** 2 *
                 (self.grid.K_x ** 2 + self.grid.K_y ** 2 + self.grid.K_z ** 2) /
                 (2.0 * self.m)),
                dtype=np.float64
            )

        self.update_operators()

    def update_operators(self) -> None:
        """Re-compute potential and kinetic evolution operators."""
        if self._gpu:
            self.U_V = cp.exp(-1j * self.V * self.dt / (2.0 * self.hbar))
            self.U_T = cp.exp(-1j * self.T_k * self.dt / self.hbar)
        else:
            self.U_V = np.exp(-1j * self.V * self.dt / (2.0 * self.hbar))
            self.U_T = np.exp(-1j * self.T_k * self.dt / self.hbar)

    def step(self, wavefunction: WaveFunction3D) -> WaveFunction3D:
        """Single-step propagation returning CPU WaveFunction3D object."""
        self._step_count += 1
        if self._gpu:
            psi_gpu = cp.asarray(wavefunction.psi, dtype=cp.complex128)
            psi_gpu = self._step_gpu_internal(psi_gpu)
            return WaveFunction3D(self.grid, cp.asnumpy(psi_gpu))
        else:
            psi = wavefunction.psi.copy()
            psi *= self.U_V
            psi_k = fftn(psi)
            psi_k *= self.U_T
            psi = ifftn(psi_k)
            psi *= self.U_V
            if self.renormalize_interval > 0 and self._step_count % self.renormalize_interval == 0:
                norm = np.sqrt(np.sum(np.abs(psi)**2) * self.grid.dx * self.grid.dy * self.grid.dz)
                if norm > 0:
                    psi /= norm
            return WaveFunction3D(self.grid, psi)

    def step_sequence(self, initial_wf: WaveFunction3D, steps: int) -> WaveFunction3D:
        """
        In-VRAM multi-step propagation eliminating PCI-E transfer bottlenecks.
        Transfers initial psi to GPU ONCE, steps N times on GPU, and downloads final state.
        """
        if not self._gpu:
            wf = initial_wf
            for _ in range(steps):
                wf = self.step(wf)
            return wf

        psi_gpu = cp.asarray(initial_wf.psi, dtype=cp.complex128)
        for _ in range(steps):
            self._step_count += 1
            psi_gpu = self._step_gpu_internal(psi_gpu)

        return WaveFunction3D(self.grid, cp.asnumpy(psi_gpu))

    def _step_gpu_internal(self, psi_gpu):
        """Internal GPU propagation step on CuPy array."""
        psi_gpu *= self.U_V
        psi_k = cp_fftn(psi_gpu)
        psi_k *= self.U_T
        psi_gpu = cp_ifftn(psi_k)
        psi_gpu *= self.U_V

        if self.renormalize_interval > 0 and self._step_count % self.renormalize_interval == 0:
            norm = cp.sqrt(cp.sum(cp.abs(psi_gpu)**2) * self.grid.dx * self.grid.dy * self.grid.dz)
            if norm > 0:
                psi_gpu /= norm
        return psi_gpu

    @property
    def backend(self) -> str:
        return "cupy" if self._gpu else "scipy-cpu"
