"""
Probability current and angular momentum, with derivatives evaluated
spectrally (exact for band-limited wavefunctions on the periodic grid).
"""
import numpy as np
from scipy.fft import fftn, ifftn
from quantumlab.core.grid import Grid2D, Grid3D
from quantumlab.core.spectral import wavevectors, volume_element


def gradient(psi: np.ndarray, grid):
    """Spectral gradient (∂ψ/∂x, ∂ψ/∂y, ∂ψ/∂z) truncated to the grid dimension."""
    psi_k = fftn(psi)
    return tuple(ifftn(1j * K * psi_k) for K in wavevectors(grid))


def probability_current(wf, hbar: float = 1.0, m: float = 1.0):
    """
    Probability current density j = (ℏ/m) Im(ψ* ∇ψ), returned as a tuple of
    arrays (j_x[, j_y[, j_z]]). It satisfies ∂|ψ|²/∂t + ∇·j = 0.
    """
    psi = wf.psi
    return tuple((hbar / m) * np.imag(np.conj(psi) * d) for d in gradient(psi, wf.grid))


def angular_momentum_z_expectation(wf, hbar: float = 1.0) -> float:
    """⟨L_z⟩ = ⟨ψ| −iℏ (x ∂_y − y ∂_x) |ψ⟩ for 2D and 3D wavefunctions."""
    if not isinstance(wf.grid, (Grid2D, Grid3D)):
        raise TypeError('L_z requires a 2D or 3D wavefunction')
    d_dx, d_dy = gradient(wf.psi, wf.grid)[:2]
    Lz_psi = -1j * hbar * (wf.grid.X * d_dy - wf.grid.Y * d_dx)
    return float(np.real(np.sum(np.conj(wf.psi) * Lz_psi) * volume_element(wf.grid)))


def autocorrelation(psi0, psi_t, grid) -> complex:
    """C(t) = ⟨ψ(0)|ψ(t)⟩. Its Fourier transform peaks at the eigenenergies."""
    return complex(np.sum(np.conj(psi0) * psi_t) * volume_element(grid))
