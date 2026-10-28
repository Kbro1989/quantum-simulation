import numpy as np
from scipy.fft import fftn, ifftn
from quantumlab.core.spectral import k_squared


def apply_hamiltonian(psi: np.ndarray, grid, V: np.ndarray, hbar: float = 1.0, m: float = 1.0,
                      nonlinearity: float = 0.0, workers: int = -1, rotation: float = 0.0) -> np.ndarray:
    """
    Hψ = −ℏ²/2m ∇²ψ + (V + g|ψ|²)ψ − Ω L_z ψ with all derivatives evaluated
    spectrally (exact for band-limited ψ on the periodic grid). L_z is taken
    about the z axis, matching ``SplitOperatorSolver(rotation=Ω)``.
    """
    T_k = hbar ** 2 * k_squared(grid) / (2.0 * m)
    psi_k = fftn(psi, workers=workers)
    H_psi = ifftn(T_k * psi_k, workers=workers)
    V_eff = V if nonlinearity == 0.0 else V + nonlinearity * np.abs(psi) ** 2
    H_psi = H_psi + V_eff * psi
    if rotation != 0.0:
        d_dx = ifftn(1j * grid.K_x * psi_k, workers=workers)
        d_dy = ifftn(1j * grid.K_y * psi_k, workers=workers)
        L_z_psi = -1j * hbar * (grid.X * d_dy - grid.Y * d_dx)
        H_psi = H_psi - rotation * L_z_psi
    return H_psi
