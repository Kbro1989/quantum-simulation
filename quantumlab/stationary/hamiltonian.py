import numpy as np
from scipy.fft import fftn, ifftn
from quantumlab.core.spectral import k_squared


def apply_hamiltonian(psi: np.ndarray, grid, V: np.ndarray, hbar: float = 1.0, m: float = 1.0,
                      nonlinearity: float = 0.0, workers: int = -1) -> np.ndarray:
    """
    Hψ = −ℏ²/2m ∇²ψ + (V + g|ψ|²)ψ with the Laplacian evaluated spectrally
    (exact for band-limited ψ on the periodic grid).
    """
    T_k = hbar ** 2 * k_squared(grid) / (2.0 * m)
    H_psi = ifftn(T_k * fftn(psi, workers=workers), workers=workers)
    V_eff = V if nonlinearity == 0.0 else V + nonlinearity * np.abs(psi) ** 2
    return H_psi + V_eff * psi
