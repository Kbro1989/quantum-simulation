import numpy as np
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D, WaveFunction3D

def get_momentum_wavefunction(wf):
    phi = wf.fourier_transform()
    if isinstance(wf, WaveFunction3D):
        kx_shifted = np.fft.fftshift(wf.grid.k_x)
        ky_shifted = np.fft.fftshift(wf.grid.k_y)
        kz_shifted = np.fft.fftshift(wf.grid.k_z)
        phi_shifted = np.fft.fftshift(phi)
        return ((kx_shifted, ky_shifted, kz_shifted), phi_shifted)
    elif isinstance(wf, WaveFunction2D):
        kx_shifted = np.fft.fftshift(wf.grid.k_x)
        ky_shifted = np.fft.fftshift(wf.grid.k_y)
        phi_shifted = np.fft.fftshift(phi)
        return ((kx_shifted, ky_shifted), phi_shifted)
    else:
        k_shifted = np.fft.fftshift(wf.grid.k)
        phi_shifted = np.fft.fftshift(phi)
        return (k_shifted, phi_shifted)

def get_momentum_probability_density(wf):
    if isinstance(wf, WaveFunction3D):
        (kx, ky, kz), phi = get_momentum_wavefunction(wf)
        return ((kx, ky, kz), np.abs(phi) ** 2)
    elif isinstance(wf, WaveFunction2D):
        (kx, ky), phi = get_momentum_wavefunction(wf)
        return ((kx, ky), np.abs(phi) ** 2)
    else:
        k, phi = get_momentum_wavefunction(wf)
        return (k, np.abs(phi) ** 2)
