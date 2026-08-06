import numpy as np
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D

class WaveFunction1D:

    def __init__(self, grid: Grid1D, psi: np.ndarray=None):
        self.grid = grid
        if psi is None:
            self.psi = np.zeros(grid.N, dtype=complex)
        else:
            if psi.shape != grid.shape:
                raise ValueError(f'psi shape {psi.shape} does not match grid shape {grid.shape}')
            self.psi = np.array(psi, dtype=complex)

    @property
    def probability_density(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        return float(np.sum(self.probability_density) * self.grid.dx)

    def normalize(self) -> 'WaveFunction1D':
        n = self.norm()
        if n > 0:
            self.psi /= np.sqrt(n)
        return self

    def fourier_transform(self) -> np.ndarray:
        return np.fft.fft(self.psi) * (self.grid.dx / np.sqrt(2 * np.pi))

    def copy(self) -> 'WaveFunction1D':
        return WaveFunction1D(self.grid, self.psi.copy())

    @classmethod
    def gaussian(cls, grid: Grid1D, x0: float, k0: float, sigma: float) -> 'WaveFunction1D':
        psi = np.exp(-(grid.x - x0) ** 2 / (4.0 * sigma ** 2)) * np.exp(1j * k0 * grid.x)
        wf = cls(grid, psi)
        wf.normalize()
        return wf

class WaveFunction2D:

    def __init__(self, grid: Grid2D, psi: np.ndarray=None):
        self.grid = grid
        if psi is None:
            self.psi = np.zeros(grid.shape, dtype=complex)
        else:
            if psi.shape != grid.shape:
                raise ValueError(f'psi shape {psi.shape} does not match grid shape {grid.shape}')
            self.psi = np.array(psi, dtype=complex)

    @property
    def probability_density(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        return float(np.sum(self.probability_density) * self.grid.dx * self.grid.dy)

    def normalize(self) -> 'WaveFunction2D':
        n = self.norm()
        if n > 0:
            self.psi /= np.sqrt(n)
        return self

    def fourier_transform(self) -> np.ndarray:
        return np.fft.fft2(self.psi) * (self.grid.dx * self.grid.dy / (2 * np.pi))

    def copy(self) -> 'WaveFunction2D':
        return WaveFunction2D(self.grid, self.psi.copy())

    @classmethod
    def gaussian(cls, grid: Grid2D, x0: float, y0: float, k0_x: float, k0_y: float, sigma_x: float, sigma_y: float) -> 'WaveFunction2D':
        psi_x = np.exp(-(grid.X - x0) ** 2 / (4.0 * sigma_x ** 2)) * np.exp(1j * k0_x * grid.X)
        psi_y = np.exp(-(grid.Y - y0) ** 2 / (4.0 * sigma_y ** 2)) * np.exp(1j * k0_y * grid.Y)
        psi = psi_x * psi_y
        wf = cls(grid, psi)
        wf.normalize()
        return wf

class WaveFunction3D:

    def __init__(self, grid: Grid3D, psi: np.ndarray=None):
        self.grid = grid
        if psi is None:
            self.psi = np.zeros(grid.shape, dtype=complex)
        else:
            if psi.shape != grid.shape:
                raise ValueError(f'psi shape {psi.shape} does not match grid shape {grid.shape}')
            self.psi = np.array(psi, dtype=complex)

    @property
    def probability_density(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        return float(np.sum(self.probability_density) * self.grid.dx * self.grid.dy * self.grid.dz)

    def normalize(self) -> 'WaveFunction3D':
        n = self.norm()
        if n > 0:
            self.psi /= np.sqrt(n)
        return self

    def fourier_transform(self) -> np.ndarray:
        return np.fft.fftn(self.psi) * (self.grid.dx * self.grid.dy * self.grid.dz / (2 * np.pi) ** 1.5)

    def copy(self) -> 'WaveFunction3D':
        return WaveFunction3D(self.grid, self.psi.copy())

    def slice_xy(self, z_idx: int=None) -> np.ndarray:
        """Return the probability density slice |ψ|² in the XY plane at z index z_idx."""
        if z_idx is None:
            z_idx = self.grid.N_z // 2
        return self.probability_density[:, :, z_idx]

    def slice_xz(self, y_idx: int=None) -> np.ndarray:
        """Return the probability density slice |ψ|² in the XZ plane at y index y_idx."""
        if y_idx is None:
            y_idx = self.grid.N_y // 2
        return self.probability_density[:, y_idx, :]

    def slice_yz(self, x_idx: int=None) -> np.ndarray:
        """Return the probability density slice |ψ|² in the YZ plane at x index x_idx."""
        if x_idx is None:
            x_idx = self.grid.N_x // 2
        return self.probability_density[x_idx, :, :]

    @classmethod
    def gaussian(cls, grid: Grid3D, x0: float, y0: float, z0: float,
                 k0_x: float, k0_y: float, k0_z: float,
                 sigma_x: float, sigma_y: float, sigma_z: float) -> 'WaveFunction3D':
        psi_x = np.exp(-(grid.X - x0) ** 2 / (4.0 * sigma_x ** 2)) * np.exp(1j * k0_x * grid.X)
        psi_y = np.exp(-(grid.Y - y0) ** 2 / (4.0 * sigma_y ** 2)) * np.exp(1j * k0_y * grid.Y)
        psi_z = np.exp(-(grid.Z - z0) ** 2 / (4.0 * sigma_z ** 2)) * np.exp(1j * k0_z * grid.Z)
        psi = psi_x * psi_y * psi_z
        wf = cls(grid, psi)
        wf.normalize()
        return wf

