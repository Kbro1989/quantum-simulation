"""
Dimension-agnostic helpers for spectral (FFT-based) operations on
Grid1D / Grid2D / Grid3D. Used by the generic split-operator solver,
the eigensolvers and the current/angular-momentum observables.
"""
import numpy as np
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D, WaveFunction3D


def ndim(grid) -> int:
    if isinstance(grid, Grid3D):
        return 3
    elif isinstance(grid, Grid2D):
        return 2
    elif isinstance(grid, Grid1D):
        return 1
    raise TypeError(f'Unsupported grid type: {type(grid)}')


def wavefunction_class(grid):
    """Return the WaveFunction class matching the grid dimensionality."""
    return {1: WaveFunction1D, 2: WaveFunction2D, 3: WaveFunction3D}[ndim(grid)]


def volume_element(grid) -> float:
    d = ndim(grid)
    if d == 1:
        return grid.dx
    elif d == 2:
        return grid.dx * grid.dy
    return grid.dx * grid.dy * grid.dz


def coordinates(grid):
    """Position meshes (X, Y, Z) truncated to the grid dimension."""
    d = ndim(grid)
    if d == 1:
        return (grid.x,)
    elif d == 2:
        return (grid.X, grid.Y)
    return (grid.X, grid.Y, grid.Z)


def wavevectors(grid):
    """Angular wavenumber meshes (K_x, K_y, K_z) in FFT ordering."""
    d = ndim(grid)
    if d == 1:
        return (grid.k,)
    elif d == 2:
        return (grid.K_x, grid.K_y)
    return (grid.K_x, grid.K_y, grid.K_z)


def k_squared(grid) -> np.ndarray:
    return sum(K ** 2 for K in wavevectors(grid))


def k_volume_element(grid) -> float:
    d = ndim(grid)
    if d == 1:
        return grid.dk
    elif d == 2:
        return grid.dk_x * grid.dk_y
    return grid.dk_x * grid.dk_y * grid.dk_z
