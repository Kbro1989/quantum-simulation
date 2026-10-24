import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D

class CustomPotential(Potential):

    def __init__(self, func):
        if not callable(func):
            raise TypeError('func must be callable')
        self.func = func

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid3D):
            return self.func(grid.X, grid.Y, grid.Z)
        elif isinstance(grid, Grid2D):
            return self.func(grid.X, grid.Y)
        else:
            return self.func(grid.x)
