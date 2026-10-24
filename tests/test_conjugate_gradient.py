import numpy as np
import pytest
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction2D
from quantumlab.potentials import HarmonicTrap, SoftCoulomb, DipoleDrive
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.stationary import ground_state_pcg, ground_state
from quantumlab.observables import (
    kinetic_energy_expectation, potential_energy_expectation, interaction_energy, angular_momentum_z_expectation,
)


@pytest.mark.parametrize('grid, potential, kwargs, exact', [
    (Grid1D(256, -12.0, 12.0), HarmonicTrap(1.0), {}, 0.5),
    (Grid1D(2048, -100.0, 100.0), SoftCoulomb(), {}, -0.5),                       # V < 0
    (Grid1D(256, -12.0, 12.0), HarmonicTrap(2.0, m=0.5), dict(hbar=0.7, m=0.5), 0.7),
    (Grid3D(32, -7.0, 7.0, 32, -7.0, 7.0, 32, -7.0, 7.0), HarmonicTrap((1.0, 1.2, 1.4)), {}, 1.8),
])
def test_linear_ground_states_to_machine_precision(grid, potential, kwargs, exact):
    wf, info = ground_state_pcg(grid, potential, tol=1e-10, max_iter=3000, **kwargs)
    assert info.converged
    assert info.residual < 1e-10
    assert info.energy == pytest.approx(exact, abs=1e-12)
    assert wf.norm() == pytest.approx(1.0, abs=1e-12)


def test_gp_ground_state_without_dtau_floor():
    """The minimiser's fixed point is exact: the virial theorem holds to ~1e-9, not to O(dτ²).
    (The virial theorem is a continuum identity; the grid must resolve the state well.)"""
    grid = Grid3D(48, -8.0, 8.0, 48, -8.0, 8.0, 48, -8.0, 8.0)
    trap, g = HarmonicTrap((1.0, 1.2, 1.4)), 60.0
    wf, info = ground_state_pcg(grid, trap, nonlinearity=g, tol=1e-9)
    T, V, E_int = kinetic_energy_expectation(wf), potential_energy_expectation(wf, trap), interaction_energy(wf, g)
    assert info.converged
    assert abs(2 * T - 2 * V + 3 * E_int) / V < 1e-8
    _, it = ground_state(grid, trap, dtau=0.005, nonlinearity=g)
    assert info.residual < 1e-3 * it.residual
    assert info.energy <= it.energy + 1e-12


def test_initial_norm_is_kept():
    grid = Grid1D(256, -12.0, 12.0)
    guess = np.exp(-grid.x ** 2).astype(complex) * 3.0
    wf, info = ground_state_pcg(grid, HarmonicTrap(1.0), initial=guess, nonlinearity=5.0, tol=1e-9)
    assert wf.norm() == pytest.approx(np.sum(np.abs(guess) ** 2) * grid.dx, rel=1e-12)
    assert info.converged


def test_rotating_vortex_lattice_converges_below_imaginary_time():
    grid = Grid2D(64, -8.0, 8.0, 64, -8.0, 8.0)
    trap, g, Om = HarmonicTrap(1.0), 300.0, 0.8
    no_vortex, _ = ground_state(grid, trap, dtau=0.005, nonlinearity=g)
    seed = WaveFunction2D(grid, no_vortex.psi * np.exp(1j * np.arctan2(grid.Y - 0.3, grid.X - 0.2)))
    lattice = SplitOperatorSolver(grid, trap, 0.005, nonlinearity=g, imaginary_time=True, rotation=Om).evolve(seed, 1500)
    _, it = ground_state(grid, trap, dtau=0.005, nonlinearity=g, rotation=Om, initial=lattice, max_steps=2000)
    wf, info = ground_state_pcg(grid, trap, initial=lattice, nonlinearity=g, rotation=Om, tol=1e-7, max_iter=3000)
    assert info.converged
    assert info.energy < it.energy
    assert angular_momentum_z_expectation(wf) > 2.0


def test_callback_and_validation():
    calls = []
    ground_state_pcg(Grid1D(64, -6.0, 6.0), HarmonicTrap(1.0), tol=1e-8,
                     callback=lambda i, r, E: calls.append((i, r, E)), callback_interval=5)
    assert calls and calls[0][0] == 0 and all(c[0] % 5 == 0 for c in calls)
    with pytest.raises(ValueError):
        ground_state_pcg(Grid1D(32, -4.0, 4.0), DipoleDrive(None, 1.0, 1.0))
