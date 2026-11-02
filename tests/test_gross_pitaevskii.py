import numpy as np
import pytest
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.potentials import HarmonicTrap
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.stationary import ground_state
from quantumlab.observables import (
    kinetic_energy_expectation, potential_energy_expectation, interaction_energy,
    gross_pitaevskii_energy, chemical_potential, position_uncertainty, y_uncertainty, z_uncertainty,
)


def _bright_soliton(grid, kappa, x0, v):
    """Exact solution of iψ_t = −½ψ_xx − |ψ|²ψ: ψ = κ sech(κ(x−x0)) e^{ivx} (norm 2κ)."""
    return kappa / np.cosh(kappa * (grid.x - x0)) * np.exp(1j * v * grid.x)


@pytest.mark.parametrize('order', [2, 4])
def test_bright_soliton_propagates_without_dispersion(order):
    grid = Grid1D(2048, -60.0, 60.0)
    kappa, x0, v = 1.0, -20.0, 1.5
    wf = WaveFunction1D(grid, _bright_soliton(grid, kappa, x0, v))
    solver = SplitOperatorSolver(grid, None, 0.005, nonlinearity=-1.0, order=order)
    wf = solver.evolve(wf, 4000)
    t = solver.t
    exact = (kappa / np.cosh(kappa * (grid.x - x0 - v * t))) ** 2
    tol = 1e-8 if order == 4 else 1e-4
    assert np.max(np.abs(wf.probability_density - exact)) < tol
    assert wf.norm() == pytest.approx(2.0 * kappa, abs=1e-10)


def test_soliton_disperses_without_nonlinearity():
    grid = Grid1D(2048, -60.0, 60.0)
    wf = WaveFunction1D(grid, _bright_soliton(grid, 1.0, 0.0, 0.0))
    wf = SplitOperatorSolver(grid, None, 0.01, nonlinearity=0.0).evolve(wf, 1000)
    assert wf.probability_density.max() < 0.5


def test_gp_energy_conserved_in_real_time():
    grid = Grid2D(96, -10.0, 10.0, 96, -10.0, 10.0)
    trap = HarmonicTrap((1.0, 1.0))
    g = 30.0
    wf0, _ = ground_state(grid, trap, dtau=0.01, nonlinearity=g)
    # Kick the condensate and let it slosh in a displaced trap
    shaken = HarmonicTrap((1.0, 1.0), center=(1.0, 0.0))
    E0 = gross_pitaevskii_energy(wf0, shaken, g)
    wf = SplitOperatorSolver(grid, shaken, 0.005, nonlinearity=g, order=4).evolve(wf0, 400)
    assert gross_pitaevskii_energy(wf, shaken, g) == pytest.approx(E0, rel=1e-6)
    assert wf.norm() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_virial_theorem_for_gp_ground_state(dim):
    """For V = ½Σω²r², the GP ground state obeys 2⟨T⟩ − 2⟨V⟩ + d·E_int = 0."""
    if dim == 1:
        grid, omegas, g = Grid1D(256, -12.0, 12.0), 1.0, 20.0
    elif dim == 2:
        grid, omegas, g = Grid2D(96, -10.0, 10.0, 96, -10.0, 10.0), (1.0, 1.3), 50.0
    else:
        grid, omegas, g = Grid3D(48, -8.0, 8.0, 48, -8.0, 8.0, 48, -8.0, 8.0), (1.0, 1.2, 1.4), 60.0
    trap = HarmonicTrap(omegas)
    wf, info = ground_state(grid, trap, dtau=0.005, nonlinearity=g)
    T = kinetic_energy_expectation(wf)
    V = potential_energy_expectation(wf, trap)
    E_int = interaction_energy(wf, g)
    assert info.converged
    assert abs(2 * T - 2 * V + dim * E_int) / V < 1e-4
    assert chemical_potential(wf, trap, g) == pytest.approx(info.chemical_potential, rel=1e-10)


def test_repulsion_broadens_condensate():
    grid = Grid3D(40, -8.0, 8.0, 40, -8.0, 8.0, 40, -8.0, 8.0)
    trap = HarmonicTrap(1.0)
    wf0, _ = ground_state(grid, trap, dtau=0.01, nonlinearity=0.0)
    wf1, _ = ground_state(grid, trap, dtau=0.01, nonlinearity=100.0)
    for f in (position_uncertainty, y_uncertainty, z_uncertainty):
        assert f(wf0) == pytest.approx(np.sqrt(0.5), abs=1e-5)
        assert f(wf1) > 1.2 * f(wf0)
