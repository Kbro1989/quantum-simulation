import numpy as np
import pytest
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction2D, WaveFunction3D
from quantumlab.potentials import HarmonicTrap
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.stationary import ground_state
from quantumlab.observables import (
    position_expectation, y_expectation, z_expectation, angular_momentum_z_expectation,
    autocorrelation, total_energy_expectation, gross_pitaevskii_energy,
)


def _coherent_2d(grid, x0):
    return WaveFunction2D.gaussian(grid, x0, 0.0, 0.0, 0.0, 1 / np.sqrt(2), 1 / np.sqrt(2))


@pytest.mark.parametrize('order, tol', [(2, 2e-4), (4, 1e-7)])
def test_coherent_state_in_rotating_frame(order, tol):
    """
    Frames coincide at t = 0, so ψ'(r, t) = ψ_lab(R(Ωt) r, t) and
    ⟨r'⟩ = R(−Ωt) ⟨r⟩_lab with ⟨r⟩_lab = (x0 cos ωt, 0) for a packet at rest.
    """
    grid = Grid2D(128, -12.0, 12.0, 128, -12.0, 12.0)
    x0, Om = 2.0, 0.6
    wf = _coherent_2d(grid, x0)
    solver = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.01, order=order, rotation=Om)
    for _ in range(10):
        wf = solver.evolve(wf, 100)
        t = solver.t
        assert position_expectation(wf) == pytest.approx(x0 * np.cos(t) * np.cos(Om * t), abs=tol)
        assert y_expectation(wf) == pytest.approx(-x0 * np.cos(t) * np.sin(Om * t), abs=tol)
    assert wf.norm() == pytest.approx(1.0, abs=1e-11)


def test_coherent_state_in_rotating_frame_3d():
    grid = Grid3D(48, -9.0, 9.0, 48, -9.0, 9.0, 32, -6.0, 6.0)
    x0, Om = 2.0, 0.6
    wf = WaveFunction3D.gaussian(grid, x0, 0.0, 0.0, 0.0, 0.0, 0.0, *[1 / np.sqrt(2)] * 3)
    solver = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.01, order=4, rotation=Om)
    wf = solver.evolve(wf, 300)
    t = solver.t
    assert position_expectation(wf) == pytest.approx(x0 * np.cos(t) * np.cos(Om * t), abs=1e-7)
    assert y_expectation(wf) == pytest.approx(-x0 * np.cos(t) * np.sin(Om * t), abs=1e-7)
    assert z_expectation(wf) == pytest.approx(0.0, abs=1e-10)


def test_vortex_is_stationary_with_rotating_frame_energy():
    """ψ ∝ (x+iy)e^{−r²/2} has E = 2ℏω and L_z = ℏ, so E' = 2 − Ω in the rotating frame."""
    grid = Grid2D(128, -12.0, 12.0, 128, -12.0, 12.0)
    Om = 0.6
    wf0 = WaveFunction2D(grid, (grid.X + 1j * grid.Y) * np.exp(-(grid.X ** 2 + grid.Y ** 2) / 2)).normalize()
    solver = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.01, order=4, rotation=Om)
    wf = solver.evolve(wf0, 500)
    C = autocorrelation(wf0.psi, wf.psi, grid)
    assert abs(C) == pytest.approx(1.0, abs=1e-10)
    assert np.angle(C * np.exp(1j * (2.0 - Om) * solver.t)) == pytest.approx(0.0, abs=1e-7)
    assert angular_momentum_z_expectation(wf) == pytest.approx(1.0, abs=1e-10)


def test_rotating_frame_energy_conserved_in_anisotropic_trap():
    """For a potential static in the rotating frame, E' = ⟨H⟩ − Ω⟨L_z⟩ is conserved."""
    grid = Grid2D(96, -10.0, 10.0, 96, -10.0, 10.0)
    trap, Om = HarmonicTrap((1.0, 1.4)), 0.5
    wf = WaveFunction2D.gaussian(grid, 1.0, -0.5, 0.3, 0.8, 0.8, 0.7)

    def E_rot(w):
        return total_energy_expectation(w, trap) - Om * angular_momentum_z_expectation(w)

    E0 = E_rot(wf)
    wf = SplitOperatorSolver(grid, trap, 0.005, order=4, rotation=Om).evolve(wf, 600)
    assert E_rot(wf) == pytest.approx(E0, rel=1e-8)


def test_damping_relaxes_to_ground_state_and_keeps_norm():
    grid = Grid2D(64, -8.0, 8.0, 64, -8.0, 8.0)
    trap = HarmonicTrap(1.0)
    wf = _coherent_2d(grid, 2.0)
    solver = SplitOperatorSolver(grid, trap, 0.01, damping=0.3)
    energies = []
    for _ in range(40):
        wf = solver.evolve(wf, 100)
        energies.append(total_energy_expectation(wf, trap))
    assert np.all(np.diff(energies) < 1e-12)
    assert energies[-1] == pytest.approx(1.0, abs=1e-6)
    assert wf.norm() == pytest.approx(1.0, abs=1e-12)


def test_damped_gpe_keeps_input_norm():
    grid = Grid2D(64, -8.0, 8.0, 64, -8.0, 8.0)
    wf = _coherent_2d(grid, 1.0)
    wf.psi *= np.sqrt(3.0)
    solver = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.01, damping=0.05, nonlinearity=5.0)
    wf = solver.evolve(wf, 200)
    assert wf.norm() == pytest.approx(3.0, abs=1e-10)


def test_rotating_gp_ground_state_with_vortex_lattice_beats_vortex_free_state():
    """Above the critical Ω the vortex lattice has lower rotating-frame energy."""
    grid = Grid2D(96, -8.0, 8.0, 96, -8.0, 8.0)
    trap, g, Om = HarmonicTrap(1.0), 300.0, 0.8
    no_vortex, _ = ground_state(grid, trap, dtau=0.005, nonlinearity=g)
    seed = no_vortex.psi * np.exp(1j * np.arctan2(grid.Y - 0.3, grid.X - 0.2))
    solver = SplitOperatorSolver(grid, trap, 0.005, nonlinearity=g, imaginary_time=True, rotation=Om)
    lattice = solver.evolve(WaveFunction2D(grid, seed), 3000)

    def E_rot(w):
        return gross_pitaevskii_energy(w, trap, g) - Om * angular_momentum_z_expectation(w)

    assert angular_momentum_z_expectation(lattice) > 2.0
    assert E_rot(lattice) < E_rot(no_vortex)


def test_rotation_argument_validation():
    with pytest.raises(ValueError):
        SplitOperatorSolver(Grid1D(32, -4.0, 4.0), None, 0.01, rotation=0.5)
    grid = Grid2D(16, -4.0, 4.0, 16, -4.0, 4.0)
    with pytest.raises(ValueError):
        SplitOperatorSolver(grid, None, 0.01, damping=-0.1)
    with pytest.raises(ValueError):
        SplitOperatorSolver(grid, None, 0.01, damping=0.1, order=4)
    with pytest.raises(ValueError):
        SplitOperatorSolver(grid, None, 0.01, damping=0.1, imaginary_time=True)


def test_rotating_hamiltonian_has_vortex_eigenstate():
    grid = Grid2D(96, -10.0, 10.0, 96, -10.0, 10.0)
    from quantumlab.stationary import apply_hamiltonian
    psi = (grid.X + 1j * grid.Y) * np.exp(-(grid.X ** 2 + grid.Y ** 2) / 2)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * grid.dx * grid.dy)
    H_psi = apply_hamiltonian(psi, grid, HarmonicTrap(1.0).evaluate(grid), rotation=0.6)
    np.testing.assert_allclose(H_psi, (2.0 - 0.6) * psi, atol=1e-12)


def test_rotating_ground_state_below_critical_rotation_is_unchanged():
    """For Ω < ω the linear ground state of H − ΩL_z is still the L_z = 0 Gaussian."""
    grid = Grid2D(96, -10.0, 10.0, 96, -10.0, 10.0)
    wf, info = ground_state(grid, HarmonicTrap(1.0), dtau=0.005, rotation=0.6)
    assert info.energy == pytest.approx(1.0, abs=1e-8)
    assert angular_momentum_z_expectation(wf) == pytest.approx(0.0, abs=1e-8)
