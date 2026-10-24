import numpy as np
import pytest
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.potentials import HarmonicTrap, StadiumBilliard
from quantumlab.stationary import eigenstates, ground_state, excited_states, apply_hamiltonian
from quantumlab.core.spectral import volume_element


def _overlap_matrix(states):
    dV = volume_element(states[0].grid)
    return np.array([[np.sum(np.conj(a.psi) * b.psi) * dV for b in states] for a in states])


def test_1d_harmonic_spectrum_lanczos():
    grid = Grid1D(256, -12.0, 12.0)
    energies, states = eigenstates(grid, HarmonicTrap(1.3), n_states=8, method='lanczos')
    np.testing.assert_allclose(energies, 1.3 * (np.arange(8) + 0.5), atol=1e-10)
    np.testing.assert_allclose(_overlap_matrix(states), np.eye(8), atol=1e-10)


def test_2d_anisotropic_spectrum_lobpcg():
    grid = Grid2D(96, -9.0, 9.0, 96, -9.0, 9.0)
    wx, wy = 1.0, 1.5
    energies, _ = eigenstates(grid, HarmonicTrap((wx, wy)), n_states=6, method='lobpcg')
    levels = sorted(wx * (nx + 0.5) + wy * (ny + 0.5) for nx in range(6) for ny in range(6))[:6]
    np.testing.assert_allclose(energies, levels, atol=1e-8)


def test_3d_isotropic_degeneracy():
    grid = Grid3D(32, -7.0, 7.0, 32, -7.0, 7.0, 32, -7.0, 7.0)
    energies, states = eigenstates(grid, HarmonicTrap(1.0), n_states=4)
    np.testing.assert_allclose(energies, [1.5, 2.5, 2.5, 2.5], atol=1e-8)
    for E, wf in zip(energies, states):
        H_psi = apply_hamiltonian(wf.psi, grid, HarmonicTrap(1.0).evaluate(grid))
        assert np.max(np.abs(H_psi - E * wf.psi)) < 1e-4


def test_lanczos_and_lobpcg_agree_on_chaotic_billiard():
    grid = Grid2D(64, -6.0, 6.0, 64, -4.0, 4.0)
    pot = StadiumBilliard(V0=40.0, radius=2.5, length=3.0, smoothing=0.3)
    e_l, _ = eigenstates(grid, pot, n_states=5, method='lanczos')
    e_b, _ = eigenstates(grid, pot, n_states=5, method='lobpcg', tol=1e-7)
    np.testing.assert_allclose(e_l, e_b, rtol=1e-9)


def test_imaginary_time_ground_state_3d():
    grid = Grid3D(32, -7.0, 7.0, 32, -7.0, 7.0, 32, -7.0, 7.0)
    wf, info = ground_state(grid, HarmonicTrap((1.0, 1.2, 1.4)), dtau=0.01)
    assert info.converged
    assert info.energy == pytest.approx(0.5 * (1.0 + 1.2 + 1.4), abs=1e-7)
    assert wf.norm() == pytest.approx(1.0, abs=1e-12)


def test_imaginary_time_excited_states_match_lanczos():
    grid = Grid1D(256, -10.0, 10.0)
    pot = HarmonicTrap(1.0)
    e_it, states, infos = excited_states(grid, pot, n_states=4, dtau=0.005)
    assert all(i.converged for i in infos)
    np.testing.assert_allclose(e_it, np.arange(4) + 0.5, atol=1e-8)
    np.testing.assert_allclose(np.abs(_overlap_matrix(states)), np.eye(4), atol=1e-6)


def test_ground_state_error_is_second_order_in_dtau():
    grid = Grid1D(256, -10.0, 10.0)
    pot = HarmonicTrap(1.0)
    _, i1 = ground_state(grid, pot, dtau=0.02, tol=1e-14)
    _, i2 = ground_state(grid, pot, dtau=0.01, tol=1e-14)
    assert i1.residual / i2.residual == pytest.approx(4.0, rel=0.1)
