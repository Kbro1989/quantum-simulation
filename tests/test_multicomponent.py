import numpy as np
import pytest
from quantumlab.core.grid import Grid1D, Grid2D
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D
from quantumlab.potentials import HarmonicTrap, DipoleDrive
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.solvers.multicomponent import CoupledGPESolver
from quantumlab.stationary import ground_state, coupled_ground_state
from quantumlab.observables import (
    coupled_gp_energy, miscibility_overlap, kinetic_energy_expectation, potential_energy_expectation,
)


def _gauss2d(grid, x0, y0=0.0, norm=1.0, kx=0.0, ky=0.0, s=1.0):
    wf = WaveFunction2D.gaussian(grid, x0, y0, kx, ky, s, s)
    wf.psi *= np.sqrt(norm)
    return wf


def test_uncoupled_components_match_independent_solvers():
    grid = Grid2D(48, -7.0, 7.0, 48, -7.0, 7.0)
    t1, t2 = HarmonicTrap((1.0, 1.2)), HarmonicTrap((1.3, 0.9))
    a = _gauss2d(grid, 0.5, kx=0.3)
    b = _gauss2d(grid, -0.4, 0.2, norm=0.7, ky=0.2)
    c = CoupledGPESolver(grid, (t1, t2), [[20.0, 0.0], [0.0, 35.0]], 0.01, order=4, rotation=0.3).evolve((a, b), 60)
    r1 = SplitOperatorSolver(grid, t1, 0.01, order=4, nonlinearity=20.0, rotation=0.3).evolve(a, 60)
    r2 = SplitOperatorSolver(grid, t2, 0.01, order=4, nonlinearity=35.0, rotation=0.3).evolve(b, 60)
    np.testing.assert_allclose(c[0].psi, r1.psi, atol=1e-13)
    np.testing.assert_allclose(c[1].psi, r2.psi, atol=1e-13)


def test_manakov_point_total_density_equals_single_component():
    """For g11 = g22 = g12 = g the total density obeys the one-component GP equation."""
    grid = Grid2D(64, -8.0, 8.0, 64, -8.0, 8.0)
    trap, g = HarmonicTrap((1.0, 1.3)), 60.0
    (s1, s2), info = coupled_ground_state(grid, (trap, trap), [[g, g], [g, g]],
                                          (_gauss2d(grid, 1.0, norm=0.3), _gauss2d(grid, -1.0, 0.5, norm=0.7)),
                                          dtau=0.005)
    single, _ = ground_state(grid, trap, dtau=0.005, nonlinearity=g)
    assert info.converged
    np.testing.assert_allclose(s1.probability_density + s2.probability_density, single.probability_density,
                               atol=1e-6)
    assert info.chemical_potentials[0] == pytest.approx(info.chemical_potentials[1], rel=1e-6)
    assert (s1.norm(), s2.norm()) == (pytest.approx(0.3, abs=1e-12), pytest.approx(0.7, abs=1e-12))


@pytest.fixture
def box():
    return Grid1D(256, -20 * np.pi, 20 * np.pi)       # periodic, dk = 0.05


def _spin_seeded(grid, k, n=1.0, eps=1e-6):
    return (WaveFunction1D(grid, np.sqrt(n) * (1 + eps * np.cos(k * grid.x)) + 0j),
            WaveFunction1D(grid, np.sqrt(n) * (1 - eps * np.cos(k * grid.x)) + 0j))


def _spin_amplitude(states, k):
    s = states[0].probability_density - states[1].probability_density
    return float(np.sum(s * np.cos(k * states[0].grid.x)) * 2 / len(s))


@pytest.mark.parametrize('g12, kk', [(0.6, 10), (0.3, 25)])
def test_bogoliubov_spin_mode_frequency(box, g12, kk):
    """Miscible mixture: δ(n1−n2) ∝ cos(ω_s t), ω_s = √(ε_k(ε_k + 2n(g − g12)))."""
    n, g, dt = 1.0, 1.0, 0.01
    k = box.dk * kk
    ek = k ** 2 / 2
    omega = np.sqrt(ek * (ek + 2 * n * (g - g12)))
    solver = CoupledGPESolver(box, (None, None), [[g, g12], [g12, g]], dt, order=4)
    state = _spin_seeded(box, k)
    for _ in range(6):
        state = solver.evolve(state, 500)
        assert _spin_amplitude(state, k) == pytest.approx(4e-6 * np.cos(omega * solver.t), abs=1e-12)


@pytest.mark.parametrize('g12', [1.1, 1.5])
def test_phase_separation_growth_rate(box, g12):
    """Immiscible mixture: the spin mode at k* grows as exp(γt), γ = n(g12 − g)."""
    n, g, dt = 1.0, 1.0, 0.01
    k = box.dk * np.round(np.sqrt(2 * n * (g12 - g)) / box.dk)
    ek = k ** 2 / 2
    gamma = np.sqrt(ek * (2 * n * (g12 - g) - ek))
    solver = CoupledGPESolver(box, (None, None), [[g, g12], [g12, g]], dt, order=4)
    state = _spin_seeded(box, k)
    t, amp = [], []
    while not amp or abs(amp[-1]) < 1e-3:
        state = solver.evolve(state, 100)
        t.append(solver.t)
        amp.append(abs(_spin_amplitude(state, k)))
    t, amp = np.array(t), np.array(amp)
    m = amp > 30 * 4e-6
    assert np.polyfit(t[m], np.log(amp[m]), 1)[0] == pytest.approx(gamma, rel=1e-3)


def test_energy_and_populations_conserved_in_real_time():
    grid = Grid2D(64, -8.0, 8.0, 64, -8.0, 8.0)
    traps = (HarmonicTrap((1.0, 1.1)), HarmonicTrap((1.05, 0.95), center=(0.3, 0.0)))
    G = [[100.0, 70.0], [70.0, 100.0]]
    state = (_gauss2d(grid, 1.0, norm=0.5, kx=0.4), _gauss2d(grid, -1.0, norm=0.5, ky=-0.3))
    E0 = coupled_gp_energy(state, traps, G)
    out = CoupledGPESolver(grid, traps, G, 0.005, order=4).evolve(state, 300)
    assert coupled_gp_energy(out, traps, G) == pytest.approx(E0, rel=1e-9)
    assert out[0].norm() == pytest.approx(0.5, abs=1e-12)
    assert out[1].norm() == pytest.approx(0.5, abs=1e-12)


def test_coupled_virial_theorem():
    """Stationary states in harmonic traps obey 2⟨T⟩ − 2⟨V⟩ + d·E_int = 0 (total, all components)."""
    grid = Grid2D(80, -9.0, 9.0, 80, -9.0, 9.0)
    trap = HarmonicTrap(1.0)
    G = np.array([[100.0, 70.0], [70.0, 120.0]])
    states, info = coupled_ground_state(grid, (trap, trap), G, (_gauss2d(grid, 1.5, norm=0.5), _gauss2d(grid, -1.5, norm=0.5)),
                                        dtau=0.005)
    T = sum(kinetic_energy_expectation(s) for s in states)
    V = sum(potential_energy_expectation(s, trap) for s in states)
    n = [s.probability_density for s in states]
    E_int = 0.5 * sum(G[i, j] * np.sum(n[i] * n[j]) for i in range(2) for j in range(2)) * grid.dx * grid.dy
    assert info.converged
    assert abs(2 * T - 2 * V + 2 * E_int) / V < 1e-4


def test_miscibility_overlap():
    grid = Grid2D(64, -8.0, 8.0, 64, -8.0, 8.0)
    a = _gauss2d(grid, 0.0)
    assert miscibility_overlap(a, a) == pytest.approx(1.0)
    far = _gauss2d(grid, 5.0)
    assert miscibility_overlap(_gauss2d(grid, -5.0), far) < 1e-6


def test_invalid_coupled_configurations():
    grid = Grid1D(32, -4.0, 4.0)
    with pytest.raises(ValueError):
        CoupledGPESolver(grid, (None, None), [[1.0, 0.5], [0.4, 1.0]], 0.01)       # not symmetric
    with pytest.raises(ValueError):
        CoupledGPESolver(grid, (None, None), [[1.0]], 0.01)                        # wrong shape
    with pytest.raises(ValueError):
        CoupledGPESolver(grid, (DipoleDrive(None, 1.0, 1.0), None), np.eye(2), 0.01)
    solver = CoupledGPESolver(grid, (None, None), np.eye(2), 0.01)
    with pytest.raises(ValueError):
        solver.step((WaveFunction1D.gaussian(grid, 0.0, 0.0, 1.0),))
