import numpy as np
import pytest
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D, WaveFunction3D
from quantumlab.potentials import (
    GaussianBarrier, Gaussian2DBarrier, Gaussian3DBarrier, CustomPotential,
    HarmonicTrap, DipoleDrive, TimeDependentCustomPotential,
)
from quantumlab.solvers import SplitStep1DSolver, SplitStep2DSolver, SplitStep3DSolver
from quantumlab.solvers.split_operator import SplitOperatorSolver, YOSHIDA_W0, YOSHIDA_W1
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.observables import position_expectation, total_energy_expectation


def _l2(a, b, dx):
    return float(np.sqrt(np.sum(np.abs(a - b) ** 2) * dx))


def test_yoshida_coefficients_sum_to_one():
    assert 2 * YOSHIDA_W1 + YOSHIDA_W0 == pytest.approx(1.0, abs=1e-15)
    # Third-order condition of the triple jump: 2 w1³ + w0³ = 0
    assert 2 * YOSHIDA_W1 ** 3 + YOSHIDA_W0 ** 3 == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_order2_matches_legacy_solvers(dim):
    if dim == 1:
        grid = Grid1D(256, -20.0, 20.0)
        wf = WaveFunction1D.gaussian(grid, -5.0, 2.0, 1.0)
        pot, legacy = GaussianBarrier(4.0, 1.0, 0.0), SplitStep1DSolver
    elif dim == 2:
        grid = Grid2D(64, -10.0, 10.0, 64, -10.0, 10.0)
        wf = WaveFunction2D.gaussian(grid, -3.0, 0.0, 2.0, 0.0, 1.0, 1.0)
        pot, legacy = Gaussian2DBarrier(3.0, 1.0), SplitStep2DSolver
    else:
        grid = Grid3D(24, -8.0, 8.0, 24, -8.0, 8.0, 24, -8.0, 8.0)
        wf = WaveFunction3D.gaussian(grid, -2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.0, 1.0, 1.0)
        pot, legacy = Gaussian3DBarrier(3.0, 1.0), SplitStep3DSolver
    a = legacy(grid, pot, 0.02)
    b = SplitOperatorSolver(grid, pot, 0.02, order=2)
    wa = wb = wf
    for _ in range(20):
        wa = a.step(wa)
        wb = b.step(wb)
    np.testing.assert_allclose(wb.psi, wa.psi, atol=1e-12)
    assert type(wb) is type(wf)


@pytest.mark.parametrize('order, energy_rtol', [(2, 1e-5), (4, 1e-8)])
def test_norm_and_energy_conservation(order, energy_rtol):
    grid = Grid1D(512, -30.0, 30.0)
    pot = GaussianBarrier(3.0, 1.0, 0.0)
    wf = WaveFunction1D.gaussian(grid, -8.0, 2.0, 1.5)
    solver = SplitOperatorSolver(grid, pot, 0.01, order=order)
    E0 = total_energy_expectation(wf, pot)
    wf = solver.evolve(wf, 300)
    assert wf.norm() == pytest.approx(1.0, abs=1e-12)
    # Splitting methods conserve a modified energy; the drift is O(dt^order)
    assert total_energy_expectation(wf, pot) == pytest.approx(E0, rel=energy_rtol)
    assert solver.t == pytest.approx(3.0)


@pytest.mark.parametrize('order', [2, 4])
def test_convergence_order(order):
    """Halving dt reduces the global error by 2^order."""
    grid = Grid1D(256, -12.0, 12.0)
    pot = CustomPotential(lambda x: 0.1 * x ** 4 - x ** 2 + 2.0 * np.exp(-x ** 2))
    wf0 = WaveFunction1D.gaussian(grid, -2.0, 1.5, 0.8)
    T = 2.0

    def run(o, dt):
        return SplitOperatorSolver(grid, pot, dt, order=o).evolve(wf0, int(round(T / dt))).psi

    ref = run(4, 0.0005)
    errors = [_l2(run(order, dt), ref, grid.dx) for dt in (0.04, 0.02, 0.01)]
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    np.testing.assert_allclose(rates, order, atol=0.1)


def test_order4_more_accurate_than_order2():
    grid = Grid1D(256, -12.0, 12.0)
    pot = CustomPotential(lambda x: 0.1 * x ** 4 - x ** 2)
    wf0 = WaveFunction1D.gaussian(grid, -2.0, 1.0, 0.8)
    ref = SplitOperatorSolver(grid, pot, 0.0005, order=4).evolve(wf0, 4000).psi
    e2 = _l2(SplitOperatorSolver(grid, pot, 0.02, order=2).evolve(wf0, 100).psi, ref, grid.dx)
    e4 = _l2(SplitOperatorSolver(grid, pot, 0.02, order=4).evolve(wf0, 100).psi, ref, grid.dx)
    assert e4 < e2 / 50


def test_driven_oscillator_follows_classical_trajectory():
    """Ehrenfest's theorem is exact for quadratic + linear potentials."""
    omega, F0, Omega = 1.0, 0.5, 0.6
    pot = DipoleDrive(HarmonicTrap(omega), F0=F0, omega=Omega)
    grid = Grid1D(512, -30.0, 30.0)
    wf = WaveFunction1D.gaussian(grid, 0.0, 0.0, 1.0 / np.sqrt(2.0))
    solver = SplitOperatorSolver(grid, pot, 0.01, order=4)
    for _ in range(10):
        wf = solver.evolve(wf, 100)
        t = solver.t
        x_cl = F0 / (omega ** 2 - Omega ** 2) * (np.sin(Omega * t) - (Omega / omega) * np.sin(omega * t))
        assert position_expectation(wf) == pytest.approx(x_cl, abs=1e-7)


def test_time_dependent_custom_potential_matches_dipole_drive():
    grid = Grid1D(256, -20.0, 20.0)
    wf0 = WaveFunction1D.gaussian(grid, 0.0, 0.0, 1.0)
    a = DipoleDrive(HarmonicTrap(1.0), F0=0.3, omega=1.7)
    b = TimeDependentCustomPotential(lambda x, t: 0.5 * x ** 2 - 0.3 * np.sin(1.7 * t) * x)
    wa = SplitOperatorSolver(grid, a, 0.01).evolve(wf0, 200)
    wb = SplitOperatorSolver(grid, b, 0.01).evolve(wf0, 200)
    np.testing.assert_allclose(wa.psi, wb.psi, atol=1e-12)


def test_free_particle_matches_analytic_gaussian_spreading():
    grid = Grid1D(1024, -60.0, 60.0)
    sigma0, k0, t = 1.0, 1.0, 5.0
    wf = WaveFunction1D.gaussian(grid, 0.0, k0, sigma0)
    wf = SplitOperatorSolver(grid, None, 0.05).evolve(wf, int(round(t / 0.05)))
    sigma_t = sigma0 * np.sqrt(1.0 + (t / (2.0 * sigma0 ** 2)) ** 2)
    exact = np.exp(-(grid.x - k0 * t) ** 2 / (2.0 * sigma_t ** 2)) / np.sqrt(2.0 * np.pi * sigma_t ** 2)
    np.testing.assert_allclose(wf.probability_density, exact, atol=1e-10)


def test_absorber_removes_outgoing_flux():
    grid = Grid1D(512, -20.0, 20.0)
    wf = WaveFunction1D.gaussian(grid, 0.0, 4.0, 1.0)
    free = SplitOperatorSolver(grid, None, 0.01).evolve(wf, 1000)
    absorbed = SplitOperatorSolver(grid, None, 0.01, absorber=AbsorbingBoundaryLayer(grid, 4.0)).evolve(wf, 1000)
    assert free.norm() == pytest.approx(1.0, abs=1e-12)
    # A finite-width mask absorbs most but not all of the outgoing packet
    assert absorbed.norm() < 0.01


def test_invalid_configurations_raise():
    grid = Grid1D(64, -5.0, 5.0)
    with pytest.raises(ValueError):
        SplitOperatorSolver(grid, None, 0.01, order=3)
    with pytest.raises(ValueError):
        SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.01, order=4, imaginary_time=True)
    with pytest.raises(ValueError):
        SplitOperatorSolver(grid, DipoleDrive(None, 1.0, 1.0), 0.01, imaginary_time=True)


def test_evolve_callback_schedule():
    grid = Grid1D(64, -5.0, 5.0)
    wf = WaveFunction1D.gaussian(grid, 0.0, 0.0, 1.0)
    calls = []
    SplitOperatorSolver(grid, None, 0.1).evolve(wf, 10, callback=lambda n, t, w: calls.append((n, t)),
                                               callback_interval=5)
    assert [c[0] for c in calls] == [5, 10]
    assert calls[-1][1] == pytest.approx(1.0)
