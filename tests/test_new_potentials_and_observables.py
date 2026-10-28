import numpy as np
import pytest
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D, WaveFunction3D
from quantumlab.potentials import (
    HarmonicTrap, HarmonicOscillator, ApertureScreen, StadiumBilliard, CustomPotential,
    DipoleDrive, sin2_envelope, create_potential,
)
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.observables import (
    probability_current, angular_momentum_z_expectation, autocorrelation, momentum_expectation,
)


def test_harmonic_trap_isotropic_matches_oscillator():
    grid = Grid3D(16, -4.0, 4.0, 16, -4.0, 4.0, 16, -4.0, 4.0)
    np.testing.assert_allclose(HarmonicTrap(1.7).evaluate(grid), HarmonicOscillator(1.7).evaluate(grid))


def test_harmonic_trap_anisotropic_and_centered():
    grid = Grid2D(16, -4.0, 4.0, 16, -4.0, 4.0)
    V = HarmonicTrap((1.0, 2.0), center=(0.5, -1.0)).evaluate(grid)
    np.testing.assert_allclose(V, 0.5 * (grid.X - 0.5) ** 2 + 2.0 * (grid.Y + 1.0) ** 2)


def test_aperture_screen_geometry_3d_pinholes():
    grid = Grid3D(64, -8.0, 8.0, 64, -8.0, 8.0, 64, -8.0, 8.0)
    screen = ApertureScreen(V0=50.0, wall_position=0.0, wall_thickness=1.0, separation=4.0,
                            aperture_width=2.0, aperture='pinhole', smoothing=0.1)
    V = screen.evaluate(grid)
    ix = np.argmin(np.abs(grid.x))
    iy_hole = np.argmin(np.abs(grid.y - 2.0))
    iz0 = np.argmin(np.abs(grid.z))
    assert V[ix, iy_hole, iz0] < 1e-6                      # centre of a pinhole is open
    assert V[ix, np.argmin(np.abs(grid.y)), iz0] > 49.0   # between the holes is wall
    assert V[ix, iy_hole, np.argmin(np.abs(grid.z - 3.0))] > 49.0  # above the hole is wall
    assert V[np.argmin(np.abs(grid.x - 3.0)), 0, 0] < 1e-6  # away from the wall is free
    np.testing.assert_allclose(screen.aperture_centers, [-2.0, 2.0])


def test_aperture_screen_slits_are_translation_invariant_in_z():
    grid = Grid3D(32, -8.0, 8.0, 32, -8.0, 8.0, 16, -4.0, 4.0)
    V = ApertureScreen(V0=10.0, aperture='slit', n_apertures=3).evaluate(grid)
    np.testing.assert_allclose(V, np.repeat(V[:, :, :1], 16, axis=2))


def test_stadium_is_open_inside_and_closed_outside():
    grid = Grid2D(128, -8.0, 8.0, 128, -6.0, 6.0)
    V = StadiumBilliard(V0=30.0, radius=2.0, length=4.0, smoothing=0.1).evaluate(grid)
    # tanh wall tail 5 smoothing lengths inside the boundary: 30·(1 − tanh 5)/2 ≈ 1.4e-3
    assert V[np.argmin(np.abs(grid.x - 3.5)), np.argmin(np.abs(grid.y))] < 2e-3   # inside the cap
    assert V[np.argmin(np.abs(grid.x)), np.argmin(np.abs(grid.y))] < 1e-6        # centre
    assert V[np.argmin(np.abs(grid.x)), np.argmin(np.abs(grid.y - 2.5))] > 29.0    # above the flat side
    assert V[np.argmin(np.abs(grid.x - 4.5)), np.argmin(np.abs(grid.y - 1.5))] > 29.0  # outside the cap


def test_custom_potential_3d():
    grid = Grid3D(8, -1.0, 1.0, 8, -1.0, 1.0, 8, -1.0, 1.0)
    V = CustomPotential(lambda x, y, z: x * y * z).evaluate(grid)
    np.testing.assert_allclose(V, grid.X * grid.Y * grid.Z)


def test_registry_has_new_potentials():
    assert isinstance(create_potential('harmonic_trap', omegas=1.0), HarmonicTrap)
    assert isinstance(create_potential('double_slit', V0=1.0), ApertureScreen)
    assert isinstance(create_potential('stadium', V0=1.0, radius=1.0, length=1.0), StadiumBilliard)


def test_sin2_envelope_and_dipole_force():
    env = sin2_envelope(10.0)
    assert env(-1.0) == 0.0 and env(10.0) == 0.0
    assert env(5.0) == pytest.approx(1.0)
    drive = DipoleDrive(None, F0=2.0, omega=1.0, envelope=env)
    assert drive.force(5.0) == pytest.approx(2.0 * np.sin(5.0))
    grid = Grid1D(8, -1.0, 1.0)
    np.testing.assert_allclose(drive.evaluate_at(grid, 5.0), -drive.force(5.0) * grid.x)


def test_dipole_drive_rejects_missing_axis():
    with pytest.raises(ValueError):
        DipoleDrive(None, 1.0, 1.0, axis='z').evaluate_at(Grid1D(8, -1.0, 1.0), 0.0)


def test_probability_current_of_moving_gaussian():
    grid = Grid1D(512, -25.0, 25.0)
    wf = WaveFunction1D.gaussian(grid, 0.0, 2.0, 2.0)
    (j,) = probability_current(wf)
    np.testing.assert_allclose(j, 2.0 * wf.probability_density, atol=1e-12)


def test_probability_current_integrates_to_momentum_3d():
    grid = Grid3D(32, -8.0, 8.0, 32, -8.0, 8.0, 32, -8.0, 8.0)
    wf = WaveFunction3D.gaussian(grid, 0.0, 0.0, 0.0, 1.0, -0.5, 0.3, 1.2, 1.2, 1.2)
    dV = grid.dx * grid.dy * grid.dz
    for j, axis in zip(probability_current(wf), 'xyz'):
        assert np.sum(j) * dV == pytest.approx(momentum_expectation(wf, axis), abs=1e-10)


def test_vortex_angular_momentum():
    grid = Grid2D(128, -8.0, 8.0, 128, -8.0, 8.0)
    for ell in (1, 2, -1):
        z = grid.X + 1j * np.sign(ell) * grid.Y
        wf = WaveFunction2D(grid, z ** abs(ell) * np.exp(-(grid.X ** 2 + grid.Y ** 2) / 2)).normalize()
        assert angular_momentum_z_expectation(wf) == pytest.approx(float(ell), abs=1e-10)


def test_autocorrelation_spectrum_peaks_at_eigenenergies():
    """Feit–Fleck spectral method: FFT of C(t) peaks at E_n = n + ½."""
    grid = Grid1D(256, -12.0, 12.0)
    trap = HarmonicTrap(1.0)
    wf0 = WaveFunction1D.gaussian(grid, 1.5, 0.0, 0.7)
    solver = SplitOperatorSolver(grid, trap, 0.05, order=4)
    n_steps = 4096
    C = np.empty(n_steps, dtype=complex)
    wf = wf0
    for n in range(n_steps):
        C[n] = autocorrelation(wf0.psi, wf.psi, grid)
        wf = solver.step(wf)
    window = np.hanning(n_steps)
    spectrum = np.abs(np.fft.fft(C * window))
    E = -2.0 * np.pi * np.fft.fftfreq(n_steps, d=0.05)
    peaks = sorted(E[i] for i in np.argsort(spectrum)[::-1][:60] if 0 < E[i] < 4)
    for level in (0.5, 1.5, 2.5, 3.5):
        assert min(abs(p - level) for p in peaks) < 2 * np.pi / (n_steps * 0.05)


def test_vortex_positions_finds_imprinted_vortices():
    from quantumlab.observables import vortex_positions
    grid = Grid2D(128, -8.0, 8.0, 128, -8.0, 8.0)
    centers = [(-2.0, 1.0, 1), (2.5, -1.5, 1), (0.5, 2.5, -1)]
    psi = np.exp(-(grid.X ** 2 + grid.Y ** 2) / 32).astype(complex)
    for cx, cy, q in centers:
        z = (grid.X - cx) + 1j * q * (grid.Y - cy)
        psi *= z / np.sqrt(np.abs(z) ** 2 + 0.25)
    pos, charge = vortex_positions(WaveFunction2D(grid, psi))
    assert len(pos) == 3
    for cx, cy, q in centers:
        k = np.argmin(np.hypot(pos[:, 0] - cx, pos[:, 1] - cy))
        assert np.hypot(pos[k, 0] - cx, pos[k, 1] - cy) < grid.dx
        assert charge[k] == q
