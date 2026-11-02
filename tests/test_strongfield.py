import numpy as np
import pytest
from quantumlab.core.grid import Grid1D, Grid2D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.potentials import SoftCoulomb, HarmonicTrap, flat_top_envelope, create_potential
from quantumlab.stationary import eigenstates
from quantumlab.observables import position_expectation
from quantumlab.strongfield import (
    wavelength_to_omega, intensity_to_field, ponderomotive_energy, classical_returns, classical_cutoff,
    harmonic_spectrum, gabor_transform, attosecond_pulses, dipole_acceleration, HARTREE_EV,
)


def test_laser_unit_conversions():
    assert wavelength_to_omega(800.0) == pytest.approx(0.0569542, rel=1e-6)    # 1.55 eV photon
    assert wavelength_to_omega(800.0) * HARTREE_EV == pytest.approx(1.5498, abs=1e-4)
    assert intensity_to_field(3.50944758e16) == pytest.approx(1.0)
    assert intensity_to_field(1e14) == pytest.approx(0.05338, abs=1e-5)
    assert ponderomotive_energy(0.05338, 0.05695) == pytest.approx(0.2196, abs=1e-4)


def test_classical_three_step_maximum():
    """Maximum return energy 3.17 U_p for birth ≈ 17° after the field crest, return ≈ 252°."""
    births, returns, energy = classical_returns(4000)
    k = np.argmax(energy)
    assert energy[k] == pytest.approx(3.1731, abs=2e-4)
    assert np.degrees(births[k]) == pytest.approx(17.0, abs=1.5)
    assert np.degrees(returns[k]) == pytest.approx(252.0, abs=1.5)
    assert np.all(returns > births)
    assert classical_cutoff(0.5, 0.05338, 0.05695) == pytest.approx(0.5 + 3.1731 * 0.2196, abs=1e-3)


def test_soft_coulomb_hydrogen_ground_state():
    energies, _ = eigenstates(Grid1D(2048, -100.0, 100.0), SoftCoulomb(), n_states=2)
    assert energies[0] == pytest.approx(-0.5, abs=1e-9)        # hydrogen I_p = 13.6 eV
    assert energies[1] == pytest.approx(-0.2329, abs=1e-4)
    assert isinstance(create_potential('soft_coulomb'), SoftCoulomb)


def test_soft_coulomb_gradient_matches_finite_difference():
    # central differences have an O(h²) error, so a fine grid is used near the sharp core
    grid = Grid2D(600, -6.0, 6.0, 600, -6.0, 6.0)
    pot = SoftCoulomb(Z=2.0, softening=0.8, center=(0.5, -1.0))
    V = pot.evaluate(grid)
    gx, gy = pot.gradient(grid)
    np.testing.assert_allclose(gx[1:-1, :], (V[2:, :] - V[:-2, :]) / (2 * grid.dx), atol=1e-3)
    np.testing.assert_allclose(gy[:, 1:-1], (V[:, 2:] - V[:, :-2]) / (2 * grid.dy), atol=1e-3)


def test_flat_top_envelope():
    env = flat_top_envelope(10.0, 2.0)
    assert [env(t) for t in (-1.0, 0.0, 10.0, 11.0)] == [0.0, 0.0, 0.0, 0.0]
    assert env(1.0) == pytest.approx(0.5)
    assert env(5.0) == 1.0
    assert env(9.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        flat_top_envelope(3.0, 2.0)


def test_harmonic_spectrum_peaks_at_harmonic_orders():
    w0, dt = 0.05, 0.05
    t = np.arange(0, 400 * 2 * np.pi / w0, dt)
    sig = np.sin(w0 * t) + 1e-3 * np.sin(5 * w0 * t)
    H, S = harmonic_spectrum(sig, dt, w0)
    assert H[np.argmax(S)] == pytest.approx(1.0, abs=0.01)
    near5 = (H > 4) & (H < 6)
    assert H[near5][np.argmax(S[near5])] == pytest.approx(5.0, abs=0.01)
    # the weak harmonic stands far above the spectral floor between the lines
    assert S[near5].max() > 1e3 * S[(H > 2.5) & (H < 3.5)].max()


def test_gabor_transform_locates_burst():
    w0, dt = 0.057, 0.05
    t = np.arange(0, 1000, dt)
    sig = np.exp(-((t - 600) / 20) ** 2) * np.cos(9 * w0 * t)
    tt, oo, G = gabor_transform(sig, dt, w0, np.linspace(1, 15, 141), width=1 / (3 * w0),
                                times=np.arange(400, 800, 2.0))
    i, j = np.unravel_index(np.argmax(G), G.shape)
    assert tt[i] == pytest.approx(600.0, abs=2.0)
    assert oo[j] == pytest.approx(9.0, abs=0.1)


def test_attosecond_pulses_from_harmonic_comb():
    """Odd harmonics of equal phase add up to one burst every half period."""
    w0, dt = 0.057, 0.05
    t = np.arange(0, 20 * 2 * np.pi / w0, dt)
    sig = sum(np.cos(q * w0 * t) for q in range(11, 22, 2))
    P = attosecond_pulses(sig, dt, w0, 10, 22)
    period = 2 * np.pi / w0
    mid = (t > 5 * period) & (t < 15 * period)
    peaks = t[mid][np.r_[False, (P[mid][1:-1] > P[mid][:-2]) & (P[mid][1:-1] > P[mid][2:]), False]
                   & (P[mid] > 0.5 * P[mid].max())]
    np.testing.assert_allclose(np.diff(peaks) / period, 0.5, atol=0.01)


def test_dipole_acceleration_obeys_ehrenfest_in_harmonic_trap():
    """For V0 = x²/2 and a constant force F: a = −⟨x⟩ + F exactly."""
    grid = Grid1D(256, -12.0, 12.0)
    wf = WaveFunction1D.gaussian(grid, 1.3, 0.4, 0.8)
    a = dipole_acceleration(wf, HarmonicTrap(1.0), force=0.25)
    assert a == pytest.approx(-position_expectation(wf) + 0.25, abs=1e-10)
