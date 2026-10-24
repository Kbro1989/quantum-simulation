"""
Strong-field and attosecond physics in atomic units (ℏ = mₑ = e = 1).

Laser-parameter conversions, the classical three-step ("simple-man") model
of high-harmonic generation, and the spectral tools used to analyse a
time-dependent dipole: harmonic spectra, Gabor time–frequency maps and
attosecond pulses synthesised from a band of harmonics.
"""
import numpy as np
from scipy.optimize import brentq

HARTREE_EV = 27.211386245988
AU_TIME_FS = 0.024188843265857
AU_INTENSITY_W_CM2 = 3.50944758e16     # intensity of a field of amplitude 1 a.u.


def wavelength_to_omega(wavelength_nm: float) -> float:
    """Photon angular frequency (a.u.) for a vacuum wavelength in nm."""
    return 45.56335253 / wavelength_nm


def intensity_to_field(intensity_w_cm2: float) -> float:
    """Peak electric field (a.u.) of a linearly polarised beam of given peak intensity."""
    return float(np.sqrt(intensity_w_cm2 / AU_INTENSITY_W_CM2))


def ponderomotive_energy(field: float, omega: float) -> float:
    """U_p = E₀² / (4ω²), the cycle-averaged quiver energy of a free electron."""
    return field ** 2 / (4.0 * omega ** 2)


def classical_returns(n_phases: int = 2000):
    """
    Three-step model in a field E(t) = E₀ cos(ωt): an electron born at rest at
    phase φ₀ = ωt₀ moves as x ∝ cos φ − cos φ₀ + (φ − φ₀) sin φ₀ and returns
    to the core at the first φ_r > φ₀ where x = 0, with kinetic energy
    2U_p (sin φ_r − sin φ₀)².

    Returns
    -------
    birth_phase, return_phase : ndarray
        φ₀ and φ_r in radians, for the trajectories that return.
    return_energy : ndarray
        Kinetic energy at return in units of U_p; its maximum is ≈ 3.17.
    """
    def x(phi, phi0):
        return np.cos(phi) - np.cos(phi0) + (phi - phi0) * np.sin(phi0)

    births, returns, energies = [], [], []
    # Only electrons born in the quarter cycle after the field crest return
    for phi0 in np.linspace(1e-4, np.pi / 2 - 1e-4, n_phases):
        grid = np.linspace(phi0 + 1e-6, phi0 + 2.5 * np.pi, 4000)
        values = x(grid, phi0)
        crossing = np.nonzero(np.sign(values[:-1]) != np.sign(values[1:]))[0]
        if crossing.size == 0:
            continue
        k = crossing[0]
        phi_r = brentq(x, grid[k], grid[k + 1], args=(phi0,))
        births.append(phi0)
        returns.append(phi_r)
        energies.append(2.0 * (np.sin(phi_r) - np.sin(phi0)) ** 2)
    return np.array(births), np.array(returns), np.array(energies)


def classical_cutoff(ionisation_potential: float, field: float, omega: float) -> float:
    """Classical HHG cutoff energy I_p + max(E_return), i.e. I_p + 3.17 U_p (a.u.)."""
    return ionisation_potential + classical_returns()[2].max() * ponderomotive_energy(field, omega)


def harmonic_spectrum(signal: np.ndarray, dt: float, omega0: float, window: bool = True):
    """
    Power spectrum |FFT|² of a dipole signal (acceleration or moment) versus
    harmonic order ω/ω₀. A Hann window suppresses spectral leakage.
    """
    s = np.asarray(signal, dtype=float)
    if window:
        s = s * np.hanning(len(s))
    spectrum = np.abs(np.fft.rfft(s)) ** 2
    omega = 2.0 * np.pi * np.fft.rfftfreq(len(s), dt)
    return omega / omega0, spectrum


def gabor_transform(signal: np.ndarray, dt: float, omega0: float, orders, width: float,
                    times=None):
    """
    Time–frequency map |∫ a(t') e^{−(t'−t)²/2σ²} e^{−iωt'} dt'|² on a grid of
    harmonic orders, with Gaussian window width σ = ``width`` (a.u. of time).
    The time resolution is σ and the frequency resolution 1/σ. Only the
    samples within ±6σ of each centre contribute (the window is < 2e-8 beyond).
    Returns (times, orders, map[time, order]).
    """
    s = np.asarray(signal, dtype=float)
    t = np.arange(len(s)) * dt
    times = t[::max(1, len(s) // 400)] if times is None else np.asarray(times, dtype=float)
    omegas = np.asarray(orders, dtype=float) * omega0
    half = int(np.ceil(6.0 * width / dt))
    out = np.empty((len(times), len(omegas)))
    for i, tc in enumerate(times):
        k = int(round(tc / dt))
        lo, hi = max(0, k - half), min(len(s), k + half + 1)
        ts = t[lo:hi]
        windowed = np.exp(-0.5 * ((ts - tc) / width) ** 2) * s[lo:hi]
        out[i] = np.abs(windowed @ np.exp(-1j * np.outer(ts, omegas)) * dt) ** 2
    return times, np.asarray(orders, dtype=float), out


def attosecond_pulses(signal: np.ndarray, dt: float, omega0: float, order_min: float, order_max: float):
    """
    Intensity envelope |a_filtered(t)|² of the band order_min ≤ ω/ω₀ ≤ order_max,
    obtained by filtering the spectrum and transforming back (analytic signal).
    """
    s = np.asarray(signal, dtype=float)
    spec = np.fft.fft(s)
    orders = 2.0 * np.pi * np.fft.fftfreq(len(s), dt) / omega0
    keep = (orders >= order_min) & (orders <= order_max)      # positive band only → analytic signal
    return np.abs(np.fft.ifft(np.where(keep, spec, 0.0))) ** 2


def dipole_acceleration(wf, base_potential, force: float) -> float:
    """
    Dipole acceleration a(t) = ⟨−∂V₀/∂x⟩ + F(t) for H = p²/2 + V₀(x) − F(t)x
    (Ehrenfest's theorem). Uses the potential's analytic ``gradient`` when it
    has one, otherwise a spectral derivative of V₀.
    """
    grid = wf.grid
    if hasattr(base_potential, 'gradient'):
        dV = base_potential.gradient(grid)[0]
    else:
        from quantumlab.observables.current import gradient
        dV = np.real(gradient(base_potential.evaluate(grid).astype(complex), grid)[0])
    from quantumlab.core.spectral import volume_element
    return float(-np.sum(dV * wf.probability_density) * volume_element(grid) + force)
