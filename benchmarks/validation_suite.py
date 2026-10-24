"""
Validation suite: every check compares a QuantumLab result with an exact,
analytic or independent numerical reference, and the measured errors are
written to benchmarks/validation_report.md.

Run from the project root:
    python benchmarks/validation_suite.py
"""
import platform
import sys
import time
from pathlib import Path
import numpy as np
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D, WaveFunction3D
from quantumlab.potentials import (HarmonicTrap, DipoleDrive, CustomPotential, GaussianBarrier, SoftCoulomb,
                                  flat_top_envelope)
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab import strongfield as sf
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.stationary import eigenstates, ground_state
from quantumlab.observables import (
    position_expectation, position_uncertainty, momentum_uncertainty, y_uncertainty, z_uncertainty,
    momentum_expectation, kinetic_energy_expectation, y_expectation, potential_energy_expectation, interaction_energy,
    angular_momentum_z_expectation, autocorrelation, total_energy_expectation,
)

RESULTS = []


def check(category, name, reference, measured_desc, error, tolerance):
    passed = error <= tolerance
    RESULTS.append((category, name, reference, measured_desc, error, tolerance, passed))
    print(f'  [{"PASS" if passed else "FAIL"}] {name}: error {error:.2e} (tolerance {tolerance:.0e})')


def l2(a, b, dV):
    return float(np.sqrt(np.sum(np.abs(a - b) ** 2) * dV))


def free_spreading():
    grid = Grid1D(1024, -60.0, 60.0)
    sigma0, k0, t = 1.0, 1.0, 5.0
    wf = SplitOperatorSolver(grid, None, 0.05).evolve(WaveFunction1D.gaussian(grid, 0.0, k0, sigma0), 100)
    sigma_t = sigma0 * np.sqrt(1 + (t / (2 * sigma0 ** 2)) ** 2)
    exact = np.exp(-(grid.x - k0 * t) ** 2 / (2 * sigma_t ** 2)) / np.sqrt(2 * np.pi * sigma_t ** 2)
    check('Dynamics', 'Free Gaussian spreading (1D)', 'σ(t) = σ₀√(1+(t/2σ₀²)²), exact density',
          'max |ρ − ρ_exact|', float(np.max(np.abs(wf.probability_density - exact))), 1e-9)


def convergence_orders():
    grid = Grid1D(256, -12.0, 12.0)
    pot = CustomPotential(lambda x: 0.1 * x ** 4 - x ** 2 + 2.0 * np.exp(-x ** 2))
    wf0 = WaveFunction1D.gaussian(grid, -2.0, 1.5, 0.8)

    def run(order, dt):
        return SplitOperatorSolver(grid, pot, dt, order=order).evolve(wf0, int(round(2.0 / dt))).psi

    ref = run(4, 0.0005)
    for order in (2, 4):
        errs = [l2(run(order, dt), ref, grid.dx) for dt in (0.04, 0.02, 0.01)]
        rate = float(np.mean(np.log2(np.array(errs[:-1]) / np.array(errs[1:]))))
        check('Dynamics', f'Temporal convergence, {"Strang" if order == 2 else "Yoshida"} splitting',
              f'global error ∝ dt^{order}', f'observed order {rate:.3f}', abs(rate - order), 0.05)


def driven_oscillator():
    omega, F0, Omega = 1.0, 0.5, 0.6
    grid = Grid1D(512, -30.0, 30.0)
    solver = SplitOperatorSolver(grid, DipoleDrive(HarmonicTrap(omega), F0, Omega), 0.01, order=4)
    wf = WaveFunction1D.gaussian(grid, 0.0, 0.0, 1 / np.sqrt(2))
    err = 0.0
    for _ in range(20):
        wf = solver.evolve(wf, 100)
        t = solver.t
        x_cl = F0 / (omega ** 2 - Omega ** 2) * (np.sin(Omega * t) - Omega / omega * np.sin(omega * t))
        err = max(err, abs(position_expectation(wf) - x_cl))
    check('Dynamics', 'Driven harmonic oscillator, t ≤ 20', 'Ehrenfest: ⟨x⟩ = classical forced trajectory',
          'max |⟨x⟩ − x_cl|', err, 1e-7)


def norm_energy_3d():
    grid = Grid3D(48, -10.0, 10.0, 48, -10.0, 10.0, 48, -10.0, 10.0)
    pot = HarmonicTrap((1.0, 1.3, 0.8))
    wf = WaveFunction3D.gaussian(grid, 1.0, -0.5, 0.0, 1.0, 0.5, -0.3, 1.0, 0.8, 1.1)
    E0 = total_energy_expectation(wf, pot)
    n_steps = 500
    wf = SplitOperatorSolver(grid, pot, 0.01, order=4).evolve(wf, n_steps)
    # Every step is unitary in exact arithmetic, so the norm can only drift by
    # floating-point roundoff: ~ (FFT pairs) × ε × log2(grid size). The actual
    # value depends on the CPU's rounding (it differs between ARM and x86).
    fft_pairs = 3 * n_steps                         # Yoshida: 3 Strang sub-steps per step
    roundoff = fft_pairs * np.finfo(float).eps * np.log2(np.prod(grid.shape))
    check('Dynamics', 'Norm conservation, 3D, 500 steps', 'unitary evolution: ‖ψ‖ = 1 up to roundoff',
          '|‖ψ‖ − 1|', abs(wf.norm() - 1.0), roundoff)
    check('Dynamics', 'Energy conservation, 3D, 500 steps (order 4)', 'E(t) = E(0)',
          '|ΔE/E|', abs(total_energy_expectation(wf, pot) - E0) / E0, 1e-8)


def heisenberg_3d():
    grid = Grid3D(64, -20.0, 20.0, 64, -20.0, 20.0, 64, -20.0, 20.0)
    wf = WaveFunction3D.gaussian(grid, 1.5, -2.0, 0.5, 1.2, -0.8, 0.4, 2.0, 2.5, 1.8)
    err = max(abs(position_uncertainty(wf) * momentum_uncertainty(wf, 'x') - 0.5),
              abs(y_uncertainty(wf) * momentum_uncertainty(wf, 'y') - 0.5),
              abs(z_uncertainty(wf) * momentum_uncertainty(wf, 'z') - 0.5))
    check('Observables', 'Minimum-uncertainty Gaussian (3D)', 'Δr_j Δp_j = ℏ/2 for each axis',
          'max |Δr Δp − ½|', err, 1e-8)


def vortex_lz():
    grid = Grid2D(128, -8.0, 8.0, 128, -8.0, 8.0)
    wf = WaveFunction2D(grid, (grid.X + 1j * grid.Y) * np.exp(-(grid.X ** 2 + grid.Y ** 2) / 2)).normalize()
    check('Observables', 'Angular momentum of a vortex state', '⟨L_z⟩ = ℏ for ψ ∝ (x+iy)e^{−r²/2}',
          '|⟨L_z⟩ − 1|', abs(angular_momentum_z_expectation(wf) - 1.0), 1e-10)


def harmonic_spectra():
    grid = Grid1D(256, -12.0, 12.0)
    E, _ = eigenstates(grid, HarmonicTrap(1.0), n_states=10, method='lanczos')
    check('Stationary states', 'Harmonic spectrum 1D, 10 levels (Lanczos)', 'E_n = n + ½',
          'max |E_n − exact|', float(np.max(np.abs(E - (np.arange(10) + 0.5)))), 1e-10)
    grid = Grid3D(40, -8.0, 8.0, 40, -8.0, 8.0, 40, -8.0, 8.0)
    w = (1.0, 1.3, 1.7)
    E, _ = eigenstates(grid, HarmonicTrap(w), n_states=6, method='lobpcg')
    levels = sorted(sum(wi * (ni + 0.5) for wi, ni in zip(w, n))
                    for n in np.ndindex(4, 4, 4))[:6]
    check('Stationary states', 'Anisotropic 3D harmonic spectrum, 6 levels (LOBPCG)',
          'E = Σ ω_j (n_j + ½)', 'max |E − exact|', float(np.max(np.abs(E - levels))), 1e-8)
    _, info = ground_state(grid, HarmonicTrap(w), dtau=0.01)
    check('Stationary states', '3D ground state by imaginary time', 'E₀ = ½(ω_x+ω_y+ω_z)',
          '|E₀ − exact|', abs(info.energy - 0.5 * sum(w)), 1e-7)


def feit_fleck():
    grid = Grid1D(256, -12.0, 12.0)
    wf0 = WaveFunction1D.gaussian(grid, 1.5, 0.0, 0.7)
    solver = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.05, order=4)
    n = 4096
    C = np.empty(n, dtype=complex)
    wf = wf0
    for i in range(n):
        C[i] = autocorrelation(wf0.psi, wf.psi, grid)
        wf = solver.step(wf)
    spec = np.abs(np.fft.fft(C * np.hanning(n)))
    E = -2 * np.pi * np.fft.fftfreq(n, 0.05)
    err = 0.0
    for level in (0.5, 1.5, 2.5, 3.5):
        window = np.abs(E - level) < 0.3
        i = np.argmax(np.where(window, spec, 0))
        # parabolic interpolation of the peak
        a, b, c = spec[i - 1], spec[i], spec[i + 1]
        dE = E[1] - E[0]
        err = max(err, abs(E[i] + 0.5 * (a - c) / (a - 2 * b + c) * dE - level))
    resolution = 2 * np.pi / (n * 0.05)
    check('Stationary states', 'Spectrum from autocorrelation (Feit–Fleck)', 'peaks of FFT[C(t)] at n + ½',
          f'max peak offset (resolution 2π/T = {resolution:.3f})', err, 0.1 * resolution)


def bright_soliton():
    grid = Grid1D(2048, -60.0, 60.0)
    psi = 1.0 / np.cosh(grid.x + 20.0) * np.exp(1.5j * grid.x)
    solver = SplitOperatorSolver(grid, None, 0.005, nonlinearity=-1.0, order=4)
    wf = solver.evolve(WaveFunction1D(grid, psi), 4000)
    exact = (1.0 / np.cosh(grid.x + 20.0 - 1.5 * solver.t)) ** 2
    check('Gross–Pitaevskii', 'Moving bright soliton, t = 20', 'exact sech solution of the 1D NLS',
          'max |ρ − ρ_exact|', float(np.max(np.abs(wf.probability_density - exact))), 1e-8)


def gp_virial():
    grid = Grid3D(48, -8.0, 8.0, 48, -8.0, 8.0, 48, -8.0, 8.0)
    trap, g = HarmonicTrap((1.0, 1.2, 1.4)), 60.0
    wf, info = ground_state(grid, trap, dtau=0.005, nonlinearity=g)
    T, V, Ei = kinetic_energy_expectation(wf), potential_energy_expectation(wf, trap), interaction_energy(wf, g)
    check('Gross–Pitaevskii', 'Virial theorem, 3D GP ground state', '2⟨T⟩ − 2⟨V⟩ + 3E_int = 0',
          '|2T − 2V + 3E_int| / V', abs(2 * T - 2 * V + 3 * Ei) / V, 1e-4)


def gp_dtau_order():
    grid = Grid2D(96, -10.0, 10.0, 96, -10.0, 10.0)
    trap = HarmonicTrap((1.0, 1.3))
    _, a = ground_state(grid, trap, dtau=0.02, nonlinearity=50.0, tol=1e-13)
    _, b = ground_state(grid, trap, dtau=0.01, nonlinearity=50.0, tol=1e-13)
    ratio = a.residual / b.residual
    check('Gross–Pitaevskii', 'Imaginary-time GP ground state accuracy', 'residual ∝ dτ² (ratio 4 when halving dτ)',
          f'observed ratio {ratio:.2f}', abs(ratio - 4.0) / 4.0, 0.1)


def rotating_frame():
    grid = Grid2D(128, -12.0, 12.0, 128, -12.0, 12.0)
    x0, Om = 2.0, 0.6
    wf = WaveFunction2D.gaussian(grid, x0, 0.0, 0.0, 0.0, 1 / np.sqrt(2), 1 / np.sqrt(2))
    solver = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.01, order=4, rotation=Om)
    err = 0.0
    for _ in range(10):
        wf = solver.evolve(wf, 100)
        t = solver.t
        err = max(err, abs(position_expectation(wf) - x0 * np.cos(t) * np.cos(Om * t)),
                  abs(y_expectation(wf) + x0 * np.cos(t) * np.sin(Om * t)))
    check('Rotating frame', 'Coherent state in a frame rotating at Ω = 0.6', "⟨r'⟩ = R(−Ωt)·(x₀ cos ωt, 0)",
          "max |⟨r'⟩ − exact|", err, 1e-7)
    wf0 = WaveFunction2D(grid, (grid.X + 1j * grid.Y) * np.exp(-(grid.X ** 2 + grid.Y ** 2) / 2)).normalize()
    solver = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.01, order=4, rotation=Om)
    C = autocorrelation(wf0.psi, solver.evolve(wf0, 500).psi, grid)
    check('Rotating frame', 'Vortex eigenstate in the rotating frame', "C(t) = e^{−i(2ℏω − Ωℏ)t/ℏ}",
          '|C(t) − exact|', abs(C - np.exp(-1j * (2.0 - Om) * solver.t)), 1e-7)


def high_harmonics():
    grid = Grid1D(4096, -200.0, 200.0)
    atom = SoftCoulomb()
    E, states = eigenstates(grid, atom, n_states=1)
    check('Strong-field physics', '1D soft-core hydrogen ground state', 'E₀ = −½ a.u. (I_p = 13.6 eV)',
          '|E₀ + ½|', abs(E[0] + 0.5), 1e-9)
    _, _, ke = sf.classical_returns(4000)
    check('Strong-field physics', 'Three-step model maximum return energy', 'max E_return = 3.17 U_p',
          '|max/U_p − 3.1731|', abs(ke.max() - 3.1731), 2e-4)
    w = sf.wavelength_to_omega(800.0)
    period = 2 * np.pi / w
    laser = DipoleDrive(atom, F0=sf.intensity_to_field(1e14), omega=w,
                        envelope=flat_top_envelope(20 * period, 3 * period))
    solver = SplitOperatorSolver(grid, laser, 0.05, order=4, absorber=AbsorbingBoundaryLayer(grid, 40.0))
    acc = []
    solver.evolve(states[0], int(round(20 * period / 0.05)),
                  callback=lambda n, t, wf: acc.append(sf.dipole_acceleration(wf, atom, laser.force(t))))
    acc = np.array(acc)
    H, S = sf.harmonic_spectrum(acc, 0.05, w)

    def level(q):
        return S[(H > q - 0.3) & (H < q + 0.3)].max()

    ratio = np.median([level(q) for q in range(5, 20, 2)]) / np.median([level(q) for q in range(6, 21, 2)])
    check('Strong-field physics', 'HHG: only odd harmonics (800 nm, 10¹⁴ W/cm²)',
          'inversion symmetry forbids even orders', f'even/odd intensity, H5–H20 (odd/even = {ratio:.1e})',
          1.0 / ratio, 1e-2)
    P = sf.attosecond_pulses(acc, 0.05, w, 11, 21)
    t = np.arange(1, len(acc) + 1) * 0.05
    m = (t > 8 * period) & (t < 12 * period)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(P[m], distance=int(0.3 * period / 0.05), height=0.2 * P[m].max())
    spacing = np.diff(t[m][peaks]) / period
    check('Strong-field physics', 'Attosecond pulse train (H11–H21)', 'one burst per half laser cycle',
          'max |spacing − ½| (cycles)', float(np.max(np.abs(spacing - 0.5))), 0.01)


def write_report(path, elapsed):
    lines = [
        '# QuantumLab validation report',
        '',
        'Generated by `python benchmarks/validation_suite.py`. Every row compares a',
        'simulation with an exact, analytic or independent numerical reference.',
        f'Platform: Python {platform.python_version()}, NumPy {np.__version__}, {platform.machine()}.',
        f'Total runtime: {elapsed:.0f} s.',
        '',
        '| Category | Check | Reference | Metric | Error | Tolerance | Result |',
        '|---|---|---|---|---|---|---|',
    ]
    def cell(text):
        return str(text).replace('|', '\\|')

    for cat, name, ref, metric, err, tol, ok in RESULTS:
        lines.append(f'| {cell(cat)} | {cell(name)} | {cell(ref)} | {cell(metric)} | {err:.2e} | {tol:.0e} | '
                     f'{"pass" if ok else "**FAIL**"} |')
    n_pass = sum(r[-1] for r in RESULTS)
    lines += ['', f'**{n_pass}/{len(RESULTS)} checks passed.**', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()
    for fn in (free_spreading, convergence_orders, driven_oscillator, norm_energy_3d, heisenberg_3d,
               vortex_lz, harmonic_spectra, feit_fleck, bright_soliton, gp_virial, gp_dtau_order,
               rotating_frame, high_harmonics):
        print(f'{fn.__name__}:')
        fn()
    out = Path(__file__).with_name('validation_report.md')
    write_report(out, time.time() - t0)
    n_pass = sum(r[-1] for r in RESULTS)
    print(f'\n{n_pass}/{len(RESULTS)} checks passed — report written to {out}')
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
