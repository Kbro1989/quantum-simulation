"""
Example 12: High-harmonic generation and attosecond pulses from a hydrogen atom.

A single-active-electron hydrogen atom (1D soft-core Coulomb potential,
I_p = 0.5 a.u. = 13.6 eV) is driven by an intense 800 nm laser pulse
(10¹⁴ W/cm², flat-top, 20 cycles). The electron tunnels out, is accelerated
by the field and recollides with its parent ion, emitting high harmonics of
the laser frequency (three-step model: Corkum 1993; Lewenstein et al. 1994).

Checks printed by the script:
  * ground-state energy −0.5 a.u. (hydrogen);
  * only odd harmonics are emitted (inversion symmetry of atom + field);
  * emission times of each harmonic (Gabor analysis) against the classical
    return times of short and long trajectories;
  * the attosecond pulse train has two bursts per laser cycle;
  * the cutoff grows linearly with U_p (intensity scan). The extracted cutoff
    of a single 1D atom depends on how it is defined, because the plateau
    has deep interference minima; the fit is printed together with the
    classical (3.17 U_p + I_p) and quantum-corrected (3.17 U_p + 1.32 I_p) laws.

The 1D model over-estimates ionisation compared with a real 3D atom; it
reproduces the plateau, cutoff and attosecond timing physics.

Run:  python examples/12_high_harmonic_generation.py [--no-scan]

Outputs (in output/):
    12_hhg_spectrum.png          harmonic spectrum with cutoff laws
    12_hhg_time_frequency.png    Gabor map with classical return energies
    12_attosecond_pulses.png     attosecond pulse train from H11–H21
    12_electron_spacetime.png    log electron density in space and time
    12_electron_density.mp4      animated electron density during two cycles
    12_hhg_cutoff_scan.png       cutoff vs ponderomotive energy (unless --no-scan)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.signal import find_peaks, peak_widths
from quantumlab.core.grid import Grid1D
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.potentials import SoftCoulomb, DipoleDrive, flat_top_envelope
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.stationary import eigenstates
from quantumlab.strongfield import (
    AU_TIME_FS, HARTREE_EV, wavelength_to_omega, intensity_to_field, ponderomotive_energy,
    classical_returns, harmonic_spectrum, gabor_transform, attosecond_pulses, dipole_acceleration,
)
from quantumlab.visualization.style import set_style

GRID = Grid1D(4096, -200.0, 200.0)
DT = 0.05
WAVELENGTH_NM = 800.0
N_CYCLES, N_RAMP = 20, 3


def run(intensity, wf0, atom, record_density=False):
    """Propagate through the pulse; return times, dipole acceleration and optional densities."""
    omega = wavelength_to_omega(WAVELENGTH_NM)
    period = 2 * np.pi / omega
    field = intensity_to_field(intensity)
    duration = N_CYCLES * period
    laser = DipoleDrive(atom, F0=field, omega=omega, envelope=flat_top_envelope(duration, N_RAMP * period))
    solver = SplitOperatorSolver(GRID, laser, DT, order=4, absorber=AbsorbingBoundaryLayer(GRID, 40.0))
    acc, dens = [], []

    def record(n, t, wf):
        acc.append(dipole_acceleration(wf, atom, laser.force(t)))
        if record_density and n % 10 == 0:
            dens.append(wf.probability_density[::2].astype(np.float32))

    wf = solver.evolve(wf0, int(round(duration / DT)), callback=record)
    return np.arange(1, len(acc) + 1) * DT, np.array(acc), np.array(dens), laser, wf


def odd_harmonic_levels(acc, omega, orders):
    H, S = harmonic_spectrum(acc, DT, omega)
    return np.array([S[(H > o - 0.3) & (H < o + 0.3)].max() for o in orders])


def main(scan=True):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs('output', exist_ok=True)
    set_style('dark')
    atom = SoftCoulomb(Z=1.0, softening=np.sqrt(2.0))
    omega = wavelength_to_omega(WAVELENGTH_NM)
    period = 2 * np.pi / omega
    intensity = 1e14
    field = intensity_to_field(intensity)
    Up = ponderomotive_energy(field, omega)

    print('--- Example 12: high-harmonic generation in hydrogen ---')
    energies, states = eigenstates(GRID, atom, n_states=1)
    Ip = -energies[0]
    print(f'Ground state: E0 = {energies[0]:.10f} a.u. = {energies[0] * HARTREE_EV:.3f} eV (hydrogen: −13.606 eV)')
    print(f'Laser: λ = {WAVELENGTH_NM:g} nm (ω = {omega:.5f} a.u., period {period * AU_TIME_FS:.2f} fs), '
          f'I = {intensity:.1e} W/cm², E0 = {field:.5f} a.u., U_p = {Up:.4f} a.u. = {Up * HARTREE_EV:.2f} eV')
    print(f'Keldysh parameter γ = {np.sqrt(Ip / (2 * Up)):.2f}; quiver amplitude {field / omega ** 2:.1f} a.u.')

    t, acc, dens, laser, wf_end = run(intensity, states[0], atom, record_density=True)
    print(f'Propagated {len(t)} steps; probability left in the box: {wf_end.norm():.3f} '
          '(the rest was ionised and absorbed at the edges)')

    # ------------------------------------------------------------- spectrum
    H, S = harmonic_spectrum(acc, DT, omega)
    odd = np.arange(1, 60, 2)
    lvl_odd = odd_harmonic_levels(acc, omega, odd)
    lvl_even = odd_harmonic_levels(acc, omega, odd + 1)
    plateau = (odd >= 5) & (odd <= 19)
    ratio = np.median(lvl_odd[plateau]) / np.median(lvl_even[plateau])
    cut_classical = (Ip + 3.1731 * Up) / omega
    cut_quantum = (1.32 * Ip + 3.17 * Up) / omega
    print(f'\nOdd/even harmonic intensity ratio (H5–H20, median): {ratio:.1e}  (symmetry forbids even harmonics)')
    print(f'Cutoff laws: classical I_p + 3.17 U_p → H{cut_classical:.1f};  quantum-corrected 1.32 I_p + 3.17 U_p '
          f'→ H{cut_quantum:.1f}')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(H, S, color='#ffb000', lw=0.8)
    ax.axvline(cut_classical, color='#7fd4ff', ls='--', lw=1.2, label=f'classical cutoff I$_p$+3.17U$_p$ (H{cut_classical:.1f})')
    ax.axvline(cut_quantum, color='#ff5fa2', ls=':', lw=1.5, label=f'quantum-corrected 1.32I$_p$+3.17U$_p$ (H{cut_quantum:.1f})')
    ax.set_xlim(0, 40)
    ax.set_ylim(S[(H > 0.5) & (H < 40)].max() * 1e-12, S[(H > 0.5) & (H < 40)].max() * 10)
    ax.set_xticks(range(1, 41, 4))
    ax.set_xlabel('harmonic order ω/ω$_0$')
    ax.set_ylabel('|a(ω)|² (arb. units)')
    ax.set_title(f'High-harmonic spectrum of hydrogen, 800 nm, {intensity:.0e} W/cm²', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig('output/12_hhg_spectrum.png', dpi=150)
    plt.close(fig)

    # ------------------------------------------------- time–frequency analysis
    times = np.arange(8 * period, 12 * period, period / 120)
    sigma = 1.0 / (3.0 * omega)
    tt, oo, G = gabor_transform(acc, DT, omega, np.linspace(1, 35, 241), width=sigma, times=times)
    births, returns, ke = classical_returns(4000)
    peak = np.argmax(ke)
    branches = {'long': slice(0, peak + 1), 'short': slice(peak, None)}   # early births return late

    def return_times(branch, n_range=range(10, 30)):
        out = []
        for n in n_range:
            tr = (returns[branch] + np.pi / 2 + n * np.pi) / omega        # F(t) = F0 sin ωt ⇒ φ = ωt − π/2
            out.append(tr)
        return np.concatenate(out), np.tile(ke[branch], len(n_range))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.pcolormesh(tt / period, oo, np.log10(G.T + 1e-300), vmin=np.log10(G.max()) - 5, cmap='magma',
                  shading='auto')
    for name, color in (('short', '#7fd4ff'), ('long', '#7dff9a')):
        tr, k = return_times(branches[name])
        m = (tr > times[0]) & (tr < times[-1])
        ax.plot(tr[m] / period, (Ip + k[m] * Up) / omega, '.', ms=1.2, color=color,
                label=f'classical {name} trajectories')
    ax.set_xlabel('time (laser cycles)')
    ax.set_ylabel('harmonic order')
    ax.set_ylim(1, 35)
    ax.set_title('When each harmonic is emitted: quantum (colour) vs classical returns (dots)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, markerscale=8)
    fig.text(0.01, 0.01, f'Gabor window σ = {sigma / period:.3f} cycle (frequency blur ±{1 / (sigma * omega):.0f} orders)',
             fontsize=8, alpha=0.8)
    fig.tight_layout()
    fig.savefig('output/12_hhg_time_frequency.png', dpi=150)
    plt.close(fig)

    print('\nEmission time vs classical return time (Gabor, flat top):')
    fine = np.arange(7 * period, 13 * period, period / 400)
    for q in (11, 13, 15, 17, 19):
        _, _, g = gabor_transform(acc, DT, omega, [q], width=sigma, times=fine)
        sig = g[:, 0]
        pk, _ = find_peaks(sig, distance=int(0.15 * 400))
        pk = pk[sig[pk] > 0.1 * sig.max()]
        target = (q * omega - Ip) / Up
        classical_t = []
        for name in ('short', 'long'):
            b = branches[name]
            order = np.argsort(ke[b])
            phase = np.interp(target, ke[b][order], returns[b][order])
            classical_t += [(phase + np.pi / 2 + n * np.pi) / omega for n in range(10, 30)]
        classical_t = np.array(classical_t)
        dev = [np.min(np.abs(classical_t - fine[p])) / period for p in pk]
        print(f'  H{q}: {len(pk)} bursts, |Δt| median {np.median(dev):.3f}, max {np.max(dev):.3f} cycle')
    print(f'  (time resolution of the analysis: σ = {sigma / period:.3f} cycle '
          f'= {sigma * AU_TIME_FS * 1000:.0f} as)')

    # ---------------------------------------------------- attosecond pulses
    P = attosecond_pulses(acc, DT, omega, 11, 21)
    m = (t > 8 * period) & (t < 12 * period)
    pk, _ = find_peaks(P[m], distance=int(0.3 * period / DT), height=0.2 * P[m].max())
    fwhm = peak_widths(P[m], pk, 0.5)[0] * DT * AU_TIME_FS * 1000
    spacing = np.diff(t[m][pk]) / period
    print(f'\nAttosecond pulse train (H11–H21): {len(pk)} bursts in 4 cycles, spacing {spacing.min():.3f}–'
          f'{spacing.max():.3f} cycle, FWHM {fwhm.min():.0f}–{fwhm.max():.0f} as')

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(t[m] / period, P[m] / P[m].max(), color='#ffb000', lw=1.4, label='attosecond bursts (H11–H21)')
    F = np.array([laser.force(x) for x in t[m]])
    ax.plot(t[m] / period, 0.5 * F / np.abs(F).max() - 0.6, color='#7fd4ff', lw=1.0, label='laser field (scaled)')
    ax.set_xlabel('time (laser cycles)')
    ax.set_yticks([])
    ax.set_title(f'Attosecond pulse train: two bursts per cycle, FWHM ≈ {np.median(fwhm):.0f} as', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig('output/12_attosecond_pulses.png', dpi=150)
    plt.close(fig)

    # -------------------------------------------------- electron in space-time
    x = GRID.x[::2]
    td = t[9::10][:len(dens)]
    sel = (x > -80) & (x < 80)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax1.pcolormesh(td / period, x[sel], np.log10(dens[:, sel].T + 1e-12), vmin=-8, vmax=0, cmap='inferno',
                   shading='auto')
    ax1.set_ylabel('x (a.u.)')
    ax1.set_title('Electron density log$_{10}$|ψ(x, t)|²: bursts leave and return every half cycle',
                  fontweight='bold')
    ax2.plot(t / period, [laser.force(x_) for x_ in t], color='#7fd4ff', lw=0.8)
    ax2.set_xlabel('time (laser cycles)')
    ax2.set_ylabel('field (a.u.)')
    fig.tight_layout()
    fig.savefig('output/12_electron_spacetime.png', dpi=150)
    plt.close(fig)

    fig, (ax, axf) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[3, 1]))
    line, = ax.plot(x[sel], np.log10(dens[0, sel] + 1e-12), color='#ffb000', lw=1.2)
    ax.set_ylim(-9, 0.5)
    ax.set_xlim(-80, 80)
    ax.set_xlabel('x (a.u.)')
    ax.set_ylabel('log$_{10}$|ψ|²')
    title = ax.set_title('', fontweight='bold')
    tf = np.linspace(8 * period, 10 * period, 400)
    axf.plot(tf / period, [laser.force(x_) for x_ in tf], color='#7fd4ff', lw=1)
    dot, = axf.plot([], [], 'o', color='#ff5fa2')
    axf.set_xlabel('cycles')
    axf.set_yticks([])
    axf.set_title('laser field')
    fig.tight_layout()
    frames = np.nonzero((td >= 8 * period) & (td <= 10 * period))[0][::3]
    writer = FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '20'])
    with writer.saving(fig, 'output/12_electron_density.mp4', dpi=100):
        for k in frames:
            line.set_ydata(np.log10(dens[k, sel] + 1e-12))
            title.set_text(f'Hydrogen electron in an 800 nm field   t = {td[k] / period:5.2f} cycles')
            dot.set_data([td[k] / period], [laser.force(td[k])])
            writer.grab_frame()
    plt.close(fig)

    # --------------------------------------------------------- intensity scan
    if scan:
        print('\nCutoff scan (8 intensities; edge = first odd harmonic 2 decades below the H5–H15 plateau):')
        intensities = np.arange(0.5, 1.21, 0.1) * 1e14
        Ups, cutoffs = [], []
        for I in intensities:
            _, a, _, _, _ = run(I, states[0], atom)
            levels = np.log10(odd_harmonic_levels(a, omega, odd))
            plateau_level = np.median(levels[(odd >= 5) & (odd <= 15)])
            k = np.nonzero((odd > 15) & (levels < plateau_level - 2.0))[0][0]
            q = odd[k - 1] + (plateau_level - 2.0 - levels[k - 1]) / (levels[k] - levels[k - 1]) * 2
            up = ponderomotive_energy(intensity_to_field(I), omega)
            Ups.append(up)
            cutoffs.append(q * omega)
            print(f'  I = {I:.1e}: U_p = {up:.4f}, spectral edge at H{q:.1f}')
        Ups, cutoffs = np.array(Ups), np.array(cutoffs)
        (slope, icpt), cov = np.polyfit(Ups, cutoffs, 1, cov=True)
        print(f'  linear fit: E_edge = ({slope:.2f} ± {np.sqrt(cov[0, 0]):.2f}) U_p + '
              f'({icpt / Ip:.2f} ± {np.sqrt(cov[1, 1]) / Ip:.2f}) I_p')
        print('  The edge is the end of the roll-off, so it lies above the cutoff and its slope depends on the\n'
              '  chosen depth; the linear growth with U_p is the robust result.')
        fig, ax = plt.subplots(figsize=(7.5, 5))
        u = np.linspace(0, Ups.max() * 1.05, 50)
        ax.plot(Ups / Ip, cutoffs / Ip, 'o', color='#ffb000', label='simulation (spectral edge)')
        ax.plot(u / Ip, (slope * u + icpt) / Ip, '-', color='#ffb000', lw=1, alpha=0.7,
                label=f'fit {slope:.2f} U$_p$ + {icpt / Ip:.2f} I$_p$')
        ax.plot(u / Ip, (3.17 * u + Ip) / Ip, '--', color='#7fd4ff', label='classical 3.17 U$_p$ + I$_p$')
        ax.plot(u / Ip, (3.17 * u + 1.32 * Ip) / Ip, ':', color='#ff5fa2', lw=1.6, label='quantum 3.17 U$_p$ + 1.32 I$_p$')
        ax.set_xlabel('U$_p$ / I$_p$')
        ax.set_ylabel('cutoff energy / I$_p$')
        ax.set_title('HHG cutoff grows linearly with the ponderomotive energy', fontweight='bold')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig('output/12_hhg_cutoff_scan.png', dpi=150)
        plt.close(fig)

    print('\nSaved: output/12_hhg_spectrum.png, 12_hhg_time_frequency.png, 12_attosecond_pulses.png, '
          '12_electron_spacetime.png, 12_electron_density.mp4' + (', 12_hhg_cutoff_scan.png' if scan else ''))


if __name__ == '__main__':
    main(scan='--no-scan' not in sys.argv)
