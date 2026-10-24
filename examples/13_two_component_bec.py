"""
Example 13: Two-component Bose–Einstein condensates: phase separation and
interlaced ("two-colour") vortex lattices.

Two condensates of equal mass with intra-species couplings g11 = g22 = g and
inter-species coupling g12 obey coupled Gross–Pitaevskii equations. A uniform
mixture is stable (miscible) when g12² < g11 g22 and separates into domains
(immiscible) when g12² > g11 g22.

Part A: miscibility in a trap. The ground-state overlap of the two components
        is scanned across g12/g = 1.
Part B: spinodal decomposition. A uniform mixture with g12 > g, seeded with
        weak noise, separates into domains that then coarsen. The early growth
        is compared with Bogoliubov theory: the spin mode with
        ε_k = ℏ²k²/2m grows as exp(γt) with γ(k) = √(ε_k (2n(g12 − g) − ε_k)),
        fastest at k* = √(2mn(g12 − g))/ℏ with γ* = n(g12 − g). The noise
        seeds every wavevector, so the growth is measured on the shell
        |k| ≈ k* of the spin structure factor, not on the total rms.
Part C: rotation. Both components in a rotating trap (damped dynamics, as in
        example 11) nucleate vortices. With g12 > 0 the two lattices interlace:
        the vortex cores of one component sit where the other is densest.

Run all parts, or a subset:  python examples/13_two_component_bec.py [A] [B] [C]

Outputs (in output/):
    13_miscibility.png             overlap vs g12/g and ground-state densities
    13_phase_separation.mp4        domain formation and coarsening (two colours)
    13_phase_separation_growth.png spin-density growth vs Bogoliubov
    13_vortex_lattices.mp4         formation of the interlaced lattices
    13_vortex_lattices.png         final interlaced lattices with vortex positions
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from quantumlab.core.grid import Grid2D
from quantumlab.core.wavefunction import WaveFunction2D
from quantumlab.potentials import HarmonicTrap
from quantumlab.solvers.multicomponent import CoupledGPESolver
from quantumlab.stationary import coupled_ground_state
from quantumlab.observables import miscibility_overlap, vortex_positions, angular_momentum_z_expectation
from quantumlab.visualization.style import set_style

COLOUR_A = np.array([1.00, 0.35, 0.55])     # component 1
COLOUR_B = np.array([0.25, 0.85, 1.00])     # component 2


def two_colour(n1, n2, vmax=None):
    """RGB image: component 1 in pink, component 2 in cyan, overlap tends to white."""
    vmax = vmax or max(n1.max(), n2.max())
    a = np.clip(n1 / vmax, 0, 1)[..., None]
    b = np.clip(n2 / vmax, 0, 1)[..., None]
    return np.clip(a * COLOUR_A + b * COLOUR_B, 0, 1).transpose(1, 0, 2)


def displaced(grid, dx, norm):
    wf = WaveFunction2D.gaussian(grid, dx, 0.0, 0.0, 0.0, 1.8, 1.8)
    wf.psi *= np.sqrt(norm)
    return wf


def part_a():
    print('\n[A] Miscibility of the ground state in a harmonic trap (g11 = g22 = g = 100, N1 = N2 = 0.5)')
    grid = Grid2D(96, -9.0, 9.0, 96, -9.0, 9.0)
    trap = HarmonicTrap(1.0)
    g = 100.0
    ratios = np.array([0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4])
    overlaps, examples = [], {}
    for r in ratios:
        states, info = coupled_ground_state(grid, (trap, trap), [[g, r * g], [r * g, g]],
                                            (displaced(grid, 1.5, 0.5), displaced(grid, -1.5, 0.5)),
                                            dtau=0.005, tol=1e-10, max_steps=30000)
        overlaps.append(miscibility_overlap(*states))
        print(f'    g12/g = {r:4.2f}: overlap Λ = {overlaps[-1]:.4f}   (residuals {max(info.residuals):.1e}, '
              f'{info.steps} steps)')
        if r in (0.8, 1.2):
            examples[r] = states
    fig = plt.figure(figsize=(12, 4.2))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(ratios, overlaps, 'o-', color='#ffb000')
    ax.axvline(1.0, color='#7fd4ff', ls='--', lw=1, label='g$_{12}^2$ = g$_{11}$g$_{22}$')
    ax.set_xlabel('g$_{12}$ / g')
    ax.set_ylabel('overlap Λ')
    ax.set_title('Miscible → immiscible', fontweight='bold')
    ax.legend(fontsize=8)
    extent = [grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]]
    for k, r in enumerate((0.8, 1.2)):
        s1, s2 = examples[r]
        ax = fig.add_subplot(1, 3, 2 + k)
        ax.imshow(two_colour(s1.probability_density, s2.probability_density), origin='lower', extent=extent)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_title(f'g$_{{12}}$/g = {r} ({"miscible" if r < 1 else "phase separated"})', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig('output/13_miscibility.png', dpi=150)
    plt.close(fig)
    return ratios, np.array(overlaps)


def part_b():
    print('\n[B] Spinodal decomposition of a uniform mixture (n = 1 per component, g = 1, g12 = 1.3)')
    L, N = 128.0, 256
    grid = Grid2D(N, -L / 2, L / 2, N, -L / 2, L / 2)
    n, g, g12 = 1.0, 1.0, 1.3
    dt = 0.05
    k_star = np.sqrt(2 * n * (g12 - g))
    gamma_star = n * (g12 - g)
    print(f'    Bogoliubov: fastest-growing wavelength 2π/k* = {2 * np.pi / k_star:.2f}, growth rate γ* = {gamma_star:.3f}')
    print(f'    split-step stability: kinetic phase per step {(np.pi / grid.dx) ** 2 / 2 * dt:.2f} rad (< π)')
    rng = np.random.default_rng(1)
    noise = 1e-4 * (rng.standard_normal(grid.shape) + 1j * rng.standard_normal(grid.shape))
    wf1 = WaveFunction2D(grid, np.sqrt(n) * (1 + noise))
    wf2 = WaveFunction2D(grid, np.sqrt(n) * (1 - noise))
    norms0 = (wf1.norm(), wf2.norm())
    solver = CoupledGPESolver(grid, (None, None), [[g, g12], [g12, g]], dt, order=2)

    times, spin_rms, shell_amp, k_peak = [], [], [], []
    K = np.sqrt(grid.K_x ** 2 + grid.K_y ** 2)
    shell = np.abs(K - k_star) < grid.dk_x
    k_bins = np.arange(0.0, 2.0, grid.dk_x)
    k_index = np.digitize(K, k_bins)
    bin_count = np.maximum(np.bincount(k_index.ravel(), minlength=len(k_bins) + 1), 1)

    fig, ax = plt.subplots(figsize=(6, 6.3))
    fig.patch.set_facecolor('#0b0b10')
    im = ax.imshow(two_colour(wf1.probability_density, wf2.probability_density, vmax=2 * n), origin='lower',
                   extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title('', fontsize=12, fontweight='bold')
    writer = FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '20'])
    n_steps = int(round(150.0 / dt))
    with writer.saving(fig, 'output/13_phase_separation.mp4', dpi=100):
        state = (wf1, wf2)
        for step in range(0, n_steps + 1, 20):
            if step:
                state = solver.evolve(state, 20)
            s = state[0].probability_density - state[1].probability_density
            times.append(solver.t)
            spin_rms.append(float(np.sqrt(np.mean(s ** 2))))
            S = np.abs(np.fft.fft2(s)) ** 2
            S[0, 0] = 0.0
            shell_amp.append(float(np.sqrt(S[shell].mean())))
            radial = np.bincount(k_index.ravel(), S.ravel(), minlength=len(k_bins) + 1) / bin_count
            k_peak.append(float(k_bins[np.argmax(radial[1:len(k_bins)])]))
            if step % 60 == 0:
                im.set_data(two_colour(state[0].probability_density, state[1].probability_density, vmax=2 * n))
                title.set_text(f'Phase separation of a binary BEC   t = {solver.t:5.1f}')
                writer.grab_frame(facecolor='#0b0b10')
    plt.close(fig)
    times, spin_rms, shell_amp, k_peak = map(np.array, (times, spin_rms, shell_amp, k_peak))
    lin = (shell_amp > 10 * shell_amp[0]) & (spin_rms < 1e-2)
    gamma_fit = np.polyfit(times[lin], np.log(shell_amp[lin]), 1)[0]
    k_lin = np.median(k_peak[lin])
    print(f'    linear stage (t = {times[lin][0]:.0f}–{times[lin][-1]:.0f}): growth on the |k| ≈ k* shell {gamma_fit:.4f} '
          f'vs γ* = {gamma_star:.4f} ({100 * (gamma_fit / gamma_star - 1):+.2f}%); peak of the spin structure '
          f'factor at k = {k_lin:.3f} vs k* = {k_star:.3f} (bin width {grid.dk_x:.3f})')
    print(f'    late stage: domains coarsen, the structure-factor peak moves from {k_lin:.2f} to {k_peak[-1]:.2f} '
          f'by t = {times[-1]:.0f}')
    print(f'    populations conserved: relative drift {abs(state[0].norm() / norms0[0] - 1):.1e}, '
          f'{abs(state[1].norm() / norms0[1] - 1):.1e}')

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.semilogy(times, shell_amp, color='#ffb000', lw=1.8, label='simulation: spin mode on |k| ≈ k*')
    # theory line anchored at the start of the fitted window and drawn beyond it
    t_ref = times[times <= times[lin][-1] + 12]
    a1.semilogy(t_ref, shell_amp[lin][0] * np.exp(gamma_star * (t_ref - times[lin][0])), '--', color='#7fd4ff',
                lw=1.4, label=f'Bogoliubov e$^{{γ^* t}}$, γ* = {gamma_star:.2f}')
    a1.axvspan(times[lin][0], times[lin][-1], color='#7fd4ff', alpha=0.08, label='fit window (linear stage)')
    a1.set_xlabel('t')
    a1.set_ylabel('amplitude (arb. units)')
    a1.set_title('Exponential growth of the instability', fontweight='bold')
    a1.legend(fontsize=8)
    a2.plot(times, k_peak, color='#ffb000', lw=1.8, label='simulation: peak of the spin structure factor')
    a2.axhline(k_star, color='#7fd4ff', ls='--', lw=1, label=f'fastest-growing k* = {k_star:.2f}')
    a2.set_xlabel('t')
    a2.set_ylabel('characteristic wavenumber')
    a2.set_title('Domain size: selection, then coarsening', fontweight='bold')
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('output/13_phase_separation_growth.png', dpi=150)
    plt.close(fig)
    return gamma_fit, gamma_star, k_lin, k_star


def part_c():
    print('\n[C] Rotating two-component condensate (g = 500, g12 = 0.5 g, Ω = 0.8, damping γ = 0.05)')
    grid = Grid2D(128, -10.0, 10.0, 128, -10.0, 10.0)
    g, delta, Om, gamma = 500.0, 0.5, 0.8, 0.05
    G = [[g, delta * g], [delta * g, g]]
    eps_x, eps_y = 0.03, 0.09
    trap = HarmonicTrap((np.sqrt(1.0 + eps_x), np.sqrt(1.0 + eps_y)))
    (s1, s2), info = coupled_ground_state(grid, (HarmonicTrap(1.0), HarmonicTrap(1.0)), G,
                                          (displaced(grid, 0.2, 0.5), displaced(grid, -0.2, 0.5)), dtau=0.005)
    print(f'    non-rotating ground state: overlap Λ = {miscibility_overlap(s1, s2):.4f} (miscible, g12 < g)')
    solver = CoupledGPESolver(grid, (trap, trap), G, 0.005, rotation=Om, damping=gamma)
    extent = [grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]]
    fig, ax = plt.subplots(figsize=(6, 6.3))
    fig.patch.set_facecolor('#0b0b10')
    vmax = max(s1.probability_density.max(), s2.probability_density.max())
    im = ax.imshow(two_colour(s1.probability_density, s2.probability_density, vmax), origin='lower', extent=extent)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title('', fontsize=12, fontweight='bold')
    writer = FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '20'])
    state = (s1, s2)
    with writer.saving(fig, 'output/13_vortex_lattices.mp4', dpi=100):
        for k in range(161):
            if k:
                state = solver.evolve(state, 250)
            im.set_data(two_colour(state[0].probability_density, state[1].probability_density, vmax))
            title.set_text(f'Two-component vortex lattices   t = {solver.t:5.1f}')
            writer.grab_frame(facecolor='#0b0b10')
            if k % 40 == 0 and k:
                n1v = len(vortex_positions(state[0])[0])
                n2v = len(vortex_positions(state[1])[0])
                print(f'    t = {solver.t:5.1f}: vortices {n1v} + {n2v}, '
                      f'⟨L_z⟩ = {angular_momentum_z_expectation(state[0]):.2f} + {angular_momentum_z_expectation(state[1]):.2f}')
    plt.close(fig)

    s1, s2 = state
    p1, q1 = vortex_positions(s1)
    p2, q2 = vortex_positions(s2)
    n1, n2 = s1.probability_density, s2.probability_density

    def relative_density_at(points, n):
        """Density at the given points relative to its local average (smoothing 1.5)."""
        smooth = gaussian_filter(n, 1.5 / grid.dx)
        i = np.clip(np.searchsorted(grid.x, points[:, 0]), 0, grid.N_x - 1)
        j = np.clip(np.searchsorted(grid.y, points[:, 1]), 0, grid.N_y - 1)
        return n[i, j] / smooth[i, j]

    r12 = relative_density_at(p1, n2)        # density of 2 at the cores of 1
    r21 = relative_density_at(p2, n1)
    print(f'    final: {len(p1)} + {len(p2)} vortices (charges {sorted(set(q1.tolist()) | set(q2.tolist()))})')
    print(f'    interlacing: at the vortex cores of one component the other component\'s density is '
          f'{np.median(r12):.2f}× / {np.median(r21):.2f}× its local average (median); '
          f'{100 * np.mean(r12 > 1):.0f}% / {100 * np.mean(r21 > 1):.0f}% of cores sit on a density maximum (> 1×)')
    d_same, _ = cKDTree(p1).query(p1, k=2)
    d_cross, _ = cKDTree(p2).query(p1, k=1)
    print(f'    nearest same-component vortex {np.median(d_same[:, 1]):.2f}, nearest other-component vortex '
          f'{np.median(d_cross):.2f} (interlaced: ≈ half the same-component spacing or more)')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, img, ttl in ((axes[0], two_colour(n1, 0 * n2, vmax), 'component 1'),
                         (axes[1], two_colour(0 * n1, n2, vmax), 'component 2'),
                         (axes[2], two_colour(n1, n2, vmax), 'both: interlaced lattices')):
        ax.imshow(img, origin='lower', extent=extent)
        ax.set_xlim(-7.5, 7.5)
        ax.set_ylim(-7.5, 7.5)
        ax.set_title(ttl, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    axes[2].scatter(p1[:, 0], p1[:, 1], s=14, facecolors='none', edgecolors=COLOUR_A, linewidths=1)
    axes[2].scatter(p2[:, 0], p2[:, 1], s=14, facecolors='none', edgecolors=COLOUR_B, linewidths=1)
    fig.suptitle('Rotating two-component BEC: the vortices of each component fill the other\'s density peaks',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig('output/13_vortex_lattices.png', dpi=150)
    plt.close(fig)
    return r12, r21


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs('output', exist_ok=True)
    set_style('dark')
    print('--- Example 13: two-component Bose–Einstein condensates ---')
    parts = {a.upper() for a in sys.argv[1:]} or {'A', 'B', 'C'}
    if 'A' in parts:
        part_a()
    if 'B' in parts:
        part_b()
    if 'C' in parts:
        part_c()
    print('\nSaved: output/13_miscibility.png, 13_phase_separation.mp4, 13_phase_separation_growth.png, '
          '13_vortex_lattices.mp4, 13_vortex_lattices.png')


if __name__ == '__main__':
    main()
