"""
Example 11: Vortex lattice in a rotating Bose–Einstein condensate.

Part 1 — formation (2D, real-time dynamics). A condensate at rest is put
in a slightly anisotropic trap rotating at Ω = 0.8 ω. In the rotating frame
the trap is static and the dissipative Gross–Pitaevskii equation
(i − γ)ℏ ∂ψ/∂t = (H − ΩL_z)ψ is integrated (Tsubota, Kasamatsu & Ueda,
PRA 65, 023603, 2002). The cloud elongates, its surface becomes unstable,
vortices enter from the edge and crystallise into a triangular lattice.

Part 2 — vortex lines (3D). The 2D lattice seeds a 3D condensate in a
cigar-shaped rotating trap with the same radial anisotropy as in part 1.
Vortex lattices have very soft modes (relaxation rates ~0.01 ω): imaginary
time stalls on them and its fixed point also carries an O(dτ²) error. So a
short imaginary-time run only smooths the seed, and the stationary state is
then found by preconditioned conjugate-gradient energy minimisation
(``ground_state_pcg``), whose fixed point is exact. The residual
‖(H − ΩL_z)ψ − μψ‖ is printed; the vortex lines run along z and are drawn as
tubes inside the translucent cloud. This part takes ~30–40 minutes on a CPU.

Check: a rigidly rotating superfluid has the areal vortex density
n_v = mΩ/(πℏ) (Feynman's rule), i.e. a triangular lattice with spacing
a = (2/(√3 n_v))^½. The rule is exact for an infinite uniform lattice; this
lattice has fewer than 20 vortices in an inhomogeneous cloud, so deviations
of a few percent are expected. The measured spacing is printed, and each
line is followed slice by slice along z to measure how straight it is.

Outputs (in output/):
    11_vortex_lattice_formation.mp4   density and phase during formation (2D)
    11_vortex_lattice_2d.png          final lattice with detected vortices
    11_vortex_count.png               number of vortices and ⟨L_z⟩ vs time
    11_vortex_lines_3d.mp4            orbiting view of the 3D vortex lines
    11_vortex_lines_3d.png            3D still
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from quantumlab.core.grid import Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction2D, WaveFunction3D
from quantumlab.potentials import HarmonicTrap
from quantumlab.stationary import ground_state, ground_state_pcg
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.observables import angular_momentum_z_expectation, vortex_positions
from quantumlab.visualization.volume_3d import IsosurfaceAnimation, IsosurfaceRenderer
from quantumlab.visualization.style import set_style


def lattice_spacing(positions):
    """Mean and standard deviation of the nearest-neighbour distance over all vortices."""
    d, _ = cKDTree(positions).query(positions, k=2)
    return float(np.mean(d[:, 1])), float(np.std(d[:, 1]))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs('output', exist_ok=True)
    print('--- Example 11: vortex lattice in a rotating BEC ---')

    # ------------------------------------------------------------ Part 1: 2D
    grid = Grid2D(128, -10.0, 10.0, 128, -10.0, 10.0)
    g2, Om, gamma = 500.0, 0.8, 0.05
    eps_x, eps_y = 0.03, 0.09                        # trap anisotropy that stirs the cloud
    trap = HarmonicTrap((np.sqrt(1.0 + eps_x), np.sqrt(1.0 + eps_y)))
    dt, t_end, frame_every = 0.005, 200.0, 250

    wf, info = ground_state(grid, HarmonicTrap(1.0), dtau=0.005, nonlinearity=g2)
    print(f'\n[1] Non-rotating ground state: μ = {info.chemical_potential:.3f}, '
          f'Thomas–Fermi radius ≈ {np.sqrt(2 * info.chemical_potential):.2f}')
    print(f'    Rotating the trap at Ω = {Om}, damping γ = {gamma}, until t = {t_end:g}')

    solver = SplitOperatorSolver(grid, trap, dt, nonlinearity=g2, rotation=Om, damping=gamma)
    extent = [grid.x[0], grid.x[-1] + grid.dx, grid.y[0], grid.y[-1] + grid.dy]
    set_style('dark')
    fig, (ax_d, ax_p) = plt.subplots(1, 2, figsize=(11, 5.4))
    fig.patch.set_facecolor('#0b0b10')
    vmax = float(wf.probability_density.max())
    im_d = ax_d.imshow(wf.probability_density.T, origin='lower', extent=extent, cmap='inferno', vmin=0, vmax=vmax)
    im_p = ax_p.imshow(np.angle(wf.psi).T, origin='lower', extent=extent, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    for ax, lbl in ((ax_d, 'density $|\\psi|^2$'), (ax_p, 'phase arg $\\psi$')):
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_title(lbl)
        ax.set_xlabel('x')
    ax_d.set_ylabel('y')
    title = fig.suptitle('', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.01, f'rotating frame, Ω = {Om} ω, damping γ = {gamma}', ha='center', fontsize=9, alpha=0.8)

    times, counts, Lz = [], [], []
    writer = FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '20'])
    n_steps = int(round(t_end / dt))
    with writer.saving(fig, 'output/11_vortex_lattice_formation.mp4', dpi=100):
        for step in range(0, n_steps + 1):
            if step > 0:
                wf = solver.step(wf)
            if step % frame_every == 0:
                pos, q = vortex_positions(wf)
                times.append(solver.t)
                counts.append(int(np.sum(q)))
                Lz.append(angular_momentum_z_expectation(wf))
                im_d.set_data(wf.probability_density.T)
                im_p.set_data(np.angle(wf.psi).T)
                title.set_text(f'Vortex-lattice formation in a rotating BEC    t = {solver.t:6.1f}')
                writer.grab_frame(facecolor='#0b0b10')
            if step % 4000 == 0 and step > 0:
                print(f'    t = {solver.t:6.1f}   vortices = {counts[-1]:3d}   ⟨L_z⟩/N = {Lz[-1]:6.2f}')
    plt.close(fig)
    print('    Animation saved: output/11_vortex_lattice_formation.mp4')

    pos, q = vortex_positions(wf)
    a, a_std = lattice_spacing(pos)
    a_feynman = np.sqrt(2.0 / (np.sqrt(3.0) * Om / np.pi))
    print(f'\n    Final state: {len(pos)} vortices, charges {sorted(set(q.tolist()))}')
    print(f'    Nearest-neighbour spacing a = {a:.3f} ± {a_std:.3f}   Feynman: {a_feynman:.3f}  '
          f'({100 * (a / a_feynman - 1):+.1f}%)')

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(wf.probability_density.T, origin='lower', extent=extent, cmap='inferno')
    ax.scatter(pos[:, 0], pos[:, 1], s=60, facecolors='none', edgecolors='#4fd8ff', linewidths=1.2,
               label=f'{len(pos)} detected vortices')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_title(f'Vortex lattice at t = {solver.t:.0f}\nspacing {a:.2f} ± {a_std:.2f} (Feynman {a_feynman:.2f})',
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig('output/11_vortex_lattice_2d.png', dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(times, counts, color='#ffb000', lw=2, label='vortices in the condensate')
    ax1.set_xlabel('t')
    ax1.set_ylabel('number of vortices', color='#ffb000')
    ax2 = ax1.twinx()
    ax2.plot(times, Lz, color='#4fd8ff', lw=1.5, label='⟨L_z⟩ per particle')
    ax2.set_ylabel('⟨L_z⟩ / Nℏ', color='#4fd8ff')
    ax1.set_title('Vortex nucleation and lattice formation', fontweight='bold')
    fig.tight_layout()
    fig.savefig('output/11_vortex_count.png', dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ Part 2: 3D
    # Match the 3D condensate to the 2D one in the Thomas–Fermi sense: with a
    # parabolic profile along z of half-length Z, g3 = g2 · 4Z/3 gives the same
    # peak chemical potential in the z = 0 plane.
    omega_z = 0.5
    mu_peak = g2 * float(gaussian_filter(wf.probability_density, 1.0 / grid.dx).max())
    Z = np.sqrt(2.0 * mu_peak) / omega_z
    g3 = g2 * 4.0 * Z / 3.0
    grid3 = Grid3D(128, -10.0, 10.0, 128, -10.0, 10.0, 64, -(Z + 3.0), Z + 3.0)
    trap3 = HarmonicTrap((np.sqrt(1.0 + eps_x), np.sqrt(1.0 + eps_y), omega_z))
    profile_z = np.sqrt(np.clip(1.0 - grid3.z ** 2 / Z ** 2, 0.0, None))
    wf3 = WaveFunction3D(grid3, wf.psi[:, :, None] * profile_z[None, None, :])
    wf3.normalize()
    print(f'\n[2] 3D: cigar trap ω = (1.015, 1.044, {omega_z}), g = {g3:.0f}, grid {grid3.N_x}×{grid3.N_y}×{grid3.N_z}; '
          f'stationary state in the rotating frame')
    # A short imaginary-time run smooths the extruded 2D seed; the energy minimiser does the rest
    wf3, info_it = ground_state(grid3, trap3, dtau=0.02, nonlinearity=g3, rotation=Om, initial=wf3,
                                tol=0.0, max_steps=1000, check_interval=500)
    print(f'    imaginary time (dτ = 0.02, {info_it.steps} steps): residual {info_it.residual:.2e}')

    def progress(it, residual, energy):
        print(f'    conjugate gradient {it:5d}: residual {residual:.2e}, E = {energy:.10f}', flush=True)

    wf3, info3 = ground_state_pcg(grid3, trap3, initial=wf3, nonlinearity=g3, rotation=Om, tol=1e-6,
                                  max_iter=15000, callback=progress, callback_interval=500)
    print(f'    converged = {info3.converged} after {info3.steps} iterations: E = {info3.energy:.10f}, '
          f'μ = {info3.chemical_potential:.10f}, residual ‖(H−ΩL_z)ψ − μψ‖ = {info3.residual:.2e}')
    k0 = int(np.argmin(np.abs(grid3.z)))
    mid = WaveFunction2D(Grid2D(grid3.N_x, -10.0, 10.0, grid3.N_y, -10.0, 10.0), wf3.psi[:, :, k0])
    pos3, q3 = vortex_positions(mid)
    a3, a3_std = lattice_spacing(pos3)
    print(f'    vortex lines crossing z = 0: {len(pos3)};  spacing {a3:.3f} ± {a3_std:.3f} '
          f'(Feynman {a_feynman:.3f}, {100 * (a3 / a_feynman - 1):+.1f}%)')
    # Straightness: follow each line slice by slice from z = 0 out to |z| = 0.6 Z. A line is
    # lost where no vortex lies within half a lattice spacing of its last position, i.e. it
    # has left the region where vortices are detected (density above 20% of the slice maximum).
    shift, followed, outward = np.zeros(len(pos3)), np.ones(len(pos3), dtype=bool), np.ones(len(pos3), dtype=bool)
    for step in (1, -1):
        current, alive, k = pos3.copy(), np.ones(len(pos3), dtype=bool), k0
        while abs(grid3.z[k + step]) <= 0.6 * Z:
            k += step
            p, _ = vortex_positions(WaveFunction2D(mid.grid, wf3.psi[:, :, k]))
            d, j = cKDTree(p).query(current)
            alive &= d < 0.5 * a3
            current[alive] = p[j[alive]]
            shift[alive] = np.maximum(shift[alive], np.hypot(*(current[alive] - pos3[alive]).T))
        followed &= alive
        outward &= np.sum((current - pos3) * pos3, axis=1) >= 0.0      # moved away from the axis
    radius = np.hypot(pos3[:, 0], pos3[:, 1])
    inner = radius < 0.7 * radius.max()
    print(f'    {followed.sum()} of {len(pos3)} lines followed through every slice with |z| ≤ 0.6 Z; largest '
          f'lateral shift from the z = 0 position: inner lines {shift[inner].max():.2f}, outer ring '
          f'{shift[~inner].max():.2f}{" (outwards)" if outward.all() else ""} (grid spacing {grid3.dx:.3f})')
    if not followed.all():
        print(f'    {(~followed).sum()} outermost line(s) leave the dense part of the cloud before |z| = 0.6 Z')

    rho = wf3.probability_density
    envelope = gaussian_filter(rho, sigma=(1.0 / grid3.dx, 1.0 / grid3.dy, 0.0))
    inside = envelope > 0.25 * envelope.max()
    cores = np.where(inside, rho / np.maximum(envelope, 1e-300), 1.0)
    overlays = [{'field': cores, 'level': 0.3, 'color': '#4fd8ff', 'alpha': 1.0}]
    render_kwargs = dict(levels=(0.04,), alphas=(0.16,), color_by='level', zoom=1.15)
    title3 = 'Vortex lines in a rotating BEC'
    note3 = 'tubes: vortex cores (density < 30% of its local average); shell: cloud surface'
    renderer = IsosurfaceRenderer(grid3, **render_kwargs)
    renderer.save(wf3.psi, 'output/11_vortex_lines_3d.png', elev=28, azim=-55, title=title3,
                  note=note3, overlays=overlays, dpi=150)
    renderer.close()
    n_frames = 72
    with IsosurfaceAnimation('output/11_vortex_lines_3d.mp4', grid3, n_frames, fps=24, dpi=100,
                             elev=30.0, elev_end=12.0, azim_start=-60.0, orbit_degrees=180.0,
                             title=title3, note=note3, **render_kwargs) as anim:
        for _ in range(n_frames):
            anim.add_frame(wf3.psi, overlays=overlays)
    print('    Saved: output/11_vortex_lines_3d.png, output/11_vortex_lines_3d.mp4 (orbiting camera)')
    print('\nPlots saved: output/11_vortex_lattice_2d.png, output/11_vortex_count.png')


if __name__ == '__main__':
    main()
