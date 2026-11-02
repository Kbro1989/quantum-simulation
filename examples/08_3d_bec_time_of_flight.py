"""
Example 08: Bose–Einstein condensate in 3D — ground state and time of flight.

1. The Gross–Pitaevskii ground state in a cigar-shaped trap
   (ω_x ≪ ω_y = ω_z) is found by imaginary-time propagation, starting
   from the Thomas–Fermi profile.
2. The trap is switched off. The tightly confined directions expand fastest,
   so the cloud turns from a cigar into a pancake: the aspect-ratio inversion
   that was one of the first signatures of BEC (1995).
3. The rms widths are compared with the Castin–Dum scaling solution
   λ̈_j = ω_j² / (λ_j λ_x λ_y λ_z), which is exact in the Thomas–Fermi limit.
   The quantum pressure neglected by that limit causes small deviations,
   which are printed below.

The isosurfaces are coloured by the phase arg ψ. During free expansion the
phase becomes ≈ Σ_j m λ̇_j r_j² / (2ℏλ_j); its gradient is the velocity field,
visible as concentric colour bands.

Outputs (in output/):
    08_bec_tof_3d.mp4           orbiting-camera isosurface animation
    08_bec_ground_state.png     ground state (3D)
    08_bec_tof_final.png        expanded cloud (3D)
    08_bec_aspect_ratio.png     widths vs Castin–Dum prediction
    08_bec_tof_interactive.html rotatable 3D view with a time slider
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from quantumlab.core.grid import Grid3D
from quantumlab.core.wavefunction import WaveFunction3D
from quantumlab.potentials import HarmonicTrap
from quantumlab.stationary import ground_state
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.observables import (
    position_uncertainty, y_uncertainty, z_uncertainty,
    kinetic_energy_expectation, potential_energy_expectation, interaction_energy,
)
from quantumlab.visualization.volume_3d import IsosurfaceAnimation, IsosurfaceRenderer, export_interactive_html
from quantumlab.visualization.style import set_style


def thomas_fermi(grid, omegas, g):
    """TF density n = (μ − V)/g with μ fixed by normalisation (3D harmonic trap)."""
    w_bar3 = np.prod(omegas)
    mu = 0.5 * (15.0 * g * w_bar3 / (4.0 * np.pi)) ** 0.4
    V = HarmonicTrap(omegas).evaluate(grid)
    return mu, np.sqrt(np.clip(mu - V, 0.0, None) / g).astype(complex)


def castin_dum(omegas, t_eval):
    def rhs(_, s):
        lam, dlam = s[:3], s[3:]
        return np.concatenate([dlam, np.asarray(omegas) ** 2 / (lam * np.prod(lam))])
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), [1, 1, 1, 0, 0, 0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.y[:3]


def widths(wf):
    return np.array([position_uncertainty(wf), y_uncertainty(wf), z_uncertainty(wf)])


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # By t = 2.5 the tight axes expand ~6× and the weak axis ~1.5×. The box must keep
    # the cloud away from its faces: the FFT grid is periodic, and density reaching
    # a face would re-enter on the opposite side.
    grid = Grid3D(150, -15.0, 15.0, 176, -17.5, 17.5, 176, -17.5, 17.5)
    omegas = (0.7, 2.5, 2.5)
    g = 600.0                        # g = 4πℏ²aN/m (ψ normalised to 1)
    dt, t_flight, frame_every = 0.005, 2.5, 6
    trap = HarmonicTrap(omegas)

    os.makedirs('output', exist_ok=True)
    print('--- Example 08: 3D BEC ground state and time-of-flight expansion ---')
    print(f'Grid {grid.N_x}×{grid.N_y}×{grid.N_z} = {np.prod(grid.shape):,} points, '
          f'trap ω = {omegas}, g = {g:g}')

    mu_tf, psi_tf = thomas_fermi(grid, omegas, g)
    print(f'\n[1] Imaginary-time relaxation from the Thomas–Fermi guess (μ_TF = {mu_tf:.3f})')
    wf0, info = ground_state(grid, trap, dtau=0.005, nonlinearity=g, initial=psi_tf, tol=1e-10)
    T = kinetic_energy_expectation(wf0)
    V = potential_energy_expectation(wf0, trap)
    E_int = interaction_energy(wf0, g)
    print(f'  converged = {info.converged} after {info.steps} steps')
    print(f'  E/N = {info.energy:.6f}   μ = {info.chemical_potential:.6f}   residual ‖Hψ−μψ‖ = {info.residual:.1e}')
    print(f'  virial 2T − 2V + 3E_int = {2 * T - 2 * V + 3 * E_int:+.2e}  (exact ground state: 0)')
    print(f'  kinetic fraction T/E = {T / info.energy:.3f}  (Thomas–Fermi limit: 0)')

    render_kwargs = dict(levels=(0.6, 0.25, 0.05), color_by='phase', zoom=1.25)
    renderer = IsosurfaceRenderer(grid, **render_kwargs)
    renderer.save(wf0.psi, 'output/08_bec_ground_state.png', t=0.0, elev=25, azim=-60,
                  title='BEC ground state in a cigar-shaped trap', dpi=150)
    renderer.close()

    print(f'\n[2] Time of flight: trap off, expansion with interactions for t = {t_flight}')
    solver = SplitOperatorSolver(grid, None, dt, nonlinearity=g, order=2)
    num_steps = int(round(t_flight / dt))
    times, sigmas = [0.0], [widths(wf0)]
    snapshots_html, times_html = [wf0.psi], [0.0]
    wf = wf0
    title = 'BEC time of flight: cigar → pancake'
    with IsosurfaceAnimation('output/08_bec_tof_3d.mp4', grid, num_steps // frame_every + 1, fps=20,
                             dpi=100, elev=25.0, azim_start=-70.0, orbit_degrees=100.0,
                             title=title, **render_kwargs) as anim:
        anim.add_frame(wf.psi, 0.0)
        for step in range(1, num_steps + 1):
            wf = solver.step(wf)
            if step % frame_every == 0:
                anim.add_frame(wf.psi, solver.t)
                times.append(solver.t)
                sigmas.append(widths(wf))
            if step % (5 * frame_every) == 0:
                snapshots_html.append(wf.psi.copy())
                times_html.append(solver.t)
    print('Animation saved: output/08_bec_tof_3d.mp4')
    print(f'  norm after flight = {wf.norm():.12f}')
    rho = wf.probability_density
    faces = [rho[[0, -1]].max(), rho[:, [0, -1]].max(), rho[:, :, [0, -1]].max()]
    print('  max density on the box faces / peak (periodic wrap-around check): '
          + ', '.join(f'{a} {f / rho.max():.1e}' for a, f in zip('xyz', faces)))

    renderer = IsosurfaceRenderer(grid, **render_kwargs)
    renderer.save(wf.psi, 'output/08_bec_tof_final.png', t=solver.t, elev=25, azim=-30,
                  title='BEC after time of flight', dpi=150)
    renderer.close()
    export_interactive_html(snapshots_html, times_html, grid, 'output/08_bec_tof_interactive.html',
                            level=0.2, stride=2, title='BEC time of flight')
    print('Interactive view saved: output/08_bec_tof_interactive.html')

    # ---------------------------------------------------------- validation
    times = np.array(times)
    sigmas = np.array(sigmas)
    lam = castin_dum(omegas, times)
    ratio_sim = sigmas[:, 1] / sigmas[:, 0]
    ratio_cd = ratio_sim[0] * lam[1] / lam[0]
    print('\n  t      σ_y/σ_x (sim)   Castin–Dum   deviation')
    for i in range(0, len(times), max(1, len(times) // 7)):
        print(f'  {times[i]:4.2f}   {ratio_sim[i]:10.4f}   {ratio_cd[i]:10.4f}   '
              f'{100 * (ratio_sim[i] - ratio_cd[i]) / ratio_cd[i]:+6.2f}%')
    dev = sigmas / sigmas[0] / lam.T - 1.0
    print(f'  max |σ_j(t)/σ_j(0) − λ_j(t)| / λ_j(t) over the flight: '
          f'x {100 * np.abs(dev[:, 0]).max():.2f}%, y {100 * np.abs(dev[:, 1]).max():.2f}%, '
          f'z {100 * np.abs(dev[:, 2]).max():.2f}%')
    if ratio_sim.max() > 1.0:
        t_sim = np.interp(1.0, ratio_sim, times)
        t_cd = np.interp(1.0, ratio_cd, times)
        print(f'  aspect-ratio inversion σ_y = σ_x at t = {t_sim:.3f} (Castin–Dum: {t_cd:.3f})')
    else:
        print('  no aspect-ratio inversion within this flight time')

    set_style('dark')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, (lbl, c) in enumerate(zip(('x (weak axis)', 'y', 'z'), ('#ffb000', '#7fd4ff', '#ff5fa2'))):
        axes[0].plot(times, sigmas[:, j], color=c, lw=2, label=f'σ_{lbl[0]} simulation')
        axes[0].plot(times, sigmas[0, j] * lam[j], color=c, ls='--', lw=1.2,
                     label=f'σ_{lbl[0]} Castin–Dum' if j != 2 else None)
    axes[0].set_xlabel('time of flight t')
    axes[0].set_ylabel('rms width')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Cloud widths', fontweight='bold')
    axes[1].plot(times, ratio_sim, color='#ffb000', lw=2, label='simulation')
    axes[1].plot(times, ratio_cd, color='#7fd4ff', ls='--', lw=1.5, label='Castin–Dum (TF scaling)')
    axes[1].axhline(1.0, color='#888888', lw=0.8, ls=':')
    axes[1].set_xlabel('time of flight t')
    axes[1].set_ylabel('aspect ratio σ_y / σ_x')
    axes[1].legend(fontsize=9)
    axes[1].set_title('Aspect-ratio inversion', fontweight='bold')
    fig.tight_layout()
    fig.savefig('output/08_bec_aspect_ratio.png', dpi=150)
    plt.close(fig)
    print('\nPlots saved: output/08_bec_ground_state.png, output/08_bec_tof_final.png, '
          'output/08_bec_aspect_ratio.png')


if __name__ == '__main__':
    main()
