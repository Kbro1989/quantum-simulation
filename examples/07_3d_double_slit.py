"""
Example 07: Young's double-pinhole experiment in full 3D.

A Gaussian wave packet hits an opaque screen with two circular pinholes.
The transmitted wave forms the two-source interference pattern. The fringe
period is checked against the far-field (Fraunhofer) prediction: the
momentum distribution of the transmitted wave along y is modulated by
cos²(k_y d / 2), whose zeros lie at k_y = ±(2n+1)π/d.

Outputs (in output/):
    07_double_slit_3d.mp4         orbiting-camera isosurface animation
    07_double_slit_3d_final.png   final frame
    07_double_slit_fringes.png    measured vs predicted k_y distribution
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from quantumlab.core.grid import Grid3D
from quantumlab.core.wavefunction import WaveFunction3D
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.potentials import ApertureScreen
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.visualization.volume_3d import IsosurfaceAnimation, IsosurfaceRenderer
from quantumlab.visualization.style import set_style


def fringe_analysis(psi, grid, x_cut, pad=8):
    """k_y distribution of the transmitted wave (x > x_cut), zero-padded along y."""
    psi_T = psi * (grid.X > x_cut)
    n_y = grid.N_y * pad
    phi_y = np.fft.fft(psi_T, n=n_y, axis=1)
    P = np.fft.fftshift(np.sum(np.abs(phi_y) ** 2, axis=(0, 2)))
    k_y = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n_y, grid.dy))
    return k_y, P / P.max()


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # Grid: 128 × 96 × 96 ≈ 1.2 million points
    grid = Grid3D(128, -24.0, 24.0, 96, -16.0, 16.0, 96, -16.0, 16.0)
    dt, num_steps, frame_every = 0.02, 450, 5

    # Incident packet: λ = 2π/k0 ≈ 2.1
    x0, k0, sigma_x, sigma_t = -12.0, 3.0, 2.0, 4.5
    # Screen with two pinholes
    wall_x, separation, pinhole_d = -4.0, 5.0, 1.8
    screen = ApertureScreen(V0=60.0, wall_position=wall_x, wall_thickness=1.0, separation=separation,
                            aperture_width=pinhole_d, aperture='pinhole', smoothing=0.2)
    x_cut = wall_x + 1.0

    os.makedirs('output', exist_ok=True)
    print('--- Example 07: 3D double-pinhole interference ---')
    print(f'Grid {grid.N_x}×{grid.N_y}×{grid.N_z} = {np.prod(grid.shape):,} points, dt = {dt}')
    print(f'k0 = {k0}  (λ = {2 * np.pi / k0:.3f}),  E = {k0 ** 2 / 2:.2f}  vs wall V0 = {screen.V0}')

    wf = WaveFunction3D.gaussian(grid, x0, 0.0, 0.0, k0, 0.0, 0.0, sigma_x, sigma_t, sigma_t)
    solver = SplitOperatorSolver(grid, screen, dt, order=2, absorber=AbsorbingBoundaryLayer(grid, 3.0))

    # The transmitted wave is ~1.5% of the incident peak density. For display only,
    # the region behind the screen is amplified by a fixed, stated factor.
    amplify = 40.0
    display_gain = 1.0 + (np.sqrt(amplify) - 1.0) * 0.5 * (1.0 + np.tanh((grid.X - x_cut) / 0.5))
    peak0 = float(wf.probability_density.max())
    render_kwargs = dict(levels=(0.5, 0.2, 0.06), reference_peak=peak0, potential=screen,
                         color_by='level', zoom=1.2)
    title = "Young's double-pinhole experiment in 3D"
    note = f'density behind the screen amplified ×{amplify:g} for visibility'

    n_frames = num_steps // frame_every + 1
    # Fringes are analysed when the transmitted probability peaks: the wave has
    # fully cleared the screen but has not yet reached the absorbing boundaries.
    best_T, psi_analysis, t_analysis = -1.0, None, None
    showcase_step = 400                      # fringes fully formed, before boundary losses
    psi_showcase, t_showcase = None, None
    with IsosurfaceAnimation('output/07_double_slit_3d.mp4', grid, n_frames, fps=20, dpi=100,
                             elev=62.0, elev_end=36.0, azim_start=-105.0, orbit_degrees=70.0,
                             title=title, note=note, **render_kwargs) as anim:
        anim.add_frame(wf.psi * display_gain, solver.t)
        for step in range(1, num_steps + 1):
            wf = solver.step(wf)
            if step % frame_every == 0:
                anim.add_frame(wf.psi * display_gain, solver.t)
                T = np.sum(wf.probability_density[grid.X > x_cut]) * grid.dx * grid.dy * grid.dz
                if T > best_T:
                    best_T, psi_analysis, t_analysis = T, wf.psi.copy(), solver.t
            if step == showcase_step:
                psi_showcase, t_showcase = wf.psi.copy(), solver.t
            if step % 50 == 0:
                T = np.sum(wf.probability_density[grid.X > x_cut]) * grid.dx * grid.dy * grid.dz
                print(f'  t = {solver.t:5.2f}   norm = {wf.norm():.4f}   transmitted = {T:.4f}')
    print('Animation saved: output/07_double_slit_3d.mp4')

    renderer = IsosurfaceRenderer(grid, **render_kwargs)
    renderer.save(psi_showcase * display_gain, 'output/07_double_slit_3d_final.png', t=t_showcase,
                  elev=38.0, azim=-35.0, title=title, note=note, dpi=150)
    renderer.close()

    # ---------------------------------------------------------- validation
    k_y, P = fringe_analysis(psi_analysis, grid, x_cut)
    minima, _ = find_peaks(-P)
    minima = minima[np.abs(k_y[minima]) < 2.0]
    measured = np.sort(np.abs(k_y[minima]))
    predicted = np.array([(2 * n + 1) * np.pi / separation for n in range(len(measured) // 2)])
    print(f'\nFringe zeros of the transmitted k_y distribution at t = {t_analysis:.2f} '
          f'(transmitted probability {best_T:.4f}); Fraunhofer: ±(2n+1)π/d')
    for n, pred in enumerate(predicted):
        meas = measured[2 * n:2 * n + 2].mean()
        print(f'  n = {n}:  measured |k_y| = {meas:.3f}   predicted = {pred:.3f}   '
              f'deviation = {100 * (meas - pred) / pred:+.1f}%')
    print('  The ideal two-point-source zeros assume identical, uniformly illuminated sources;\n'
          '  finite pinholes in a Gaussian beam deviate slightly from that model.')

    set_style('dark')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(k_y, P, color='#ffb000', lw=1.6, label='simulation (transmitted wave)')
    for n in range(3):
        for sgn in (-1, 1):
            ax.axvline(sgn * (2 * n + 1) * np.pi / separation, color='#7fd4ff', ls=':', lw=1,
                       label='predicted zeros $\\pm(2n+1)\\pi/d$' if (n == 0 and sgn == 1) else None)
    ax.set_xlim(-4, 4)
    ax.set_xlabel('$k_y$')
    ax.set_ylabel('$P(k_y)$ (normalised)')
    ax.set_title('Double-pinhole fringes in momentum space', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig('output/07_double_slit_fringes.png', dpi=150)
    plt.close(fig)
    print('\nPlots saved: output/07_double_slit_3d_final.png, output/07_double_slit_fringes.png')


if __name__ == '__main__':
    main()
