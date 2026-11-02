"""
Example 09: Collisions of bright matter-wave solitons (1D Gross–Pitaevskii / NLS).

Two exact sech solitons of iψ_t = −½ψ_xx − |ψ|²ψ move towards each other.
The 1D cubic NLS is integrable, so the solitons emerge from the collision
with their shapes and speeds intact. The outcome depends on their relative
phase: in phase they pass through each other with a bright interference
peak; out of phase (Δφ = π) the density between them vanishes and they
bounce off each other.

Outputs (in output/):
    09_soliton_collisions.png       space-time density for Δφ = 0 and π
    09_soliton_spacetime_3d.png     3D space-time surface (Δφ = 0)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from quantumlab.core.grid import Grid1D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.observables import gross_pitaevskii_energy
from quantumlab.visualization.plots_3d import plot_space_time_3d
from quantumlab.visualization.style import set_style


def two_solitons(grid, kappa, x0, v, dphi):
    left = kappa / np.cosh(kappa * (grid.x + x0)) * np.exp(1j * v * grid.x)
    right = kappa / np.cosh(kappa * (grid.x - x0)) * np.exp(-1j * v * grid.x + 1j * dphi)
    return WaveFunction1D(grid, left + right)


def run(grid, wf, dt, num_steps, record_every, g):
    solver = SplitOperatorSolver(grid, None, dt, nonlinearity=g, order=4)
    rows, times = [wf.probability_density], [0.0]
    for _ in range(num_steps // record_every):
        wf = solver.evolve(wf, record_every)
        rows.append(wf.probability_density)
        times.append(solver.t)
    return wf, np.array(times), np.array(rows)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    grid = Grid1D(2048, -40.0, 40.0)
    kappa, x0, v, g = 1.0, 10.0, 1.0, -1.0
    dt, num_steps, record_every = 0.005, 4000, 20

    os.makedirs('output', exist_ok=True)
    print('--- Example 09: bright soliton collisions (integrable 1D NLS) ---')
    results = {}
    for label, dphi in (('in phase (Δφ = 0)', 0.0), ('out of phase (Δφ = π)', np.pi)):
        wf0 = two_solitons(grid, kappa, x0, v, dphi)
        E0 = gross_pitaevskii_energy(wf0, None, g)
        wf, times, rows = run(grid, wf0, dt, num_steps, record_every, g)
        E1 = gross_pitaevskii_energy(wf, None, g)
        rho = wf.probability_density
        left, right = rho[grid.x < 0], rho[grid.x >= 0]
        print(f'\n  {label}')
        print(f'    norm   {wf0.norm():.10f} → {wf.norm():.10f}')
        print(f'    energy {E0:.10f} → {E1:.10f}')
        print(f'    peak heights after collision: {left.max():.4f}, {right.max():.4f}  (initial κ² = {kappa ** 2:.4f})')
        results[label] = (times, rows)

    set_style('dark')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (label, (times, rows)) in zip(axes, results.items()):
        im = ax.imshow(rows, origin='lower', aspect='auto', cmap='inferno',
                       extent=[grid.x[0], grid.x[-1] + grid.dx, times[0], times[-1]], vmax=2.5 * kappa ** 2)
        ax.set_xlim(-25, 25)
        ax.set_xlabel('x')
        ax.set_title(label, fontweight='bold')
    axes[0].set_ylabel('t')
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label='$|\\psi|^2$')
    fig.suptitle('Elastic collisions of bright solitons', fontsize=14, fontweight='bold')
    fig.savefig('output/09_soliton_collisions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    times, rows = results['in phase (Δφ = 0)']
    plot_space_time_3d(grid, times, rows, x_range=(-20, 20),
                       title='Soliton collision: space-time density', save_path='output/09_soliton_spacetime_3d.png',
                       show=False, theme='dark')
    print('\nPlots saved: output/09_soliton_collisions.png, output/09_soliton_spacetime_3d.png')


if __name__ == '__main__':
    main()
