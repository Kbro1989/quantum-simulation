import os
import numpy as np
from quantumlab.core.grid import Grid2D
from quantumlab.core.wavefunction import WaveFunction2D
from quantumlab.potentials.barriers import Gaussian2DBarrier
from quantumlab.solvers.split_step_2d import SplitStep2DSolver
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.observables.expectation import (
    position_expectation, position_uncertainty, y_expectation, y_uncertainty
)
from quantumlab.visualization.plots_2d import plot_density_2d, plot_density_snapshots_2d

def main():
    Lx, Ly = 80.0, 80.0
    Nx, Ny = 256, 256
    dt = 0.02
    num_steps = 500
    snapshot_steps = [0, 100, 200, 300, 400, 500]

    x0, y0   = -20.0,  0.0
    k0_x     =  3.5
    sigma_x, sigma_y = 4.0, 4.0

    V0           = 6.0
    barrier_width = 4.0
    boundary_width = 10.0

    os.makedirs('output', exist_ok=True)
    print('--- Example 05: 2D Gaussian Wave Packet scattering off Isotropic Gaussian Barrier ---')
    print('Initializing 2D grid and wave function...')

    grid = Grid2D(Nx, -Lx / 2, Lx / 2, Ny, -Ly / 2, Ly / 2)
    wf   = WaveFunction2D.gaussian(grid, x0, y0, k0_x, 0.0, sigma_x, sigma_y)

    print('Setting up isotropic Gaussian2DBarrier and 2D SSFM solver...')
    potential = Gaussian2DBarrier(V0=V0, width=barrier_width, x0=0.0, y0=0.0)
    solver    = SplitStep2DSolver(grid, potential, dt, hbar=1.0, m=1.0)
    abl       = AbsorbingBoundaryLayer(grid, boundary_width, order=3)

    snapshots  = []
    snap_times = []

    print(f'Running 2D simulation for {num_steps} steps...')
    wf_current = wf
    if 0 in snapshot_steps:
        snapshots.append(wf_current)
        snap_times.append(0.0)

    for step in range(1, num_steps + 1):
        wf_current = solver.step(wf_current)
        abl.apply(wf_current)

        if step in snapshot_steps:
            snapshots.append(wf_current)
            snap_times.append(step * dt)

        if step % 100 == 0:
            norm  = wf_current.norm()
            x_exp = position_expectation(wf_current)
            y_exp = y_expectation(wf_current)
            dx    = position_uncertainty(wf_current)
            dy    = y_uncertainty(wf_current)
            print(f'  Step {step:>4}/{num_steps}: Norm={norm:.6f}  '
                  f'⟨x⟩={x_exp:+.3f}  ⟨y⟩={y_exp:+.3f}  '
                  f'Δx={dx:.3f}  Δy={dy:.3f}')

    print('\nGenerating and saving plots...')

    # Final-state density heatmap with barrier contour
    plot_density_2d(
        wf_current,
        potential=potential,
        title='2D Gaussian Barrier: Final Probability Density $|\\Psi(x,y)|^2$',
        save_path='output/05_2d_gaussian_barrier_final.png',
        show=False,
        theme='light',
    )

    # Dark-theme version for the README gallery
    plot_density_2d(
        wf_current,
        potential=potential,
        title='2D Gaussian Barrier: Final Probability Density $|\\Psi(x,y)|^2$',
        save_path='output/05_2d_gaussian_barrier_final_dark.png',
        show=False,
        theme='dark',
    )

    # Snapshot grid
    plot_density_snapshots_2d(
        snapshots,
        snap_times,
        potential=potential,
        ncols=3,
        title='2D Gaussian Barrier: Wave Packet Evolution',
        save_path='output/05_2d_gaussian_barrier_snapshots.png',
        show=False,
        theme='dark',
    )

    print("Plots saved in the 'output/' directory.")

if __name__ == '__main__':
    main()
