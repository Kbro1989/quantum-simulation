import os
import numpy as np
from quantumlab.core.grid import Grid1D, Grid2D
from quantumlab.core.wavefunction import WaveFunction2D
from quantumlab.potentials.barriers import GaussianBarrier
from quantumlab.solvers.split_step_2d import SplitStep2DSolver
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.observables.expectation import position_expectation, position_uncertainty
from quantumlab.visualization.plots_3d import plot_space_time_3d

def main():
    Lx, Ly = 100.0, 100.0
    Nx, Ny = 256, 256
    dt = 0.02
    num_steps = 400

    x0, y0 = -25.0, 0.0
    k0_x, k0_y = 3.5, 0.0
    sigma_x, sigma_y = 4.0, 4.0

    V0 = 6.0
    barrier_width = 1.5
    barrier_center = 2.0
    boundary_width = 10.0

    os.makedirs('output', exist_ok=True)
    print('--- Example 05: 2D Gaussian Wave Packet scattering off Gaussian Barrier ---')
    print('Initializing 2D grid and wave function...')
    grid = Grid2D(Nx, -Lx / 2, Lx / 2, Ny, -Ly / 2, Ly / 2)
    wf = WaveFunction2D.gaussian(grid, x0, y0, k0_x, k0_y, sigma_x, sigma_y)

    print('Setting up Gaussian barrier potential and 2D SSFM solver...')
    potential = GaussianBarrier(V0, barrier_width, barrier_center)
    solver = SplitStep2DSolver(grid, potential, dt, hbar=1.0, m=1.0)
    abl = AbsorbingBoundaryLayer(grid, boundary_width, order=3)

    mid_y = Ny // 2
    space_time = np.zeros((num_steps + 1, Nx))
    space_time[0, :] = wf.probability_density[:, mid_y]

    print(f'Running 2D simulation for {num_steps} steps...')
    wf_current = wf
    for step in range(1, num_steps + 1):
        wf_current = solver.step(wf_current)
        abl.apply(wf_current)
        space_time[step, :] = wf_current.probability_density[:, mid_y]
        if step % 100 == 0:
            norm = wf_current.norm()
            x_mean = position_expectation(wf_current)
            dx = position_uncertainty(wf_current)
            print(f'  Step {step}/{num_steps}: Norm={norm:.6f}, ⟨x⟩={x_mean:.3f}, Δx={dx:.3f}')

    print('\nGenerating and saving plots...')
    t_arr = np.linspace(0, num_steps * dt, num_steps + 1)
    plot_space_time_3d(
        Grid1D(Nx, -Lx / 2, Lx / 2),
        t_arr,
        space_time,
        x_range=(-40, 40),
        title='2D Gaussian Barrier: Central Slice Space-Time Evolution',
        save_path='output/05_2d_gaussian_barrier_spacetime.png',
        show=False,
        theme='light'
    )
    print("Plots saved in the 'output/' directory.")

if __name__ == '__main__':
    main()
