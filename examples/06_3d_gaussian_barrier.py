import os
import numpy as np
from quantumlab.core.grid import Grid3D
from quantumlab.core.wavefunction import WaveFunction3D
from quantumlab.potentials.barriers import Gaussian3DBarrier
from quantumlab.solvers.split_step_3d import SplitStep3DSolver
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.visualization.plots_2d import plot_orthogonal_slices_3d

import sys

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    L  = 60.0
    N  = 64           # 64³ grid — manageable on a laptop (~128 MB)
    dt = 0.02
    num_steps    = 300
    snapshot_steps = [0, 100, 200, 300]

    x0, y0, z0    = -15.0, 0.0, 0.0
    k0_x          =  3.0
    sigma         =  3.0
    V0            =  5.0
    barrier_width =  4.0
    boundary_width = 8.0

    os.makedirs('output', exist_ok=True)
    print('--- Example 06: 3D Gaussian Wave Packet scattering off Spherical Gaussian Barrier ---')
    print(f'Grid: {N}³ points over [{-L/2:.0f}, {L/2:.0f}]³')

    grid = Grid3D(N, -L/2, L/2, N, -L/2, L/2, N, -L/2, L/2)
    wf   = WaveFunction3D.gaussian(
        grid,
        x0=x0, y0=y0, z0=z0,
        k0_x=k0_x, k0_y=0.0, k0_z=0.0,
        sigma_x=sigma, sigma_y=sigma, sigma_z=sigma,
    )

    print('Setting up Gaussian3DBarrier and 3D SSFM solver...')
    potential = Gaussian3DBarrier(V0=V0, width=barrier_width, x0=0.0, y0=0.0, z0=0.0)
    solver    = SplitStep3DSolver(grid, potential, dt, hbar=1.0, m=1.0, workers=-1)
    abl       = AbsorbingBoundaryLayer(grid, boundary_width, order=3)

    print(f'Running 3D simulation for {num_steps} steps...')
    wf_current = wf
    for step in range(1, num_steps + 1):
        wf_current = solver.step(wf_current)
        abl.apply(wf_current)

        if step % 50 == 0:
            norm = wf_current.norm()
            prob = wf_current.probability_density
            cx = int(np.round(
                np.sum(grid.X * prob) / (prob.sum() + 1e-30)
            ))
            print(f'  Step {step:>4}/{num_steps}: Norm={norm:.6f}  '
                  f'⟨x⟩ ≈ {grid.x[np.clip(cx, 0, N-1)]:.2f}')

        if step in snapshot_steps:
            plot_orthogonal_slices_3d(
                wf_current,
                potential=potential,
                title=f'3D Gaussian Barrier — Orthogonal Slices  (t = {step * dt:.2f})',
                save_path=f'output/06_3d_slices_t{step:04d}.png',
                show=False,
                theme='dark',
            )
            print(f'  Snapshot saved: output/06_3d_slices_t{step:04d}.png')

    print('\nAll plots saved in the output/ directory.')
    print('Open output/06_3d_slices_t*.png to see the 3D wave packet scatter.')

if __name__ == '__main__':
    main()
