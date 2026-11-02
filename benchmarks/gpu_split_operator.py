"""
CPU vs GPU timing of SplitOperatorSolver on 3D grids, with a correctness
check of the GPU result against the CPU result.

Run from the project root:
    python benchmarks/gpu_split_operator.py [--sizes 64 96 128] [--steps 50]

Without CuPy and a CUDA device only the CPU timings are printed.
"""
import argparse
import sys
import time
import numpy as np
from quantumlab.core.grid import Grid3D
from quantumlab.core.wavefunction import WaveFunction3D
from quantumlab.potentials import HarmonicTrap
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.solvers.backends import cupy_available

CONFIG = dict(order=2, nonlinearity=200.0, rotation=0.7)


def timed_run(grid, backend, steps):
    wf0 = WaveFunction3D.gaussian(grid, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 1.5, 1.5, 1.5)
    solver = SplitOperatorSolver(grid, HarmonicTrap((1.0, 1.0, 0.8)), 0.005, backend=backend, **CONFIG)
    solver.evolve(wf0, 2)                        # warm-up (FFT plans, CUDA kernels)
    t0 = time.perf_counter()
    wf = solver.evolve(wf0, steps)               # includes the final device → host copy
    return (time.perf_counter() - t0) / steps, wf


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 96, 128])
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()

    gpu = cupy_available()
    print(f'SplitOperatorSolver, 3D, {CONFIG}, {args.steps} steps')
    print('CUDA device: ' + ('available' if gpu else 'not available (CPU timings only)'))
    print(f'{"grid":>8} {"points":>12} {"CPU ms/step":>12} {"GPU ms/step":>12} {"speed-up":>9} {"max |Δψ|":>10}')
    for n in args.sizes:
        grid = Grid3D(n, -10.0, 10.0, n, -10.0, 10.0, n, -10.0, 10.0)
        t_cpu, wf_cpu = timed_run(grid, 'numpy', args.steps)
        row = f'{n:>6}³ {n ** 3:>12,} {1e3 * t_cpu:>12.2f}'
        if gpu:
            t_gpu, wf_gpu = timed_run(grid, 'cupy', args.steps)
            diff = float(np.max(np.abs(wf_gpu.psi - wf_cpu.psi)))
            row += f' {1e3 * t_gpu:>12.2f} {t_cpu / t_gpu:>8.1f}× {diff:>10.1e}'
        print(row)


if __name__ == '__main__':
    main()
