"""
Deterministic performance benchmark: CPU vs GPU scaling.
No randomness. Results are reproducible across runs.
"""
import time
import json
from pathlib import Path
from quantumlab.core.grid import Grid3D
from quantumlab.core.wavefunction import WaveFunction3D
from quantumlab.potentials.barriers import Gaussian3DBarrier
from quantumlab.solvers.split_step_3d import SplitStep3DSolver
from quantumlab.solvers.split_step_3d_gpu import SplitStep3DGPU, CUPY_AVAILABLE


def benchmark(N: int, steps: int = 50):
    L = 60.0
    grid = Grid3D(N, -L / 2, L / 2, N, -L / 2, L / 2, N, -L / 2, L / 2)
    wf = WaveFunction3D.gaussian(
        grid, x0=-15.0, y0=0.0, z0=0.0,
        k0_x=3.0, k0_y=0.0, k0_z=0.0,
        sigma_x=3.0, sigma_y=3.0, sigma_z=3.0
    )
    potential = Gaussian3DBarrier(V0=5.0, width=4.0)

    # CPU
    solver_cpu = SplitStep3DSolver(grid, potential, dt=0.02)
    t0 = time.perf_counter()
    wf_cpu = wf
    for _ in range(steps):
        wf_cpu = solver_cpu.step(wf_cpu)
    t_cpu = time.perf_counter() - t0

    # GPU
    if CUPY_AVAILABLE:
        solver_gpu = SplitStep3DGPU(grid, potential, dt=0.02)
        t0 = time.perf_counter()
        wf_gpu = wf
        for _ in range(steps):
            wf_gpu = solver_gpu.step(wf_gpu)
        t_gpu = time.perf_counter() - t0
        speedup = t_cpu / t_gpu
        backend = solver_gpu.backend
    else:
        t_gpu = None
        speedup = 0.0
        backend = "unavailable"

    return {
        'N': N,
        'grid_points': N ** 3,
        'cpu_time': t_cpu,
        'gpu_time': t_gpu,
        'speedup': speedup,
        'backend': backend,
        'steps': steps,
    }


def main():
    sizes = [32, 64, 96, 128, 160, 192]
    results = []

    print("QuantumLab 3D Performance Benchmark")
    print("=" * 50)
    for N in sizes:
        print(f"\nBenchmarking N={N} ({N ** 3:,} grid points)...")
        result = benchmark(N, steps=50)
        results.append(result)
        print(f"  CPU: {result['cpu_time']:.3f}s")
        if result['gpu_time']:
            print(f"  GPU: {result['gpu_time']:.3f}s  ({result['speedup']:.1f}x)")
        else:
            print(f"  GPU: unavailable")

    out = Path('benchmarks/results.json')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")


if __name__ == '__main__':
    main()
