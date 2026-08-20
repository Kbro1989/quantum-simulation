"""
Deterministic performance benchmark: CPU vs GPU scaling with warm-up, multi-run statistics, and correctness validation.
No randomness. Results are reproducible across runs.
"""
import json
import time
from pathlib import Path
import numpy as np

from quantumlab.core.grid import Grid3D
from quantumlab.core.wavefunction import WaveFunction3D
from quantumlab.potentials.barriers import Gaussian3DBarrier
from quantumlab.solvers.split_step_3d import SplitStep3DSolver
from quantumlab.solvers.split_step_3d_gpu import SplitStep3DGPU, CUPY_AVAILABLE


def benchmark(N: int, steps: int = 50, num_runs: int = 3, warmup_steps: int = 3):
    L = 60.0
    grid = Grid3D(N, -L / 2, L / 2, N, -L / 2, L / 2, N, -L / 2, L / 2)
    wf = WaveFunction3D.gaussian(
        grid, x0=-15.0, y0=0.0, z0=0.0,
        k0_x=3.0, k0_y=0.0, k0_z=0.0,
        sigma_x=3.0, sigma_y=3.0, sigma_z=3.0
    )
    potential = Gaussian3DBarrier(V0=5.0, width=4.0)

    # 1. CPU Warm-up & Timing
    solver_cpu = SplitStep3DSolver(grid, potential, dt=0.02)
    wf_cpu = wf
    for _ in range(warmup_steps):
        wf_cpu = solver_cpu.step(wf_cpu)

    cpu_times = []
    for _ in range(num_runs):
        wf_cpu = wf.copy()
        t0 = time.perf_counter()
        for _ in range(steps):
            wf_cpu = solver_cpu.step(wf_cpu)
        cpu_times.append(time.perf_counter() - t0)

    t_cpu_mean = float(np.mean(cpu_times))
    t_cpu_std = float(np.std(cpu_times))

    # 2. GPU Warm-up, Timing & Correctness Validation
    if CUPY_AVAILABLE:
        solver_gpu = SplitStep3DGPU(grid, potential, dt=0.02)
        # GPU Warm-up
        wf_gpu = wf
        for _ in range(warmup_steps):
            wf_gpu = solver_gpu.step(wf_gpu)

        gpu_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            wf_gpu = solver_gpu.step_sequence(wf.copy(), steps=steps)
            gpu_times.append(time.perf_counter() - t0)

        t_gpu_mean = float(np.mean(gpu_times))
        t_gpu_std = float(np.std(gpu_times))
        speedup = t_cpu_mean / t_gpu_mean if t_gpu_mean > 0 else 0.0
        backend = solver_gpu.backend

        # Numerical correctness parity validation
        rel_diff = float(np.linalg.norm(wf_cpu.psi - wf_gpu.psi) / (np.linalg.norm(wf_cpu.psi) + 1e-30))
    else:
        t_gpu_mean = None
        t_gpu_std = None
        speedup = 0.0
        backend = "unavailable"
        rel_diff = 0.0

    memory_mb = (N ** 3 * 16) / (1024 ** 2)  # complex128 array footprint in MB

    return {
        'N': N,
        'grid_points': N ** 3,
        'memory_mb': round(memory_mb, 2),
        'cpu_time_mean': round(t_cpu_mean, 4),
        'cpu_time_std': round(t_cpu_std, 4),
        'gpu_time_mean': round(t_gpu_mean, 4) if t_gpu_mean else None,
        'gpu_time_std': round(t_gpu_std, 4) if t_gpu_std else None,
        'speedup': round(speedup, 2),
        'backend': backend,
        'relative_error': round(rel_diff, 8),
        'steps': steps,
        'num_runs': num_runs,
        'warmup_steps': warmup_steps,
    }


def main():
    sizes = [32, 64, 96, 128, 160, 192]
    results = []

    print("QuantumLab 3D Research-Grade Performance Benchmark")
    print("=" * 60)
    for N in sizes:
        print(f"\nBenchmarking N={N} ({N ** 3:,} grid points, ~{(N**3 * 16)/(1024**2):.1f} MB)...")
        result = benchmark(N, steps=50, num_runs=3, warmup_steps=3)
        results.append(result)
        print(f"  CPU: {result['cpu_time_mean']:.3f}s ± {result['cpu_time_std']:.4f}s")
        if result['gpu_time_mean']:
            print(f"  GPU: {result['gpu_time_mean']:.3f}s ± {result['gpu_time_std']:.4f}s ({result['speedup']:.1f}x speedup)")
            print(f"  Relative Error ||ψ_cpu - ψ_gpu||: {result['relative_error']:.2e}")
        else:
            print("  GPU: unavailable")

    out = Path('benchmarks/results.json')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")


if __name__ == '__main__':
    main()
