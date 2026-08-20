"""
Comprehensive GPU solver tests covering norm conservation, energy conservation,
CPU/GPU parity, fallback path, step_sequence batch mode, and edge cases.
"""
import pytest
import numpy as np
from quantumlab.core.grid import Grid3D
from quantumlab.core.wavefunction import WaveFunction3D
from quantumlab.potentials.barriers import Gaussian3DBarrier
from quantumlab.solvers.split_step_3d_gpu import SplitStep3DGPU, CUPY_AVAILABLE
from quantumlab.solvers.split_step_3d import SplitStep3DSolver
from quantumlab.observables.expectation import total_energy_expectation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_grid():
    return Grid3D(32, -10.0, 10.0, 32, -10.0, 10.0, 32, -10.0, 10.0)


@pytest.fixture
def gaussian_wf(small_grid):
    return WaveFunction3D.gaussian(
        small_grid, x0=-3.0, y0=0.0, z0=0.0,
        k0_x=1.5, k0_y=0.0, k0_z=0.0,
        sigma_x=1.0, sigma_y=1.0, sigma_z=1.0,
    )


@pytest.fixture
def barrier():
    return Gaussian3DBarrier(V0=4.0, width=2.0)


# ---------------------------------------------------------------------------
# Norm Conservation
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
@pytest.mark.parametrize("steps", [10, 30, 100])
def test_3d_gpu_norm_conservation(small_grid, gaussian_wf, barrier, steps):
    solver = SplitStep3DGPU(small_grid, barrier, dt=0.02)
    assert solver.backend == "cupy"

    wf_current = gaussian_wf
    for _ in range(steps):
        wf_current = solver.step(wf_current)
    assert wf_current.norm() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Energy Conservation (time-independent potential => dE/dt = 0)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
def test_3d_gpu_energy_conservation(small_grid, gaussian_wf, barrier):
    solver = SplitStep3DGPU(small_grid, barrier, dt=0.02)
    E_initial = total_energy_expectation(gaussian_wf, barrier)

    wf_current = gaussian_wf
    for _ in range(50):
        wf_current = solver.step(wf_current)

    E_final = total_energy_expectation(wf_current, barrier)
    fractional_drift = abs(E_final - E_initial) / abs(E_initial)
    assert fractional_drift < 1e-3, f"Energy drifted {fractional_drift:.2e}"


# ---------------------------------------------------------------------------
# CPU / GPU Parity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
def test_3d_gpu_matches_cpu(small_grid, gaussian_wf, barrier):
    """
    GPU and CPU solvers must produce numerically identical results.
    Tolerance is relaxed to 1e-5 because CUDA FFTs use different
    floating-point accumulation order than MKL/FFTW.
    """
    solver_gpu = SplitStep3DGPU(small_grid, barrier, dt=0.02)
    solver_cpu = SplitStep3DSolver(small_grid, barrier, dt=0.02)

    wf_gpu = gaussian_wf.copy()
    wf_cpu = gaussian_wf.copy()
    for _ in range(10):
        wf_gpu = solver_gpu.step(wf_gpu)
        wf_cpu = solver_cpu.step(wf_cpu)

    np.testing.assert_allclose(wf_gpu.psi, wf_cpu.psi, rtol=1e-5, atol=1e-10)


# ---------------------------------------------------------------------------
# step_sequence batch mode (zero PCI-E bottleneck)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
def test_step_sequence_matches_individual_steps(small_grid, gaussian_wf, barrier):
    """step_sequence(N) must produce the same final state as N individual step() calls."""
    solver_seq = SplitStep3DGPU(small_grid, barrier, dt=0.02)
    solver_ind = SplitStep3DGPU(small_grid, barrier, dt=0.02)

    wf_seq = solver_seq.step_sequence(gaussian_wf, steps=20)

    wf_ind = gaussian_wf
    for _ in range(20):
        wf_ind = solver_ind.step(wf_ind)

    np.testing.assert_allclose(wf_seq.psi, wf_ind.psi, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# CPU Fallback Path (runs everywhere!)
# ---------------------------------------------------------------------------
def test_cpu_fallback_produces_valid_result():
    """When GPU is unavailable or on CPU path, SplitStep3DGPU falls back to scipy-cpu cleanly."""
    grid = Grid3D(32, -10.0, 10.0, 32, -10.0, 10.0, 32, -10.0, 10.0)
    wf = WaveFunction3D.gaussian(
        grid, x0=-3.0, y0=0.0, z0=0.0,
        k0_x=1.5, k0_y=0.0, k0_z=0.0,
        sigma_x=1.0, sigma_y=1.0, sigma_z=1.0,
    )
    potential = Gaussian3DBarrier(V0=4.0, width=2.0)

    # Force CPU fallback path
    solver = SplitStep3DGPU(grid, potential, dt=0.02, device_id=999)

    wf_current = wf
    for _ in range(20):
        wf_current = solver.step(wf_current)

    assert wf_current.norm() == pytest.approx(1.0, abs=1e-10)
    assert solver.backend == "scipy-cpu"


# ---------------------------------------------------------------------------
# Different Potentials
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
@pytest.mark.parametrize("V0,width", [(1.0, 2.0), (8.0, 1.0), (15.0, 6.0)])
def test_norm_conservation_varied_potentials(small_grid, gaussian_wf, V0, width):
    potential = Gaussian3DBarrier(V0=V0, width=width)
    solver = SplitStep3DGPU(small_grid, potential, dt=0.02)

    wf_current = gaussian_wf
    for _ in range(30):
        wf_current = solver.step(wf_current)
    assert wf_current.norm() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Backend property
# ---------------------------------------------------------------------------
def test_backend_property(small_grid, barrier):
    solver = SplitStep3DGPU(small_grid, barrier, dt=0.02)
    assert solver.backend in ("cupy", "scipy-cpu")
