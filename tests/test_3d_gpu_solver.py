import pytest
import numpy as np
from quantumlab.core.grid import Grid3D
from quantumlab.core.wavefunction import WaveFunction3D
from quantumlab.potentials.barriers import Gaussian3DBarrier
from quantumlab.solvers.split_step_3d_gpu import SplitStep3DGPU, CUPY_AVAILABLE
from quantumlab.solvers.split_step_3d import SplitStep3DSolver

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")


def test_3d_gpu_norm_conservation():
    grid = Grid3D(32, -10.0, 10.0, 32, -10.0, 10.0, 32, -10.0, 10.0)
    wf = WaveFunction3D.gaussian(
        grid, x0=-3.0, y0=0.0, z0=0.0,
        k0_x=1.5, k0_y=0.0, k0_z=0.0,
        sigma_x=1.0, sigma_y=1.0, sigma_z=1.0,
    )
    potential = Gaussian3DBarrier(V0=4.0, width=2.0)
    solver = SplitStep3DGPU(grid, potential, dt=0.02)
    assert solver.backend == "cupy"

    wf_current = wf
    for _ in range(30):
        wf_current = solver.step(wf_current)
    assert wf_current.norm() == pytest.approx(1.0, abs=1e-10)


def test_3d_gpu_matches_cpu():
    """
    GPU and CPU solvers must produce numerically identical results.
    Tolerance is relaxed to 1e-5 because CUDA FFTs use different
    floating-point accumulation order than MKL/FFTW.
    """
    grid = Grid3D(32, -10.0, 10.0, 32, -10.0, 10.0, 32, -10.0, 10.0)
    wf = WaveFunction3D.gaussian(
        grid, x0=-3.0, y0=0.0, z0=0.0,
        k0_x=1.5, k0_y=0.0, k0_z=0.0,
        sigma_x=1.0, sigma_y=1.0, sigma_z=1.0,
    )
    potential = Gaussian3DBarrier(V0=4.0, width=2.0)

    solver_gpu = SplitStep3DGPU(grid, potential, dt=0.02)
    solver_cpu = SplitStep3DSolver(grid, potential, dt=0.02)

    wf_gpu = wf.copy()
    wf_cpu = wf.copy()
    for _ in range(10):
        wf_gpu = solver_gpu.step(wf_gpu)
        wf_cpu = solver_cpu.step(wf_cpu)

    np.testing.assert_allclose(wf_gpu.psi, wf_cpu.psi, rtol=1e-5, atol=1e-10)
