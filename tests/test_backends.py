"""
GPU backend tests.

CUDA is not available on every machine, so the device code path is also
exercised with a strict emulated device backend: its arrays refuse to mix
with plain NumPy arrays (as CuPy arrays do), so any host/device mix-up in the
solver raises a TypeError. Results must match the NumPy backend exactly.
The real CuPy comparison at the end runs only where a CUDA device exists.
"""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import fft as sp_fft
from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D, WaveFunction3D
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.potentials import HarmonicTrap, GaussianBarrier, DipoleDrive
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.solvers.backends import ArrayBackend, get_backend, cupy_available
from quantumlab.stationary import ground_state


class DeviceArray(np.ndarray):
    """ndarray that, like a CuPy array, refuses implicit mixing with host arrays."""

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        for x in inputs + (out or ()):
            if isinstance(x, np.ndarray) and not isinstance(x, DeviceArray) and x.ndim > 0:
                raise TypeError(f'implicit host/device mixing in {ufunc.__name__}')
        args = tuple(x.view(np.ndarray) if isinstance(x, DeviceArray) else x for x in inputs)
        if out is not None:
            kwargs['out'] = tuple(o.view(np.ndarray) for o in out)
        result = getattr(ufunc, method)(*args, **kwargs)
        if out is not None:
            return out[0] if len(out) == 1 else out
        return result.view(DeviceArray) if isinstance(result, np.ndarray) else result


def _strict_backend():
    calls = {'to_device': 0, 'to_host': 0}

    def require_device(a):
        if not isinstance(a, DeviceArray):
            raise TypeError('expected a device array')
        return a.view(np.ndarray)

    def to_device(a):
        calls['to_device'] += 1
        if isinstance(a, DeviceArray):
            return a
        return np.array(a).view(DeviceArray)

    def to_host(a):
        calls['to_host'] += 1
        return np.array(require_device(a))

    backend = ArrayBackend(
        name='emulated-device',
        xp=SimpleNamespace(exp=np.exp, sum=np.sum),
        fftn=lambda a, axes=None: sp_fft.fftn(require_device(a), axes=axes).view(DeviceArray),
        ifftn=lambda a, axes=None: sp_fft.ifftn(require_device(a), axes=axes).view(DeviceArray),
        to_device=to_device,
        to_host=to_host,
    )
    return backend, calls


def test_emulated_device_array_rejects_host_mixing():
    d = np.ones(3).view(DeviceArray)
    with pytest.raises(TypeError):
        d * np.ones(3)
    assert isinstance(d * 2.0, DeviceArray)


CASES = {
    '1d_barrier_order4': lambda: (Grid1D(256, -20.0, 20.0), GaussianBarrier(3.0, 1.0, 0.0), dict(order=4),
                                  WaveFunction1D.gaussian(Grid1D(256, -20.0, 20.0), -5.0, 2.0, 1.0)),
    '1d_driven': lambda: (Grid1D(128, -15.0, 15.0), DipoleDrive(HarmonicTrap(1.0), 0.4, 0.7), dict(order=4),
                          WaveFunction1D.gaussian(Grid1D(128, -15.0, 15.0), 0.0, 0.0, 1.0)),
    '2d_gpe_rotating': lambda: (Grid2D(48, -8.0, 8.0, 48, -8.0, 8.0), HarmonicTrap(1.0),
                                dict(nonlinearity=20.0, rotation=0.6),
                                WaveFunction2D.gaussian(Grid2D(48, -8.0, 8.0, 48, -8.0, 8.0),
                                                        1.0, 0.0, 0.5, 0.0, 1.0, 1.0)),
    '2d_damped_rotating': lambda: (Grid2D(48, -8.0, 8.0, 48, -8.0, 8.0), HarmonicTrap((1.0, 1.1)),
                                   dict(nonlinearity=20.0, rotation=0.7, damping=0.05),
                                   WaveFunction2D.gaussian(Grid2D(48, -8.0, 8.0, 48, -8.0, 8.0),
                                                           0.5, 0.0, 0.0, 0.0, 1.0, 1.0)),
    '3d_imaginary_rotating': lambda: (Grid3D(24, -6.0, 6.0, 24, -6.0, 6.0, 16, -5.0, 5.0),
                                      HarmonicTrap((1.0, 1.0, 1.5)),
                                      dict(nonlinearity=30.0, rotation=0.5, imaginary_time=True),
                                      WaveFunction3D.gaussian(Grid3D(24, -6.0, 6.0, 24, -6.0, 6.0, 16, -5.0, 5.0),
                                                              0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
}


@pytest.mark.parametrize('case', list(CASES))
def test_device_code_path_matches_numpy(case):
    grid, potential, kwargs, wf0 = CASES[case]()
    backend, calls = _strict_backend()
    ref = SplitOperatorSolver(grid, potential, 0.01, backend='numpy', **kwargs).evolve(wf0, 30)
    dev_solver = SplitOperatorSolver(grid, potential, 0.01, backend=backend, **kwargs)
    out = dev_solver.evolve(wf0, 30)
    assert dev_solver.backend == 'emulated-device'
    np.testing.assert_allclose(out.psi, ref.psi, rtol=0, atol=1e-13)
    # evolve copies ψ back to the host exactly once (at the end)
    assert calls['to_host'] == 1


def test_device_absorber_and_callbacks():
    grid = Grid1D(256, -20.0, 20.0)
    wf0 = WaveFunction1D.gaussian(grid, 0.0, 4.0, 1.0)
    abl = AbsorbingBoundaryLayer(grid, 4.0)
    backend, calls = _strict_backend()
    seen = []
    ref = SplitOperatorSolver(grid, None, 0.01, absorber=abl).evolve(wf0, 40)
    out = SplitOperatorSolver(grid, None, 0.01, absorber=abl, backend=backend).evolve(
        wf0, 40, callback=lambda n, t, wf: seen.append(wf.norm()), callback_interval=10)
    np.testing.assert_allclose(out.psi, ref.psi, atol=1e-13)
    assert len(seen) == 4
    assert calls['to_host'] == 5      # four callbacks + final result


def test_device_step_round_trip():
    grid = Grid2D(32, -6.0, 6.0, 32, -6.0, 6.0)
    wf0 = WaveFunction2D.gaussian(grid, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0)
    backend, _ = _strict_backend()
    a = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.02, backend=backend, rotation=0.3)
    b = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.02, rotation=0.3)
    wa, wb = wf0, wf0
    for _ in range(5):
        wa, wb = a.step(wa), b.step(wb)
    assert isinstance(wa.psi, np.ndarray) and not isinstance(wa.psi, DeviceArray)
    np.testing.assert_allclose(wa.psi, wb.psi, atol=1e-13)


def test_ground_state_on_device_backend():
    grid = Grid2D(48, -8.0, 8.0, 48, -8.0, 8.0)
    backend, _ = _strict_backend()
    wf_d, info_d = ground_state(grid, HarmonicTrap(1.0), dtau=0.01, nonlinearity=10.0, backend=backend)
    wf_n, info_n = ground_state(grid, HarmonicTrap(1.0), dtau=0.01, nonlinearity=10.0)
    assert info_d.steps == info_n.steps
    assert info_d.energy == pytest.approx(info_n.energy, abs=1e-12)


def test_backend_selection():
    assert get_backend('numpy').name == 'scipy-cpu'
    with pytest.raises(ValueError):
        get_backend('opencl')
    if not cupy_available():
        assert get_backend('auto').name == 'scipy-cpu'
        with pytest.raises(RuntimeError):
            get_backend('cupy')


@pytest.mark.skipif(not cupy_available(), reason='CuPy with a CUDA device is not available')
@pytest.mark.parametrize('kwargs', [dict(order=4), dict(nonlinearity=50.0, rotation=0.7),
                                    dict(nonlinearity=50.0, imaginary_time=True)])
def test_real_cupy_matches_cpu(kwargs):
    grid = Grid3D(32, -6.0, 6.0, 32, -6.0, 6.0, 32, -6.0, 6.0)
    wf0 = WaveFunction3D.gaussian(grid, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0)
    cpu = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.01, **kwargs).evolve(wf0, 50)
    gpu_solver = SplitOperatorSolver(grid, HarmonicTrap(1.0), 0.01, backend='cupy', **kwargs)
    gpu = gpu_solver.evolve(wf0, 50)
    assert gpu_solver.backend == 'cupy'
    np.testing.assert_allclose(gpu.psi, cpu.psi, atol=1e-11)
