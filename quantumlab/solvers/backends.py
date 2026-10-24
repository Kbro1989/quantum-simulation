"""
Array backends for the split-operator solver: NumPy/SciPy on the CPU or
CuPy on a CUDA GPU. A backend bundles the array module, the FFTs and the
host/device transfer functions, so the solver code is identical for both.
"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable
import numpy as np


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any                     # NumPy-compatible array module
    fftn: Callable              # fftn(a, axes=None) -> array on the same device
    ifftn: Callable
    to_device: Callable         # numpy.ndarray -> backend array
    to_host: Callable           # backend array -> numpy.ndarray


def cupy_available() -> bool:
    """True if CuPy is importable and at least one CUDA device is visible."""
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def numpy_backend(workers: int = -1) -> ArrayBackend:
    from scipy import fft as sp_fft
    return ArrayBackend(
        name='scipy-cpu',
        xp=np,
        fftn=partial(sp_fft.fftn, workers=workers, overwrite_x=True),
        ifftn=partial(sp_fft.ifftn, workers=workers, overwrite_x=True),
        to_device=np.asarray,
        to_host=np.asarray,
    )


def cupy_backend(device_id: int = 0) -> ArrayBackend:
    import cupy
    import cupyx.scipy.fft as cp_fft
    n_devices = cupy.cuda.runtime.getDeviceCount()
    if not 0 <= device_id < n_devices:
        raise ValueError(f'Invalid CUDA device_id {device_id}; {n_devices} device(s) available')
    cupy.cuda.Device(device_id).use()
    return ArrayBackend(
        name='cupy',
        xp=cupy,
        fftn=partial(cp_fft.fftn, overwrite_x=True),
        ifftn=partial(cp_fft.ifftn, overwrite_x=True),
        to_device=cupy.asarray,
        to_host=cupy.asnumpy,
    )


def get_backend(backend='numpy', workers: int = -1, device_id: int = 0) -> ArrayBackend:
    """
    Resolve a backend specification.

    ``'numpy'`` (default) — SciPy FFTs on the CPU;
    ``'cupy'`` — CUDA GPU, raises ImportError/RuntimeError if unavailable;
    ``'auto'`` — CuPy when a CUDA device is present, otherwise NumPy;
    an ``ArrayBackend`` instance is returned unchanged.
    """
    if isinstance(backend, ArrayBackend):
        return backend
    if backend == 'numpy':
        return numpy_backend(workers)
    if backend == 'cupy':
        if not cupy_available():
            raise RuntimeError("backend='cupy' requested but CuPy or a CUDA device is not available")
        return cupy_backend(device_id)
    if backend == 'auto':
        return cupy_backend(device_id) if cupy_available() else numpy_backend(workers)
    raise ValueError("backend must be 'numpy', 'cupy', 'auto' or an ArrayBackend")
