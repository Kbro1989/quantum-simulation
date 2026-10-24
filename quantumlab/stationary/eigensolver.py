"""
Direct eigensolver for H = −ℏ²/2m ∇² + V(r) on 1D/2D/3D grids.

The Hamiltonian is never stored: it is applied matrix-free with FFTs and
handed to LOBPCG, preconditioned by the inverse kinetic operator in k-space.
For a real potential the spectral Hamiltonian is real-symmetric, so the
eigenvectors are real.
"""
import warnings
from typing import List, Tuple
import numpy as np
from scipy.fft import fftn, ifftn
from scipy.sparse.linalg import LinearOperator, lobpcg, eigsh
from quantumlab.core.spectral import k_squared, volume_element, wavefunction_class


def eigenstates(grid, potential, n_states: int = 6, hbar: float = 1.0, m: float = 1.0,
                tol: float = None, max_iter: int = 1000, seed: int = 0,
                method: str = 'auto', workers: int = -1) -> Tuple[np.ndarray, List]:
    """
    Lowest ``n_states`` eigenpairs of H.

    Parameters
    ----------
    method : {'auto', 'lobpcg', 'lanczos'}
        'lanczos' uses ARPACK (robust, best for small grids); 'lobpcg' uses a
        k-space preconditioned block solver (fast for large 2D/3D grids).
        'auto' picks Lanczos below 20 000 grid points.
    tol : float, optional
        Lanczos: relative eigenvalue accuracy (default 1e-12).
        LOBPCG: residual-norm tolerance (default 1e-6); the eigenvalue error
        scales as the residual squared.
    seed : int
        Seed of the deterministic random starting block (LOBPCG).

    Returns
    -------
    energies : ndarray, shape (n_states,)
        Ascending eigenvalues.
    states : list of WaveFunction
        Normalised eigenfunctions (∫|ψ|² dV = 1). Degenerate levels are
        returned as an arbitrary orthonormal basis of their subspace.
    """
    shape = grid.shape
    axes = tuple(range(len(shape)))
    size = int(np.prod(shape))
    V = np.asarray(potential.evaluate(grid), dtype=float)
    T_k = hbar ** 2 * k_squared(grid) / (2.0 * m)

    def _as_block(v):
        v = np.asarray(v)
        return v.reshape(shape + (-1,)), v.ndim == 1

    def matvec(v):
        block, squeeze = _as_block(v)
        k_block = fftn(block, axes=axes, workers=workers) * T_k[..., None]
        out = (ifftn(k_block, axes=axes, workers=workers).real + V[..., None] * block).reshape(size, -1)
        return out[:, 0] if squeeze else out

    H = LinearOperator((size, size), matvec=matvec, matmat=matvec, dtype=float)

    if method == 'auto':
        method = 'lanczos' if size < 20000 else 'lobpcg'

    if method == 'lanczos':
        energies, vecs = eigsh(H, k=n_states, which='SA', tol=1e-12 if tol is None else tol)
    elif method == 'lobpcg':
        shift = max(1.0, float(np.abs(V).mean()))
        inv_T = 1.0 / (T_k + shift)

        def precondition(v):
            block, squeeze = _as_block(v)
            k_block = fftn(block, axes=axes, workers=workers) * inv_T[..., None]
            out = ifftn(k_block, axes=axes, workers=workers).real.reshape(size, -1)
            return out[:, 0] if squeeze else out

        M = LinearOperator((size, size), matvec=precondition, matmat=precondition, dtype=float)
        rng = np.random.default_rng(seed)
        block_size = n_states + max(2, n_states // 2)
        X0 = rng.standard_normal((size, block_size))
        res_tol = 1e-6 if tol is None else tol
        with warnings.catch_warnings():
            # LOBPCG warns about the extra padding vectors too; the requested
            # states are checked explicitly below.
            warnings.simplefilter('ignore', UserWarning)
            energies, vecs = lobpcg(H, X0, M=M, tol=res_tol, maxiter=max_iter, largest=False)
        energies, vecs = energies[:n_states], vecs[:, :n_states]
        residuals = np.linalg.norm(H.matmat(vecs) - vecs * energies, axis=0) / np.linalg.norm(vecs, axis=0)
        if np.any(residuals > 10 * res_tol):
            warnings.warn(f'LOBPCG did not converge: max residual {residuals.max():.2e}', RuntimeWarning)
    else:
        raise ValueError("method must be 'auto', 'lobpcg' or 'lanczos'")

    order = np.argsort(energies)
    energies = energies[order]
    vecs = vecs[:, order]

    dV = volume_element(grid)
    wf_class = wavefunction_class(grid)
    states = []
    for j in range(n_states):
        psi = vecs[:, j].reshape(shape) / np.sqrt(dV)
        psi /= np.sqrt(np.sum(psi ** 2) * dV)
        # Fix the global sign so the largest lobe is positive (reproducible output)
        idx = np.unravel_index(np.argmax(np.abs(psi)), shape)
        if psi[idx] < 0:
            psi = -psi
        states.append(wf_class(grid, psi.astype(complex)))
    return energies, states
