"""
Imaginary-time propagation (ψ ← e^{−Hτ}ψ / ‖·‖) for ground and excited states,
including the nonlinear Gross–Pitaevskii ground state.

The converged state of 2nd-order Strang imaginary-time stepping differs from
the exact eigenstate by O(dτ²); the reported energy is evaluated with the
exact spectral Hamiltonian, so its error is O(dτ⁴) for the ground state.
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.stationary.hamiltonian import apply_hamiltonian
from quantumlab.core.spectral import volume_element, wavefunction_class, coordinates


@dataclass
class ConvergenceInfo:
    converged: bool
    steps: int
    energy: float
    chemical_potential: float
    residual: float


def _inner(a, b, dV):
    return np.sum(np.conj(a) * b) * dV


def _energy_and_mu(psi, grid, V, hbar, m, g, dV, rotation=0.0):
    H_lin = apply_hamiltonian(psi, grid, V, hbar, m, rotation=rotation)
    e_lin = float(np.real(_inner(psi, H_lin, dV)))
    e_int = 0.5 * g * float(np.sum(np.abs(psi) ** 4) * dV)
    H_psi = H_lin + g * np.abs(psi) ** 2 * psi
    mu = float(np.real(_inner(psi, H_psi, dV)))
    residual = float(np.sqrt(np.sum(np.abs(H_psi - mu * psi) ** 2) * dV))
    return e_lin + e_int, mu, residual


def _default_guess(grid):
    coords = coordinates(grid)
    r2 = sum(c ** 2 for c in coords)
    width2 = min(np.ptp(c) for c in coords) ** 2 / 64.0
    return np.exp(-r2 / (2.0 * width2)).astype(complex)


def ground_state(grid, potential, dtau: float = 0.01, nonlinearity: float = 0.0,
                 initial=None, tol: float = 1e-11, max_steps: int = 50000,
                 check_interval: int = 20, hbar: float = 1.0, m: float = 1.0,
                 orthogonal_to: Optional[List] = None, workers: int = -1,
                 rotation: float = 0.0, backend='numpy'):
    """
    Relax to the lowest-energy state by imaginary-time propagation.

    Parameters
    ----------
    dtau : float or sequence of float
        Imaginary time step. A decreasing sequence runs a continuation: the
        large steps relax slow modes (e.g. of vortex lattices) in few steps,
        the small ones lower the O(dτ²) error floor. ``max_steps`` and
        ``tol`` apply to each stage, and ``info`` describes the last one
        (``steps`` is the total).
    nonlinearity : float
        Gross–Pitaevskii coupling g (ψ normalised to 1). g = 0 gives the
        linear Schrödinger ground state.
    initial : WaveFunction or ndarray, optional
        Starting guess. A good guess (e.g. Thomas–Fermi) speeds convergence.
    tol : float
        Stop when the energy changes by less than ``tol`` between checks.
    orthogonal_to : list of WaveFunction, optional
        States projected out after each step (Gram–Schmidt), used to obtain
        excited states of the linear problem.
    rotation : float
        Angular velocity Ω of a frame rotating about z (2D/3D). Energies and
        μ are then those of H − ΩL_z. Above the critical Ω the GP ground state
        contains vortices; imaginary time only reaches it if the guess breaks
        rotational symmetry (e.g. an imprinted vortex or a lattice seed).
    backend : {'numpy', 'cupy', 'auto'}
        Where the propagation runs (see ``SplitOperatorSolver``). Energies are
        evaluated on the host every ``check_interval`` steps.

    Returns
    -------
    wf : WaveFunction
    info : ConvergenceInfo
        ``energy`` is the GP energy functional E[ψ] (equal to the eigenvalue
        for g = 0), ``chemical_potential`` is μ = ⟨ψ|H_GP|ψ⟩ and ``residual``
        is ‖H_GP ψ − μψ‖.
    """
    if np.ndim(dtau) > 0:
        wf, total = initial, 0
        for step_size in dtau:
            wf, info = ground_state(grid, potential, float(step_size), nonlinearity, wf, tol, max_steps,
                                    check_interval, hbar, m, orthogonal_to, workers, rotation, backend)
            total += info.steps
        info.steps = total
        return wf, info
    if orthogonal_to and nonlinearity != 0.0:
        raise ValueError('orthogonal_to is only meaningful for the linear problem')
    solver = SplitOperatorSolver(grid, potential, dtau, hbar, m, order=2,
                                 nonlinearity=nonlinearity, imaginary_time=True, workers=workers,
                                 rotation=rotation, backend=backend)
    dV = volume_element(grid)
    wf_class = wavefunction_class(grid)
    V = solver.potential_at(0.0)
    lower = [np.asarray(s.psi) for s in (orthogonal_to or [])]

    if initial is None:
        psi = _default_guess(grid)
    else:
        psi = np.array(getattr(initial, 'psi', initial), dtype=complex)

    def project(p):
        for q in lower:
            p = p - _inner(q, p, dV) * q
        return p / np.sqrt(np.real(_inner(p, p, dV)))

    psi = project(psi)
    wf = wf_class(grid, psi)
    energy_prev = np.inf
    converged = False
    steps = 0
    while steps < max_steps:
        if lower:
            for _ in range(check_interval):
                wf = solver.step(wf)
                wf.psi = project(wf.psi)
        else:
            # Keeps ψ on the device (GPU backend) for the whole interval
            wf = solver.evolve(wf, check_interval)
        steps += check_interval
        energy, _, _ = _energy_and_mu(wf.psi, grid, V, hbar, m, nonlinearity, dV, rotation)
        if abs(energy - energy_prev) < tol:
            converged = True
            break
        energy_prev = energy

    energy, mu, residual = _energy_and_mu(wf.psi, grid, V, hbar, m, nonlinearity, dV, rotation)
    return wf, ConvergenceInfo(converged, steps, energy, mu, residual)


def excited_states(grid, potential, n_states: int, dtau: float = 0.01, tol: float = 1e-11,
                   max_steps: int = 50000, seed: int = 0, hbar: float = 1.0, m: float = 1.0,
                   workers: int = -1):
    """
    Lowest ``n_states`` eigenstates of the linear Hamiltonian, found one by one
    with imaginary time + Gram–Schmidt against the states already converged.
    Starting guesses are seeded random fields (deterministic for a fixed seed).

    Returns
    -------
    energies : ndarray
    states : list of WaveFunction
    infos : list of ConvergenceInfo
    """
    rng = np.random.default_rng(seed)
    envelope = np.abs(_default_guess(grid))
    states, energies, infos = [], [], []
    for _ in range(n_states):
        guess = envelope * rng.standard_normal(grid.shape)
        wf, info = ground_state(grid, potential, dtau=dtau, initial=guess, tol=tol,
                                max_steps=max_steps, hbar=hbar, m=m,
                                orthogonal_to=states, workers=workers)
        states.append(wf)
        energies.append(info.energy)
        infos.append(info)
    return np.array(energies), states, infos


@dataclass
class CoupledConvergenceInfo:
    converged: bool
    steps: int
    energy: float
    chemical_potentials: List[float]
    residuals: List[float]


def _coupled_energy_mu(psis, grid, Vs, g, hbar, m, rotation, dV):
    n = [np.abs(p) ** 2 for p in psis]
    energy, mus, residuals = 0.0, [], []
    for i, p in enumerate(psis):
        N = float(np.sum(n[i]) * dV)
        H_lin = apply_hamiltonian(p, grid, Vs[i], hbar, m, rotation=rotation)
        energy += float(np.real(_inner(p, H_lin, dV)))
        H_psi = H_lin + sum(g[i, j] * n[j] for j in range(len(psis))) * p
        mu = float(np.real(_inner(p, H_psi, dV))) / N
        mus.append(mu)
        residuals.append(float(np.sqrt(np.sum(np.abs(H_psi - mu * p) ** 2) * dV / N)))
    energy += 0.5 * sum(g[i, j] * float(np.sum(n[i] * n[j]) * dV)
                        for i in range(len(psis)) for j in range(len(psis)))
    return energy, mus, residuals


def coupled_ground_state(grid, potentials, couplings, initial, dtau: float = 0.005, tol: float = 1e-11,
                         max_steps: int = 50000, check_interval: int = 20, hbar: float = 1.0,
                         m: float = 1.0, rotation: float = 0.0, backend='numpy', workers: int = -1):
    """
    Lowest-energy state of coupled Gross–Pitaevskii equations by imaginary
    time, at fixed populations N_i given by the norms of ``initial``.

    Parameters
    ----------
    initial : sequence of WaveFunction or ndarray
        One starting state per component. For immiscible mixtures the guess
        should break the symmetry between the components (e.g. displaced
        clouds); a symmetric guess can stay on a symmetric saddle point.

    Returns
    -------
    states : tuple of WaveFunction
    info : CoupledConvergenceInfo
        ``energy`` is the total energy functional (of H − ΩL_z when rotating),
        ``chemical_potentials`` and ``residuals`` (‖H_iψ_i − μ_iψ_i‖/√N_i) are
        per component.
    """
    from quantumlab.solvers.multicomponent import CoupledGPESolver
    g = np.asarray(couplings, dtype=float)
    solver = CoupledGPESolver(grid, potentials, g, dtau, hbar, m, order=2, imaginary_time=True,
                              rotation=rotation, backend=backend, workers=workers)
    dV = volume_element(grid)
    wf_class = wavefunction_class(grid)
    Vs = [np.zeros(grid.shape) if p is None else np.asarray(p.evaluate(grid), dtype=float) for p in potentials]
    states = tuple(wf_class(grid, np.array(getattr(s, 'psi', s), dtype=complex)) for s in initial)
    energy_prev = np.inf
    converged = False
    steps = 0
    while steps < max_steps:
        states = solver.evolve(states, check_interval)
        steps += check_interval
        energy, _, _ = _coupled_energy_mu([s.psi for s in states], grid, Vs, g, hbar, m, rotation, dV)
        if abs(energy - energy_prev) < tol:
            converged = True
            break
        energy_prev = energy
    energy, mus, residuals = _coupled_energy_mu([s.psi for s in states], grid, Vs, g, hbar, m, rotation, dV)
    return states, CoupledConvergenceInfo(converged, steps, energy, mus, residuals)
