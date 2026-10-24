"""
Ground states by preconditioned nonlinear conjugate gradient on the sphere
‖ψ‖² = N (Antoine, Levitt & Tang, J. Comput. Phys. 343, 92 (2017)).

Imaginary-time propagation is a first-order method whose fixed point carries
an O(dτ²) error, and soft modes (vortex lattices, elongated condensates)
relax in it as exp(−Δτ) with a tiny gap Δ. Minimising the energy directly
removes both problems: the fixed point is an exact stationary state, so the
residual ‖Hψ − μψ‖ can be driven to small values, and conjugate directions
converge on soft modes much faster than steepest descent.

The preconditioner is P = P_T^½ P_V P_T^½ with P_T = (α + ℏ²k²/2m)⁻¹ in
Fourier space and P_V = (α + V + g|ψ|²)⁻¹ in real space, which tames both the
kinetic and the potential stiffness.
"""
from typing import Optional
import numpy as np
from scipy.fft import fftn, ifftn
from quantumlab.core.spectral import k_squared, volume_element, wavefunction_class
from quantumlab.stationary.hamiltonian import apply_hamiltonian
from quantumlab.stationary.imaginary_time import ConvergenceInfo, _default_guess


def ground_state_pcg(grid, potential, initial=None, nonlinearity: float = 0.0, rotation: float = 0.0,
                     tol: float = 1e-8, max_iter: int = 10000, hbar: float = 1.0, m: float = 1.0,
                     callback=None, callback_interval: int = 100, workers: int = -1):
    """
    Minimise E[ψ] = ⟨ψ|−ℏ²∇²/2m + V − ΩL_z|ψ⟩ + (g/2)∫|ψ|⁴ at fixed norm.

    Parameters
    ----------
    initial : WaveFunction or ndarray, optional
        Starting state; its norm N is kept. For vortex states start from a
        guess with vortices (e.g. from damped dynamics or imaginary time).
    tol : float
        Stop when the residual ‖Hψ − μψ‖/√N falls below ``tol``.
    callback : callable, optional
        ``callback(iteration, residual, energy)`` every ``callback_interval``
        iterations.

    Returns
    -------
    wf : WaveFunction
    info : ConvergenceInfo
        ``steps`` is the number of iterations, ``converged`` whether ``tol``
        was reached, ``residual`` = ‖Hψ − μψ‖/√N.
    """
    if getattr(potential, 'time_dependent', False):
        raise ValueError('ground_state_pcg requires a time-independent potential')
    dV = volume_element(grid)
    V = np.zeros(grid.shape) if potential is None else np.asarray(potential.evaluate(grid), dtype=float)
    T_k = hbar ** 2 * k_squared(grid) / (2.0 * m)
    g = float(nonlinearity)
    psi = _default_guess(grid) if initial is None else np.array(getattr(initial, 'psi', initial), dtype=complex)

    def inner(a, b):
        return float(np.real(np.sum(np.conj(a) * b)) * dV)

    if initial is None:
        psi /= np.sqrt(inner(psi, psi))                        # default guess at unit norm

    def h_psi(p):
        return apply_hamiltonian(p, grid, V, hbar, m, nonlinearity=g, workers=workers, rotation=rotation)

    def quartic(p):
        return float(np.sum((p.real ** 2 + p.imag ** 2) ** 2) * dV)

    def energy_from(p, H_p):
        # ⟨ψ|Hψ⟩ with H including g|ψ|² counts the interaction twice: E = ⟨ψ|Hψ⟩ − (g/2)∫|ψ|⁴
        return inner(p, H_p) - 0.5 * g * quartic(p)

    # Each iteration applies H twice (trial point and accepted point); the accepted
    # point's Hψ is reused for the next residual and energy.
    N = inner(psi, psi)
    Hpsi = h_psi(psi)
    d_prev = r_prev = Pr_prev = None
    converged = False
    for it in range(max_iter + 1):
        mu = inner(psi, Hpsi) / N
        r = Hpsi - mu * psi
        residual = np.sqrt(inner(r, r) / N)
        E0 = energy_from(psi, Hpsi)
        if callback is not None and it % callback_interval == 0:
            callback(it, residual, E0)
        if residual < tol:
            converged = True
            break
        if it == max_iter:
            break
        # The shift keeps both preconditioner factors positive, also for V < 0
        alpha = max(1.0, mu - float(V.min()))
        PT_half = 1.0 / np.sqrt(alpha + T_k)
        PV = 1.0 / (alpha + V + g * np.abs(psi) ** 2)
        Pr = ifftn(PT_half * fftn(PV * ifftn(PT_half * fftn(r, workers=workers), workers=workers),
                                  workers=workers), workers=workers)
        Pr -= inner(psi, Pr) / N * psi                        # tangent space of the sphere
        beta = 0.0 if r_prev is None else max(0.0, inner(r - r_prev, Pr) / inner(r_prev, Pr_prev))
        d = -Pr if d_prev is None else -Pr + beta * d_prev
        d -= inner(psi, d) / N * psi
        if inner(r, d) >= 0.0:                                 # not a descent direction: restart
            d = -Pr
        u = d * np.sqrt(N / inner(d, d))                       # tangent direction with ‖u‖² = N
        # Line search along the great circle ψ(θ) = ψ cos θ + u sin θ (norm preserved).
        # While energy differences are resolvable, θ comes from a quadratic fit of E(θ);
        # near convergence they scale as residual² and drown in round-off, so a secant
        # step on the directional derivative E'(θ) = 2 Re⟨Hψ(θ), ψ'(θ)⟩ is used instead.
        slope0 = 2.0 * inner(Hpsi, u)
        trial = min(0.5, 0.5 * np.sqrt(inner(d, d) / N))
        psi_t = psi * np.cos(trial) + u * np.sin(trial)
        H_t = h_psi(psi_t)
        E_t = energy_from(psi_t, H_t)
        if abs(E_t - E0) > 1e-10 * max(1.0, abs(E0)):
            curvature = (E_t - E0 - slope0 * trial) / trial ** 2
            theta = -slope0 / (2.0 * curvature) if curvature > 0 else trial
        else:
            slope_t = 2.0 * inner(H_t, -psi * np.sin(trial) + u * np.cos(trial))
            theta = trial * slope0 / (slope0 - slope_t) if slope_t > slope0 else trial
        for _ in range(30):
            if theta == trial:
                candidate, H_c, E_c = psi_t, H_t, E_t
            else:
                candidate = psi * np.cos(theta) + u * np.sin(theta)
                H_c = h_psi(candidate)
                E_c = energy_from(candidate, H_c)
            # accept unless the energy rises by more than round-off
            if E_c <= E0 + 1e-12 * max(1.0, abs(E0)):
                break
            theta *= 0.5
        psi, Hpsi = candidate, H_c
        d_prev, r_prev, Pr_prev = d, r, Pr

    return wavefunction_class(grid)(grid, psi), ConvergenceInfo(converged, it, E0, mu, residual)
