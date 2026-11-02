"""
Coupled Gross–Pitaevskii equations for multi-component condensates
(e.g. two hyperfine states of ⁸⁷Rb, or two atomic species of equal mass):

    iℏ ∂ψ_i/∂t = [ −ℏ²∇²/2m + V_i(r) + Σ_j g_ij |ψ_j|² − ΩL_z ] ψ_i

The kinetic (and rotation) factor is the one of ``SplitOperatorSolver`` and is
applied to every component. The potential sub-flow is local and, in real
time, leaves every |ψ_j| unchanged, so it is integrated exactly with the
densities frozen. In imaginary time or with damping each component keeps its
own norm N_i (the populations are fixed), and the coupled potential uses the
densities rescaled to those norms at the sub-step midpoint, as for a single
component, which keeps the converged states 2nd-order accurate in dτ.
"""
from typing import Sequence
import numpy as np
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.core.spectral import volume_element, wavefunction_class


class CoupledGPESolver:
    """
    Parameters
    ----------
    grid : Grid1D, Grid2D or Grid3D
    potentials : sequence of Potential or None
        One static potential per component (None means free space).
    couplings : (n, n) array_like
        Symmetric interaction matrix g_ij. The nonlinear term uses the actual
        densities |ψ_j|², so g_ij is per particle when ψ_j is normalised to
        its particle number.
    dt : float
        Time step (dτ in imaginary time).
    order, imaginary_time, rotation, damping, workers, backend, device_id, t0
        As in ``SplitOperatorSolver``; they apply to all components.
    """

    def __init__(self, grid, potentials: Sequence, couplings, dt: float, hbar: float = 1.0,
                 m: float = 1.0, order: int = 2, imaginary_time: bool = False, rotation: float = 0.0,
                 damping: float = 0.0, workers: int = -1, backend='numpy', device_id: int = 0,
                 t0: float = 0.0):
        self.n_components = len(potentials)
        g = np.asarray(couplings, dtype=float)
        if g.shape != (self.n_components, self.n_components):
            raise ValueError(f'couplings must be a {self.n_components}×{self.n_components} matrix')
        if not np.allclose(g, g.T):
            raise ValueError('couplings must be symmetric (g_ij = g_ji)')
        if any(getattr(p, 'time_dependent', False) for p in potentials):
            raise ValueError('CoupledGPESolver supports static potentials only')
        self.grid = grid
        self.potentials = list(potentials)
        self.couplings = g
        self.dt = dt
        self.hbar = hbar
        self.m = m
        self.t0 = t0
        self.steps_taken = 0
        # The single-component engine supplies the kinetic/rotation propagators,
        # the time-stepping constant c, the substep weights and the array backend.
        self._engine = SplitOperatorSolver(grid, None, dt, hbar, m, order=order, imaginary_time=imaginary_time,
                                           rotation=rotation, damping=damping, workers=workers,
                                           backend=backend, device_id=device_id, t0=t0)
        self._be = self._engine._be
        self._xp = self._be.xp
        self._c = self._engine._c
        self._renormalise = self._engine._renormalise
        self._substeps = self._engine._substeps
        self._dV = volume_element(grid)
        self._wf_class = wavefunction_class(grid)
        self._V = [self._be.to_device(np.zeros(grid.shape) if p is None else
                                      np.asarray(p.evaluate(grid), dtype=float)) for p in self.potentials]

    @property
    def t(self) -> float:
        return self.t0 + self.steps_taken * self.dt

    @property
    def backend(self) -> str:
        return self._be.name

    def _densities(self, psis, rescale: bool):
        xp = self._xp
        rhos = [p.real ** 2 + p.imag ** 2 for p in psis]
        if rescale:
            rhos = [r * (n / (float(xp.sum(r)) * self._dV)) for r, n in zip(rhos, self._targets)]
        return rhos

    def _effective_potentials(self, rhos):
        g = self.couplings
        return [self._V[i] + sum(g[i, j] * rhos[j] for j in range(self.n_components) if g[i, j] != 0.0)
                for i in range(self.n_components)]

    def _potential_half_step(self, psis, w):
        xp = self._xp
        h = w * self.dt
        if not self._renormalise:
            V_eff = self._effective_potentials(self._densities(psis, rescale=False))
            return [p * xp.exp(self._c * V * h / (2.0 * self.hbar)) for p, V in zip(psis, V_eff)]
        V_eff = self._effective_potentials(self._densities(psis, rescale=True))
        mids = [p * xp.exp(self._c * V * h / (4.0 * self.hbar)) for p, V in zip(psis, V_eff)]
        V_mid = self._effective_potentials(self._densities(mids, rescale=True))
        return [p * xp.exp(self._c * V * h / (2.0 * self.hbar)) for p, V in zip(psis, V_mid)]

    def _norm(self, psi) -> float:
        return float(self._xp.sum(psi.real ** 2 + psi.imag ** 2)) * self._dV

    def _advance(self, psis):
        if self._renormalise:
            self._targets = [self._norm(p) for p in psis]
        for w in self._substeps:
            psis = self._potential_half_step(psis, w)
            psis = [self._engine._kinetic_step(p, w) for p in psis]
            psis = self._potential_half_step(psis, w)
        if self._renormalise:
            out = []
            for p, target in zip(psis, self._targets):
                norm = self._norm(p)
                out.append(p * np.sqrt(target / norm) if norm > 0 else p)
            psis = out
        self.steps_taken += 1
        return psis

    def _upload(self, states):
        if len(states) != self.n_components:
            raise ValueError(f'expected {self.n_components} wavefunctions')
        return [self._be.to_device(np.array(s.psi, dtype=complex)) for s in states]

    def _download(self, psis):
        return tuple(self._wf_class(self.grid, self._be.to_host(p)) for p in psis)

    def step(self, states):
        return self._download(self._advance(self._upload(states)))

    def evolve(self, states, n_steps: int, callback=None, callback_interval: int = 1):
        """
        Advance all components ``n_steps`` steps; ``callback(step, t, states)``
        is called every ``callback_interval`` steps with a tuple of wavefunctions.
        """
        psis = self._upload(states)
        for n in range(1, n_steps + 1):
            psis = self._advance(psis)
            if callback is not None and n % callback_interval == 0:
                callback(n, self.t, self._download(psis))
        return self._download(psis)
