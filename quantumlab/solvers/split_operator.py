"""
Generic split-operator propagator for 1D, 2D and 3D grids.

One solver covers:

* 2nd-order Strang splitting and 4th-order Yoshida (triple-jump) composition,
* explicitly time-dependent potentials V(r, t),
* the cubic Gross–Pitaevskii nonlinearity g|ψ|²ψ (BEC mean field, solitons),
* imaginary-time propagation (ground states).

Time-dependent potentials are handled by treating t as an extra coordinate:
the Strang step S(h) applies exp(−i V(t) h/2ℏ), exp(−i T h/ℏ),
exp(−i V(t+h) h/2ℏ). This is a symmetric 2nd-order method, so the Yoshida
composition S(w₁h) S(w₀h) S(w₁h) is 4th-order accurate.

The nonlinear sub-flow i ℏ ∂ψ/∂t = g|ψ|²ψ leaves |ψ| unchanged and is
therefore integrated exactly by the phase factor exp(−i g|ψ|² h/2ℏ).
In imaginary time the state is kept at unit norm and the nonlinear term
uses the normalised density g|ψ|²/‖ψ‖² evaluated at the sub-step midpoint,
which keeps the converged GP ground state 2nd-order accurate in dτ.
"""
import numpy as np
from scipy.fft import fftn, ifftn
from quantumlab.solvers.base import Solver
from quantumlab.core.spectral import k_squared, volume_element, wavefunction_class

_CBRT2 = 2.0 ** (1.0 / 3.0)
YOSHIDA_W1 = 1.0 / (2.0 - _CBRT2)
YOSHIDA_W0 = -_CBRT2 / (2.0 - _CBRT2)


class SplitOperatorSolver(Solver):
    """
    Parameters
    ----------
    grid : Grid1D, Grid2D or Grid3D
    potential : Potential or None
        Static or time-dependent (``time_dependent = True``) potential.
        None means free space.
    dt : float
        Time step (imaginary-time step dτ when ``imaginary_time=True``).
    hbar, m : float
        Reduced Planck constant and particle mass.
    order : {2, 4}
        Splitting order: 2 = Strang, 4 = Yoshida triple jump.
    nonlinearity : float
        Gross–Pitaevskii coupling g in V_eff = V + g|ψ|². With ψ normalised
        to 1, g includes the particle number (g = 4πℏ²aN/m in 3D).
    imaginary_time : bool
        Propagate in imaginary time t → −iτ and renormalise after every step.
        Converges to the lowest state not orthogonal to the initial guess.
    absorber : AbsorbingBoundaryLayer, optional
        Mask applied after every step.
    workers : int
        Threads for scipy.fft (-1 = all cores).
    t0 : float
        Initial time (relevant for time-dependent potentials).
    """

    def __init__(self, grid, potential, dt: float, hbar: float = 1.0, m: float = 1.0,
                 order: int = 2, nonlinearity: float = 0.0, imaginary_time: bool = False,
                 absorber=None, workers: int = -1, t0: float = 0.0):
        super().__init__(grid, potential, dt, hbar, m)
        if order not in (2, 4):
            raise ValueError('order must be 2 or 4')
        self.time_dependent = bool(getattr(potential, 'time_dependent', False))
        if imaginary_time and order == 4:
            raise ValueError('order=4 uses a negative substep, which is unstable in imaginary time')
        if imaginary_time and self.time_dependent:
            raise ValueError('imaginary-time propagation requires a time-independent potential')

        self.order = order
        self.nonlinearity = float(nonlinearity)
        self.imaginary_time = imaginary_time
        self.absorber = absorber
        self.workers = workers
        self.t0 = t0
        self.steps_taken = 0

        self._wf_class = wavefunction_class(grid)
        self._dV = volume_element(grid)
        # exp(c · E · h / ℏ): c = −i in real time, −1 in imaginary time
        self._c = -1.0 if imaginary_time else -1j
        self._T_k = self.hbar ** 2 * k_squared(grid) / (2.0 * self.m)
        self._substeps = (1.0,) if order == 2 else (YOSHIDA_W1, YOSHIDA_W0, YOSHIDA_W1)

        self._V_static = None
        if not self.time_dependent:
            self._V_static = (np.zeros(grid.shape) if potential is None
                              else np.asarray(potential.evaluate(grid), dtype=float))
        self.update_operators()

    def update_operators(self):
        """(Re)build the cached kinetic and static-potential propagators."""
        self._U_T = {}
        self._U_V_half = {}
        for w in set(self._substeps):
            h = w * self.dt
            self._U_T[w] = np.exp(self._c * self._T_k * h / self.hbar)
            if self._V_static is not None:
                self._U_V_half[w] = np.exp(self._c * self._V_static * h / (2.0 * self.hbar))

    @property
    def t(self) -> float:
        """Current simulation time."""
        return self.t0 + self.steps_taken * self.dt

    @property
    def backend(self) -> str:
        return 'scipy-cpu'

    def potential_at(self, t: float) -> np.ndarray:
        if self._V_static is not None:
            return self._V_static
        return np.asarray(self.potential.evaluate_at(self.grid, t), dtype=float)

    def _potential_half_step(self, psi, t, w):
        h = w * self.dt
        if self.nonlinearity == 0.0 and self._V_static is not None:
            psi *= self._U_V_half[w]
            return psi
        V = self.potential_at(t)
        if self.nonlinearity == 0.0:
            psi *= np.exp(self._c * V * h / (2.0 * self.hbar))
        elif not self.imaginary_time:
            # |ψ| is invariant under this sub-flow, so freezing |ψ|² is exact
            psi *= np.exp(self._c * (V + self.nonlinearity * (psi.real ** 2 + psi.imag ** 2)) * h / (2.0 * self.hbar))
        else:
            # In imaginary time |ψ| changes within the sub-step. Using the
            # normalised density (the flow then commutes with renormalisation)
            # evaluated at the sub-step midpoint keeps the scheme 2nd order;
            # freezing |ψ|² would make the converged state only O(dτ) accurate.
            def V_eff(p):
                rho = p.real ** 2 + p.imag ** 2
                return V + self.nonlinearity * rho / (np.sum(rho) * self._dV)
            psi_mid = psi * np.exp(-V_eff(psi) * h / (4.0 * self.hbar))
            psi *= np.exp(-V_eff(psi_mid) * h / (2.0 * self.hbar))
        return psi

    def _strang(self, psi, t, w):
        psi = self._potential_half_step(psi, t, w)
        psi_k = fftn(psi, workers=self.workers, overwrite_x=True)
        psi_k *= self._U_T[w]
        psi = ifftn(psi_k, workers=self.workers, overwrite_x=True)
        return self._potential_half_step(psi, t + w * self.dt, w)

    def step(self, wavefunction):
        psi = np.array(wavefunction.psi, dtype=complex)
        t = self.t
        for w in self._substeps:
            psi = self._strang(psi, t, w)
            t += w * self.dt
        if self.imaginary_time:
            norm = np.sqrt(np.sum(psi.real ** 2 + psi.imag ** 2) * self._dV)
            if norm > 0:
                psi /= norm
        if self.absorber is not None:
            psi *= self.absorber.mask
        self.steps_taken += 1
        return self._wf_class(self.grid, psi)

    def evolve(self, wavefunction, n_steps: int, callback=None, callback_interval: int = 1):
        """
        Advance ``n_steps`` steps. If given, ``callback(step, t, wf)`` is called
        every ``callback_interval`` steps (step counted from 1 within this call).
        """
        wf = wavefunction
        for n in range(1, n_steps + 1):
            wf = self.step(wf)
            if callback is not None and n % callback_interval == 0:
                callback(n, self.t, wf)
        return wf
