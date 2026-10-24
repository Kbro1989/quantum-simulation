"""
Generic split-operator propagator for 1D, 2D and 3D grids.

One solver covers:

* 2nd-order Strang splitting and 4th-order Yoshida (triple-jump) composition,
* explicitly time-dependent potentials V(r, t),
* the cubic Gross–Pitaevskii nonlinearity g|ψ|²ψ (BEC mean field, solitons),
* imaginary-time propagation (ground states),
* a rotating frame, H → H − ΩL_z (2D/3D, rotation about the z axis),
* phenomenological damping, (i − γ)ℏ ∂ψ/∂t = Hψ, with the norm held fixed,
* CPU (NumPy/SciPy) or CUDA GPU (CuPy) execution.

Time-dependent potentials are handled by treating t as an extra coordinate:
the Strang step S(h) applies exp(−i V(t) h/2ℏ), exp(−i T h/ℏ),
exp(−i V(t+h) h/2ℏ). This is a symmetric 2nd-order method, so the Yoshida
composition S(w₁h) S(w₀h) S(w₁h) is 4th-order accurate.

The nonlinear sub-flow i ℏ ∂ψ/∂t = g|ψ|²ψ leaves |ψ| unchanged and is
therefore integrated exactly by the phase factor exp(−i g|ψ|² h/2ℏ).
In imaginary time the state is kept at unit norm and the nonlinear term
uses the normalised density g|ψ|²/‖ψ‖² evaluated at the sub-step midpoint,
which keeps the converged GP ground state 2nd-order accurate in dτ.

With rotation, −ΩL_z = Ω(y p_x − x p_y) is split together with the kinetic
energy into H_x = p_x²/2m + Ω y p_x, diagonal in (k_x, y), and
H_yz = (p_y² + p_z²)/2m − Ω x p_y, diagonal in (x, k_y, k_z). Each factor is
applied exactly with 1D/2D FFTs, and the symmetric product
e^{H_x h/2} e^{H_yz h} e^{H_x h/2} keeps the step 2nd order (and symmetric,
so the Yoshida composition stays 4th order).
"""
import numpy as np
from quantumlab.solvers.base import Solver
from quantumlab.solvers.backends import get_backend
from quantumlab.core.spectral import k_squared, ndim, volume_element, wavefunction_class

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
    rotation : float
        Angular velocity Ω of the frame about the z axis (2D/3D grids). The
        frame rotates counter-clockwise for Ω > 0, so states with positive
        L_z are favoured. The potential is given in the rotating frame.
    damping : float
        Dissipation γ ≥ 0 in (i − γ)ℏ ∂ψ/∂t = Hψ (Choi et al. 1998, Tsubota
        et al. 2002). The norm is restored after every step, which plays the
        role of the chemical potential. Order 2 only.
    absorber : AbsorbingBoundaryLayer, optional
        Mask applied after every step.
    workers : int
        Threads for scipy.fft (-1 = all cores); ignored on the GPU.
    t0 : float
        Initial time (relevant for time-dependent potentials).
    backend : {'numpy', 'cupy', 'auto'} or ArrayBackend
        'cupy' runs on a CUDA GPU in double precision; 'auto' uses the GPU
        when CuPy and a CUDA device are available. ``evolve`` keeps ψ on the
        device between steps and only copies it back for callbacks and at the
        end; ``step`` copies to and from the device on every call.
        Time-dependent potentials are evaluated on the host and copied to the
        device at every sub-step.
    device_id : int
        CUDA device for the 'cupy' backend.

    Notes
    -----
    Every factor is unitary, but for nonlinear problems on a uniform (or
    broad) background the split-step method has a parametric numerical
    instability when the kinetic phase of the highest grid wavenumber per
    (sub)step exceeds π (Weideman & Herbst, SIAM J. Numer. Anal. 23, 485
    (1986)). Keep ``ℏ k_max² / (2m) · |w| dt < π`` with k_max = π/dx, where
    |w| ≤ 1.71 for order 4, otherwise round-off noise in the shortest
    wavelengths can grow exponentially. Localised states (solitons, trapped
    condensates) are much less sensitive.
    """

    def __init__(self, grid, potential, dt: float, hbar: float = 1.0, m: float = 1.0,
                 order: int = 2, nonlinearity: float = 0.0, imaginary_time: bool = False,
                 absorber=None, workers: int = -1, t0: float = 0.0,
                 rotation: float = 0.0, damping: float = 0.0,
                 backend='numpy', device_id: int = 0):
        super().__init__(grid, potential, dt, hbar, m)
        if order not in (2, 4):
            raise ValueError('order must be 2 or 4')
        self.time_dependent = bool(getattr(potential, 'time_dependent', False))
        if imaginary_time and order == 4:
            raise ValueError('order=4 uses a negative substep, which is unstable in imaginary time')
        if imaginary_time and self.time_dependent:
            raise ValueError('imaginary-time propagation requires a time-independent potential')
        if damping < 0:
            raise ValueError('damping must be non-negative')
        if damping > 0 and (order == 4 or imaginary_time):
            raise ValueError('damping requires order=2 real-time propagation')
        if rotation != 0.0 and ndim(grid) < 2:
            raise ValueError('rotation requires a 2D or 3D grid')

        self.order = order
        self.nonlinearity = float(nonlinearity)
        self.imaginary_time = imaginary_time
        self.rotation = float(rotation)
        self.damping = float(damping)
        # Steps that are not unitary are followed by renormalisation
        self._renormalise = imaginary_time or self.damping > 0
        self.absorber = absorber
        self.workers = workers
        self.t0 = t0
        self.steps_taken = 0
        self._be = get_backend(backend, workers=workers, device_id=device_id)
        self._xp = self._be.xp

        self._wf_class = wavefunction_class(grid)
        self._dV = volume_element(grid)
        # exp(c · E · h / ℏ): c = −i in real time, −1 in imaginary time and
        # −(i + γ)/(1 + γ²) with damping, from ∂ψ/∂t = Hψ / ((i − γ)ℏ)
        self._c = -1.0 if imaginary_time else -(1j + self.damping) / (1.0 + self.damping ** 2)
        self._T_k = self.hbar ** 2 * k_squared(grid) / (2.0 * self.m)
        self._substeps = (1.0,) if order == 2 else (YOSHIDA_W1, YOSHIDA_W0, YOSHIDA_W1)

        self._V_static = None
        if not self.time_dependent:
            self._V_static = (np.zeros(grid.shape) if potential is None
                              else np.asarray(potential.evaluate(grid), dtype=float))
        self._V_static_dev = None if self._V_static is None else self._be.to_device(self._V_static)
        self._mask_dev = None if absorber is None else self._be.to_device(np.asarray(absorber.mask))
        self.update_operators()

    def update_operators(self):
        """(Re)build the cached kinetic and static-potential propagators."""
        dev = self._be.to_device
        self._U_T = {}
        self._U_x_half = {}
        self._U_yz = {}
        self._U_V_half = {}
        for w in set(self._substeps):
            h = w * self.dt
            if self.rotation == 0.0:
                self._U_T[w] = dev(np.exp(self._c * self._T_k * h / self.hbar))
            else:
                H_x, H_yz = self._rotating_kinetic_parts()
                self._U_x_half[w] = dev(np.exp(self._c * H_x * h / (2.0 * self.hbar)))
                self._U_yz[w] = dev(np.exp(self._c * H_yz * h / self.hbar))
            if self._V_static is not None:
                self._U_V_half[w] = dev(np.exp(self._c * self._V_static * h / (2.0 * self.hbar)))

    def _rotating_kinetic_parts(self):
        """H_x(k_x, y) and H_yz(x, k_y, k_z) as arrays broadcastable to the grid."""
        g, hb, m, Om = self.grid, self.hbar, self.m, self.rotation
        kx, y = g.k_x[:, None], g.y[None, :]
        x, ky = g.x[:, None], g.k_y[None, :]
        H_x = hb ** 2 * kx ** 2 / (2.0 * m) + Om * hb * y * kx
        H_yz = hb ** 2 * ky ** 2 / (2.0 * m) - Om * hb * x * ky
        if ndim(g) == 3:
            H_x = H_x[:, :, None]
            H_yz = H_yz[:, :, None] + hb ** 2 * g.k_z[None, None, :] ** 2 / (2.0 * m)
        return H_x, H_yz

    @property
    def t(self) -> float:
        """Current simulation time."""
        return self.t0 + self.steps_taken * self.dt

    @property
    def backend(self) -> str:
        return self._be.name

    def potential_at(self, t: float) -> np.ndarray:
        """V(r, t) as a NumPy array on the host."""
        if self._V_static is not None:
            return self._V_static
        return np.asarray(self.potential.evaluate_at(self.grid, t), dtype=float)

    def _potential_on_device(self, t):
        if self._V_static_dev is not None:
            return self._V_static_dev
        return self._be.to_device(self.potential_at(t))

    def _potential_half_step(self, psi, t, w):
        xp = self._xp
        h = w * self.dt
        if self.nonlinearity == 0.0 and self._V_static is not None:
            psi *= self._U_V_half[w]
            return psi
        V = self._potential_on_device(t)
        if self.nonlinearity == 0.0:
            psi *= xp.exp(self._c * V * h / (2.0 * self.hbar))
        elif not self._renormalise:
            # |ψ| is invariant under this sub-flow, so freezing |ψ|² is exact
            psi *= xp.exp(self._c * (V + self.nonlinearity * (psi.real ** 2 + psi.imag ** 2)) * h / (2.0 * self.hbar))
        else:
            # In imaginary time (or with damping) |ψ| changes within the
            # sub-step. Using the density rescaled to the target norm (the flow
            # then commutes with renormalisation) evaluated at the sub-step
            # midpoint keeps the scheme 2nd order; freezing |ψ|² would make the
            # converged state only O(dτ) accurate.
            target = self._target_norm

            def V_eff(p):
                rho = p.real ** 2 + p.imag ** 2
                return V + self.nonlinearity * rho * (target / (float(xp.sum(rho)) * self._dV))
            psi_mid = psi * xp.exp(self._c * V_eff(psi) * h / (4.0 * self.hbar))
            psi *= xp.exp(self._c * V_eff(psi_mid) * h / (2.0 * self.hbar))
        return psi

    def _kinetic_step(self, psi, w):
        fftn, ifftn = self._be.fftn, self._be.ifftn
        if self.rotation == 0.0:
            psi_k = fftn(psi)
            psi_k *= self._U_T[w]
            return ifftn(psi_k)
        yz_axes = (1,) if psi.ndim == 2 else (1, 2)
        for U, axes in ((self._U_x_half[w], (0,)), (self._U_yz[w], yz_axes), (self._U_x_half[w], (0,))):
            psi_k = fftn(psi, axes=axes)
            psi_k *= U
            psi = ifftn(psi_k, axes=axes)
        return psi

    def _strang(self, psi, t, w):
        psi = self._potential_half_step(psi, t, w)
        psi = self._kinetic_step(psi, w)
        return self._potential_half_step(psi, t + w * self.dt, w)

    def _norm(self, psi) -> float:
        return float(self._xp.sum(psi.real ** 2 + psi.imag ** 2)) * self._dV

    def _advance(self, psi):
        """One full step on a backend array (modified in place where possible)."""
        if self._renormalise:
            # Imaginary time works at unit norm; damped dynamics keeps the input norm
            self._target_norm = 1.0 if self.imaginary_time else self._norm(psi)
        t = self.t
        for w in self._substeps:
            psi = self._strang(psi, t, w)
            t += w * self.dt
        if self._renormalise:
            norm = self._norm(psi)
            if norm > 0:
                psi *= np.sqrt(self._target_norm / norm)
        if self._mask_dev is not None:
            psi *= self._mask_dev
        self.steps_taken += 1
        return psi

    def _upload(self, wavefunction):
        return self._be.to_device(np.array(wavefunction.psi, dtype=complex))

    def _download(self, psi):
        return self._wf_class(self.grid, self._be.to_host(psi))

    def step(self, wavefunction):
        return self._download(self._advance(self._upload(wavefunction)))

    def evolve(self, wavefunction, n_steps: int, callback=None, callback_interval: int = 1):
        """
        Advance ``n_steps`` steps. If given, ``callback(step, t, wf)`` is called
        every ``callback_interval`` steps (step counted from 1 within this call).
        On the GPU, ψ stays on the device except for those callbacks.
        """
        psi = self._upload(wavefunction)
        for n in range(1, n_steps + 1):
            psi = self._advance(psi)
            if callback is not None and n % callback_interval == 0:
                callback(n, self.t, self._download(psi))
        return self._download(psi)
