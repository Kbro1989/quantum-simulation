# Time evolution

## The split-operator engine

[`SplitOperatorSolver`](../api/solvers.md) integrates

> iℏ ∂ψ/∂t = [ −ℏ²∇²/2m + V(r, t) + g|ψ|² − ΩL_z ] ψ

on periodic 1D, 2D or 3D grids. Each step alternates exact potential and kinetic factors, with the kinetic factor applied in Fourier space:

```python
from quantumlab.solvers.split_operator import SplitOperatorSolver

solver = SplitOperatorSolver(grid, potential, dt=0.01, order=4)
wf = solver.evolve(wf, n_steps=500)
print(solver.t)          # current time
```

| Option | Meaning |
|---|---|
| `order=2` / `order=4` | Strang splitting or the 4th-order Yoshida triple jump. The measured global error scales as dt² and dt⁴. |
| `nonlinearity=g` | Gross–Pitaevskii term g\|ψ\|²ψ (BEC mean field, solitons). With ψ normalised to 1, g includes the particle number. |
| `imaginary_time=True` | Propagation in imaginary time with renormalisation every step (see [Stationary states](stationary.md)). |
| `rotation=Ω` | Rotating frame about the z axis in 2D/3D. The potential is given in the rotating frame. |
| `damping=γ` | Dissipative dynamics (i − γ)ℏ∂ψ/∂t = Hψ at fixed norm, e.g. for vortex-lattice formation. |
| `absorber=` | An `AbsorbingBoundaryLayer` mask applied after every step. |
| `backend=` | `'numpy'` (default), `'cupy'` or `'auto'` (see [GPU acceleration](gpu.md)). |

### Callbacks

`evolve` can call a function every few steps, for example to record observables or render frames:

```python
energies = []
solver.evolve(wf, 1000, callback=lambda n, t, w: energies.append(total_energy_expectation(w, potential)),
              callback_interval=10)
```

### Time-dependent potentials

Potentials with `time_dependent = True` are evaluated at every sub-step. The splitting stays symmetric, so order 4 still holds:

```python
from quantumlab.potentials import DipoleDrive, HarmonicTrap, sin2_envelope

laser = DipoleDrive(HarmonicTrap(1.0), F0=0.5, omega=0.6, envelope=sin2_envelope(40.0))
solver = SplitOperatorSolver(grid, laser, dt=0.01, order=4)
```

### Rotating condensates

```python
solver = SplitOperatorSolver(grid, trap, dt=0.005, nonlinearity=500.0, rotation=0.8, damping=0.05)
```

The rotation term −ΩL_z = Ω(y p_x − x p_y) is split with the kinetic energy into two parts, each diagonal after an FFT along one axis. Both are applied exactly, so the scheme keeps its order. In the validation suite a coherent state in a rotating frame follows the exact trajectory R(−Ωt)·(x₀ cos ωt, 0) to 2 × 10⁻⁸.

## Legacy solvers

`SplitStep1DSolver`, `SplitStep2DSolver`, `SplitStep3DSolver` and `SplitStep3DGPU` from the first release are kept for compatibility. For a static potential, `SplitOperatorSolver(order=2)` gives bit-identical results to them.

## Numerical stability of nonlinear runs

Every factor of the split-step method is unitary. Still, for nonlinear problems on a uniform or broad background it has a parametric instability when the kinetic phase of the shortest grid wavelength per (sub)step exceeds π (Weideman & Herbst 1986). Keep ℏk_max²/2m · |w|dt < π with k_max = π/dx, where |w| ≤ 1.71 for order 4. Localised states such as solitons and trapped condensates are much less sensitive.

Two-component condensates are covered in [Two-component condensates](multicomponent.md).

