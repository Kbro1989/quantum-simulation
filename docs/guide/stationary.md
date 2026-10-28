# Stationary states

## Direct eigensolvers

[`eigenstates`](../api/stationary.md) returns the lowest eigenpairs of H = −ℏ²∇²/2m + V. The Hamiltonian is never stored as a matrix: it is applied with FFTs.

```python
from quantumlab.stationary import eigenstates
from quantumlab.potentials import HarmonicTrap

energies, states = eigenstates(grid, HarmonicTrap((1.0, 1.3, 1.7)), n_states=6)
```

- `method='lanczos'` uses ARPACK. It is robust and best for small grids.
- `method='lobpcg'` uses a block solver preconditioned by the inverse kinetic operator in k-space. It is much faster for large 2D/3D grids.
- `method='auto'` (the default) chooses Lanczos below 20 000 grid points.

Eigenvectors are returned normalised. Degenerate levels come back as an arbitrary orthonormal basis of their subspace.

## Imaginary-time propagation

```python
from quantumlab.stationary import ground_state, excited_states

wf, info = ground_state(grid, trap, dtau=0.005, nonlinearity=500.0)
print(info.converged, info.energy, info.chemical_potential, info.residual)
```

- `ConvergenceInfo.residual` is ‖Hψ − μψ‖, a direct measure of how close the result is to an eigenstate.
- For the nonlinear problem the step uses the normalised density at the sub-step midpoint. This keeps the converged Gross–Pitaevskii ground state **O(dτ²)** accurate: freezing |ψ|² would make it only O(dτ).
- A good initial guess helps. For BECs, a Thomas–Fermi profile is a good choice.
- `excited_states` finds the lowest states one by one with Gram–Schmidt projection (linear problems only).

### Continuation in dτ

`dtau` can also be a decreasing sequence, e.g. `dtau=[0.04, 0.01, 0.0025]`. The large steps relax slow modes in few iterations, and the small ones lower the O(dτ²) error floor. `max_steps` and `tol` apply per stage.

## Conjugate-gradient energy minimisation

[`ground_state_pcg`](../api/stationary.md) minimises the (Gross–Pitaevskii) energy directly on the sphere ‖ψ‖² = N with a preconditioned nonlinear conjugate-gradient method (Antoine, Levitt & Tang, *J. Comput. Phys.* **343**, 92 (2017)):

```python
from quantumlab.stationary import ground_state_pcg

wf, info = ground_state_pcg(grid, trap, initial=guess, nonlinearity=500.0, rotation=0.8, tol=1e-8)
print(info.converged, info.steps, info.residual)
```

Use it when imaginary time is too slow or not accurate enough:
- **Exact fixed point.** There is no dτ, so the residual ‖Hψ − μψ‖ can be driven to ~10⁻¹⁰. In the validation suite the 3D Gross–Pitaevskii virial theorem holds to 10⁻⁹, against 10⁻⁵–10⁻⁶ from imaginary time.
- **Soft modes.** Vortex lattices and elongated condensates have modes with tiny energy gaps. Imaginary time relaxes them as e^{−Δτ}, while conjugate directions converge on them much faster. In [example 11](../examples.md#11-vortex-lattice-in-a-rotating-bec) a 3D vortex lattice on a 128 × 128 × 64 grid reaches a residual below 10⁻⁶ in about 9300 iterations; imaginary time stalled near 3 × 10⁻³.

The preconditioner combines the kinetic operator (in Fourier space) and the potential plus mean field (in real space), so it works with both stiff kinetic and stiff potential terms, including attractive potentials such as `SoftCoulomb`. Near convergence the line search switches from energy values to directional derivatives, which stay resolvable below residuals of 10⁻⁷.

### Rotating frames and vortex states

`ground_state(..., rotation=Ω)` works in the rotating frame, and energies and μ refer to H − ΩL_z. Above the critical rotation the ground state contains vortices. Imaginary time (or the conjugate-gradient minimiser) reaches it only from a guess that breaks rotational symmetry, for example an imprinted vortex or a lattice obtained from damped real-time dynamics, as in [example 11](../examples.md#11-vortex-lattice-in-a-rotating-bec).
