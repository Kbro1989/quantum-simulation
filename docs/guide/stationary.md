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

### Rotating frames and vortex states

`ground_state(..., rotation=Ω)` works in the rotating frame, and energies and μ refer to H − ΩL_z. Above the critical rotation the ground state contains vortices. Imaginary time reaches it only from a guess that breaks rotational symmetry, for example an imprinted vortex or a lattice obtained from damped real-time dynamics, as in [example 11](../examples.md#11-vortex-lattice-in-a-rotating-bec).
