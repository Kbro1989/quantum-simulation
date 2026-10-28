# Two-component condensates

[`CoupledGPESolver`](../api/solvers.md) integrates coupled Gross–Pitaevskii equations for condensates of equal mass, for example two hyperfine states of ⁸⁷Rb:

> iℏ ∂ψ_i/∂t = [ −ℏ²∇²/2m + V_i(r) + Σ_j g_ij |ψ_j|² − ΩL_z ] ψ_i

```python
from quantumlab.solvers.multicomponent import CoupledGPESolver
from quantumlab.stationary import coupled_ground_state

G = [[100.0, 120.0],
     [120.0, 100.0]]                       # g11, g12 / g21, g22 (symmetric)
states, info = coupled_ground_state(grid, (trap, trap), G, (guess_1, guess_2), dtau=0.005)
print(info.chemical_potentials, info.residuals)

solver = CoupledGPESolver(grid, (trap, trap), G, dt=0.005, order=4, rotation=0.8)
psi_1, psi_2 = solver.evolve(states, 1000)
```

- The solver supports the same options as `SplitOperatorSolver`: order 2/4, imaginary time, rotation, damping and the GPU backend.
- Each component keeps its own particle number, which is the norm of its initial state.
- In real time the coupled potential step is exact. In imaginary time and with damping it uses rescaled midpoint densities, as for one component.

## Miscibility

A uniform mixture with g₁₁, g₂₂ > 0 is stable (**miscible**) when g₁₂² < g₁₁g₂₂ and separates into domains (**immiscible**) when g₁₂² > g₁₁g₂₂. `miscibility_overlap(wf_1, wf_2)` gives Λ = ∫n₁n₂ / √(∫n₁²∫n₂²): 1 for identical profiles, close to 0 for separated ones. In a trap the transition is slightly shifted above g₁₂ = g, because the domain wall between the components costs gradient energy.

!!! tip
    Immiscible ground states break the symmetry between the components. Start imaginary time from a guess that already breaks it, e.g. displaced clouds; a symmetric guess can stay on a symmetric saddle point.

## Dynamics and the Bogoliubov spin mode

For equal densities n and g₁₁ = g₂₂ = g, small differences between the components (the *spin* mode) obey

> ω_s(k)² = ε_k (ε_k + 2n(g − g₁₂)),   ε_k = ℏ²k²/2m

For g₁₂ < g these are stable oscillations. For g₁₂ > g the long wavelengths become unstable and grow as e^{γt}, fastest at k\* = √(2mn(g₁₂ − g))/ℏ with γ\* = n(g₁₂ − g). This **spinodal decomposition** is reproduced to 10⁻⁹ (frequency) and 10⁻⁴ (growth rate) in the validation suite.

`coupled_gp_energy` gives the conserved total energy Σ(⟨T⟩ + ⟨V_i⟩) + ½Σg_ij∫n_in_j.

## Numerical stability

For nonlinear real-time runs on a uniform background, keep the kinetic phase of the shortest grid wavelength below π per (sub)step, i.e. ℏk_max²/2m · |w|dt < π with k_max = π/dx (|w| ≤ 1.71 for order 4). Above it, the split-step method has a parametric instability in which round-off noise grows (Weideman & Herbst 1986).
