# Observables

All observables take a wavefunction (1D, 2D or 3D unless noted) and return plain floats or arrays.

## Expectation values

| Function | Quantity |
|---|---|
| `position_expectation`, `y_expectation`, `z_expectation` | ⟨x⟩, ⟨y⟩, ⟨z⟩ |
| `position_uncertainty`, `y_uncertainty`, `z_uncertainty` | Δx, Δy, Δz |
| `momentum_expectation(wf, direction)` | ⟨p_x⟩, ⟨p_y⟩, ⟨p_z⟩ |
| `momentum_uncertainty(wf, direction)` | Δp along an axis |
| `kinetic_energy_expectation`, `potential_energy_expectation`, `total_energy_expectation` | ⟨T⟩, ⟨V⟩, ⟨H⟩ |
| `transmission_coefficient`, `reflection_coefficient` | probability beyond / before a position along x |

Momentum-space quantities use the Fourier transform of ψ, so they are exact for band-limited states. For a Gaussian packet Δx·Δp = ℏ/2 holds to 10⁻¹².

## Currents, angular momentum and vortices

| Function | Quantity |
|---|---|
| `probability_current(wf)` | j = (ℏ/m) Im(ψ*∇ψ) per axis |
| `gradient(psi, grid)` | spectral ∇ψ |
| `angular_momentum_z_expectation(wf)` | ⟨L_z⟩ (2D/3D) |
| `vortex_positions(wf)` | positions and charges of quantised vortices in a 2D state, from the phase winding around every grid cell |
| `autocorrelation(psi0, psi_t, grid)` | C(t) = ⟨ψ(0)\|ψ(t)⟩. Its Fourier transform peaks at the eigenenergies (Feit–Fleck spectral method). |

## Gross–Pitaevskii quantities

| Function | Quantity |
|---|---|
| `interaction_energy(wf, g)` | E_int = (g/2)∫\|ψ\|⁴ |
| `gross_pitaevskii_energy(wf, V, g)` | the energy functional ⟨T⟩ + ⟨V⟩ + E_int, conserved by real-time GP dynamics |
| `chemical_potential(wf, V, g)` | μ = ⟨T⟩ + ⟨V⟩ + 2E_int |

Pass `potential=None` for free space.
