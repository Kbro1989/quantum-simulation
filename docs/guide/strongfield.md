# Strong-field physics

`quantumlab.strongfield` collects the tools for atoms in intense laser fields: laser-parameter conversions, the classical three-step model of high-harmonic generation (HHG), and the spectral analysis of a time-dependent dipole. Everything is in atomic units (ℏ = mₑ = e = 1).

## A hydrogen atom in an 800 nm pulse

```python
import numpy as np
from quantumlab.core.grid import Grid1D
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.potentials import SoftCoulomb, DipoleDrive, flat_top_envelope
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.stationary import eigenstates
from quantumlab import strongfield as sf

grid = Grid1D(4096, -200.0, 200.0)
atom = SoftCoulomb()                                   # −1/√(x² + 2): ground state −0.5 a.u. (hydrogen)
E, states = eigenstates(grid, atom, n_states=1)

omega = sf.wavelength_to_omega(800.0)                  # 0.0570 a.u.
field = sf.intensity_to_field(1e14)                    # 0.0534 a.u.
period = 2 * np.pi / omega
laser = DipoleDrive(atom, F0=field, omega=omega, envelope=flat_top_envelope(20 * period, 3 * period))

solver = SplitOperatorSolver(grid, laser, dt=0.05, order=4, absorber=AbsorbingBoundaryLayer(grid, 40.0))
acc = []
solver.evolve(states[0], int(20 * period / 0.05),
              callback=lambda n, t, wf: acc.append(sf.dipole_acceleration(wf, atom, laser.force(t))))

orders, spectrum = sf.harmonic_spectrum(np.array(acc), 0.05, omega)
```

## Tools

| Function | Purpose |
|---|---|
| `wavelength_to_omega`, `intensity_to_field`, `ponderomotive_energy` | Laser parameters in atomic units. U_p = E₀²/4ω² is the quiver energy of a free electron. |
| `classical_returns` | Three-step model: for every birth phase, the return phase and the kinetic energy at return in units of U_p (maximum 3.17). |
| `classical_cutoff` | I_p + 3.17 U_p |
| `dipole_acceleration` | a(t) = ⟨−∂V₀/∂x⟩ + F(t) from Ehrenfest's theorem, using the potential's analytic gradient when available |
| `harmonic_spectrum` | \|FFT a\|² against harmonic order, with a Hann window |
| `gabor_transform` | Time–frequency map with a Gaussian window of width σ (time resolution σ, frequency resolution 1/σ) |
| `attosecond_pulses` | Intensity of the pulse train synthesised from a band of harmonics |

`SoftCoulomb(Z, softening, center)` is the soft-core Coulomb potential in 1D/2D/3D with an analytic `gradient`. `flat_top_envelope(duration, ramp)` gives long, smooth pulses whose harmonics are well resolved.

## What to expect

- **Odd harmonics only.** An atom and a linearly polarised field together are symmetric under x → −x combined with t → t + T/2, so even harmonics are forbidden. In example 12 they are 2300× weaker.
- **Plateau and cutoff.** The classical cutoff is I_p + 3.17 U_p. Quantum calculations (Lewenstein et al. 1994) give about 1.32 I_p + 3.17 U_p. A single 1D atom has a plateau with deep interference minima, and its roll-off is gradual. A numerically extracted "cutoff" therefore depends on how it is defined, while the linear growth with U_p is robust.
- **Emission times.** In a Gabor map, each harmonic is emitted twice per cycle, at the classical return times of the *short* and *long* trajectories.
- **Attosecond pulses.** Filtering a band of plateau harmonics gives one burst per half laser cycle.

!!! note
    1D models over-estimate ionisation compared with a 3D atom, because the electron cannot miss the ion. They reproduce the recollision physics (plateau, cutoff scaling, timing) and are the standard testbed for it.
