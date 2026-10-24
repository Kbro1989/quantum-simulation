# Potentials

Every potential implements `evaluate(grid)` and returns V on the grid. Most work in 1D, 2D and 3D. They can be created directly or by name through `create_potential`, which the YAML runner uses:

```python
from quantumlab.potentials import create_potential, HarmonicTrap

trap = HarmonicTrap((1.0, 1.0, 0.5))                 # anisotropic 3D trap
screen = create_potential('double_slit', V0=60.0, separation=5.0, aperture='pinhole')
```

| Registry name | Class | Description |
|---|---|---|
| `gaussian_barrier`, `barrier` | `GaussianBarrier` | 1D Gaussian barrier |
| `gaussian_2d_barrier` | `Gaussian2DBarrier` | radially symmetric barrier in 2D (or a cylinder in 3D) |
| `gaussian_3d_barrier` | `Gaussian3DBarrier` | spherical barrier |
| `rectangular_barrier` | `RectangularBarrier` | square barrier |
| `potential_step` | `PotentialStep` | step potential |
| `multiple_barriers` | `MultipleBarriers` | periodic array of barriers |
| `rtd` | `ResonantTunnelingDiode` | double-barrier resonant tunnelling structure |
| `infinite_well`, `finite_well` | `InfiniteSquareWell`, `FiniteSquareWell` | square wells |
| `double_well` | `DoubleWell` | quartic double well |
| `harmonic` | `HarmonicOscillator` | isotropic harmonic oscillator |
| `harmonic_trap` | `HarmonicTrap` | anisotropic harmonic trap with an optional centre |
| `aperture_screen`, `double_slit` | `ApertureScreen` | opaque screen with N slits or circular pinholes (2D/3D) |
| `stadium` | `StadiumBilliard` | soft-walled Bunimovich stadium (classically chaotic) |
| `crystal` | `CrystalPotential` | sinusoidal lattice |
| `disorder` | `RandomDisorder` | cell-based random potential |

Sharp edges (screens, billiards) are smoothed over a short length with a tanh profile, so the potential stays resolvable on the FFT grid.

## Custom potentials

```python
from quantumlab.potentials import CustomPotential
V = CustomPotential(lambda x, y, z: 0.5 * (x**2 + y**2) + 0.1 * z**4)
```

## Time-dependent potentials

```python
import numpy as np
from quantumlab.potentials import DipoleDrive, HarmonicTrap, TimeDependentCustomPotential, sin2_envelope

# static potential plus a uniform oscillating force, F(t) = F0 env(t) sin(ωt + φ)
drive = DipoleDrive(base=HarmonicTrap(1.0), F0=0.5, omega=0.6, axis='x', envelope=sin2_envelope(40.0))

# any V(r, t)
shaken = TimeDependentCustomPotential(lambda x, t: 0.5 * (x - 0.5 * np.sin(t))**2)
```

Solvers detect these through `time_dependent = True` and call `evaluate_at(grid, t)` at every sub-step.
