# QuantumLab

**QuantumLab** is a Python framework for numerical quantum mechanics. It solves the time-dependent Schrödinger equation and the nonlinear Gross–Pitaevskii equation in **1D, 2D and 3D** with split-step Fourier methods, computes stationary states with matrix-free eigensolvers, runs on CUDA GPUs, and renders 3D wavefunctions as animated isosurfaces.

<div class="grid" markdown>

![3D double-pinhole interference](images/07_double_slit_3d.gif){ width="49%" }
![BEC time of flight](images/08_bec_tof_3d.gif){ width="49%" }

</div>

*Left: Young's double-pinhole experiment in 3D. Right: a Bose–Einstein condensate released from a cigar-shaped trap, coloured by the quantum phase arg ψ.*

## What it does

| Area | Capabilities |
|---|---|
| **Time evolution** | Split-operator propagation of order 2 (Strang) and 4 (Yoshida); time-dependent potentials V(r, t); Gross–Pitaevskii nonlinearity; rotating frames (H − ΩL_z); dissipative dynamics; absorbing boundaries |
| **Stationary states** | Lanczos and preconditioned LOBPCG eigensolvers on a matrix-free FFT Hamiltonian; imaginary-time ground and excited states, including GP ground states in rotating frames |
| **Physics models** | Barriers, wells, (anisotropic) harmonic traps, multi-slit and pinhole screens, chaotic stadium billiard, lattices, disorder, laser-like dipole driving, custom potentials |
| **Observables** | ⟨x⟩, ⟨p⟩, energies, uncertainties, transmission and reflection, probability current, ⟨L_z⟩, autocorrelation spectra, vortex detection |
| **Performance** | CPU (SciPy FFT, multithreaded) or CUDA GPU through CuPy, 7.6–12.5× faster than Colab's CPU on a Tesla T4 |
| **Visualization** | Publication-style 1D/2D plots, 3D isosurfaces coloured by phase, orbiting-camera MP4/GIF animations, interactive HTML |

## Trustworthy by construction

Every feature is checked against exact, analytic or independent numerical results. Examples include convergence orders 2.001 and 3.997, a driven oscillator against its classical trajectory, harmonic spectra to 10⁻¹³, exact soliton propagation, the Gross–Pitaevskii virial theorem, and exact rotating-frame solutions. The tests run on Python 3.10–3.14 in continuous integration, and the GPU path has been verified on real NVIDIA hardware. See [Validation](validation.md).

## Where to go next

- [Getting started](getting-started.md): installation and a first simulation
- [User guide](guide/solvers.md): the solvers, stationary states, potentials and visualization in depth
- [Examples](examples.md): eleven ready-to-run simulations with their results
- [API reference](api/core.md): every class and function, generated from the source
