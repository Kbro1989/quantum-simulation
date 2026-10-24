# Examples

All scripts live in `examples/` and write their figures and animations to `output/`. Run them from the project root, for example `python examples/07_3d_double_slit.py`.

## 1. Gaussian barrier scattering

A 1D packet splits into reflected and transmitted parts, with R + T = 1 to six digits.

![Gaussian barrier](images/01_gaussian_barrier_final.png)
![Space-time](images/01_gaussian_barrier_spacetime.png)

## 2. Harmonic oscillator

A coherent state oscillates in a parabolic well; the total energy is conserved to better than 10⁻⁵.

![Harmonic oscillator](images/02_harmonic_oscillator_spacetime.png)

## 3. Double well

Tunnelling between two symmetric quartic wells.

![Double well](images/03_double_well_spacetime.png)

## 4. Multiple barriers

Interference fringes from scattering off a periodic barrier array.

![Multiple barriers](images/04_multiple_barriers_dual.png)

## 5. 2D Gaussian barrier

Diffraction around a radially symmetric barrier on a 256 × 256 grid.

![2D snapshots](images/05_2d_gaussian_barrier_snapshots.png)

## 6. 3D Gaussian barrier

Orthogonal slices of a 3D packet scattering off a spherical barrier.

![3D slices](images/06_3d_slices_t0200.png)

## 7. 3D double-pinhole interference

Young's experiment in full 3D on 1.2 million grid points. The fringe zeros of the transmitted momentum distribution lie within **1.6% and 0.3%** of the Fraunhofer prediction ±(2n+1)π/d. The density behind the screen is amplified ×40 for visibility, as stated on every frame.

![Double pinhole](images/07_double_slit_3d.gif)
![Fringes](images/07_double_slit_fringes.png)

## 8. BEC time of flight

A Gross–Pitaevskii ground state in a cigar-shaped trap is released, and the cloud turns into a pancake. The **aspect-ratio inversion happens at t = 1.776**, against the Castin–Dum scaling prediction of 1.772. The rms widths follow Castin–Dum to within 0.6%, the ground-state virial theorem holds to 10⁻⁴, and the norm is conserved to 12 digits.

![BEC time of flight](images/08_bec_tof_3d.gif)
![Aspect ratio](images/08_bec_aspect_ratio.png)

## 9. Bright soliton collisions

Solitons of the integrable 1D nonlinear Schrödinger equation pass through each other or repel, depending on their relative phase, and emerge unchanged. Norm and energy are conserved to 10 digits.

![Soliton collisions](images/09_soliton_collisions.png)

## 10. Quantum chaos in the stadium billiard

The first 60 eigenstates of a chaotic billiard computed with LOBPCG. LOBPCG and imaginary time agree to 10⁻⁶. The level count N(15.25) = 60 compares with 60.3 from Weyl's law.

![Stadium eigenstates](images/10_stadium_eigenstates.png)

## 11. Vortex lattice in a rotating BEC

A condensate in a rotating, slightly anisotropic trap develops surface instabilities. Vortices then enter and crystallise into a triangular lattice of 18 vortices (dissipative GPE, following Tsubota, Kasamatsu & Ueda 2002). The mean lattice spacing is within 4.5% of Feynman's rule n_v = mΩ/πℏ, which is exact only for an infinite uniform lattice. The 3D version shows the vortex lines.

![Vortex lattice formation](images/11_vortex_lattice_formation.gif)
![Vortex lines](images/11_vortex_lines_3d.gif)
