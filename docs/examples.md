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

## 12. High-harmonic generation in hydrogen

A hydrogen atom (soft-core model, I_p = 13.6 eV) is driven by an 800 nm pulse at 10¹⁴ W/cm². The electron tunnels out, is driven back and recollides, emitting odd harmonics up to a cutoff.
- **Odd harmonics only:** even orders are 2300× weaker, as inversion symmetry requires.
- **Emission times:** each harmonic is emitted when the classical short and long trajectories return, within the 0.05-cycle resolution of the analysis.
- **Attosecond pulses:** filtering H11–H21 gives bursts of about 200 as, one every half cycle (spacing 0.499–0.503 cycle).
- **Cutoff scaling:** an intensity scan shows the cutoff growing linearly with U_p. The fitted spectral edge, 3.48 U_p + 1.52 I_p, lies near the three-step laws, and its exact value depends on how the edge is defined.

![HHG spectrum](images/12_hhg_spectrum.png)
![Time-frequency analysis](images/12_hhg_time_frequency.png)
![Attosecond pulses](images/12_attosecond_pulses.png)
![Electron in space and time](images/12_electron_spacetime.png)
![Cutoff scan](images/12_hhg_cutoff_scan.png)

## 13. Two-component condensates

Two condensates with couplings g₁₁ = g₂₂ = g and g₁₂:
- **Miscibility:** in a trap, the overlap of the ground states drops sharply just above g₁₂ = g. The uniform criterion is g₁₂² = g₁₁g₂₂, and a finite cloud shifts the transition slightly upward.
- **Spinodal decomposition:** a uniform mixture with g₁₂ = 1.3g separates into labyrinthine domains. Their growth on the |k| ≈ k\* shell matches the Bogoliubov rate γ\* = n(g₁₂ − g) to 0.3%, and the structure-factor peak sits at k\* within one bin. Afterwards the domains coarsen.
- **Two-colour vortex lattices:** in a rotating trap both components nucleate vortices, which interlace. Every vortex core of one component sits on a density maximum of the other, at a median 1.8× its local average.

![Phase separation](images/13_phase_separation.gif)
![Interlaced vortex lattices](images/13_vortex_lattices.gif)
![Miscibility](images/13_miscibility.png)
![Growth of the instability](images/13_phase_separation_growth.png)
![Interlaced lattices](images/13_vortex_lattices.png)

