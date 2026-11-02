import numpy as np
from quantumlab.observables.expectation import kinetic_energy_expectation, potential_energy_expectation
from quantumlab.core.spectral import volume_element


def interaction_energy(wf, nonlinearity: float) -> float:
    """Mean-field interaction energy E_int = (g/2) ∫ |ψ|⁴ dV."""
    return float(0.5 * nonlinearity * np.sum(wf.probability_density ** 2) * volume_element(wf.grid))


def gross_pitaevskii_energy(wf, potential, nonlinearity: float, hbar: float = 1.0, m: float = 1.0) -> float:
    """GP energy functional E[ψ] = ⟨T⟩ + ⟨V⟩ + (g/2)∫|ψ|⁴ (conserved by real-time GPE dynamics)."""
    return float(kinetic_energy_expectation(wf, hbar, m) + potential_energy_expectation(wf, potential)
                 + interaction_energy(wf, nonlinearity))


def chemical_potential(wf, potential, nonlinearity: float, hbar: float = 1.0, m: float = 1.0) -> float:
    """μ = ⟨T⟩ + ⟨V⟩ + g∫|ψ|⁴, the eigenvalue of the stationary GP equation."""
    return float(kinetic_energy_expectation(wf, hbar, m) + potential_energy_expectation(wf, potential)
                 + 2.0 * interaction_energy(wf, nonlinearity))


def coupled_gp_energy(states, potentials, couplings, hbar: float = 1.0, m: float = 1.0) -> float:
    """
    Energy functional of a multi-component condensate,
    E = Σ_i (⟨T⟩_i + ⟨V_i⟩) + ½ Σ_ij g_ij ∫ n_i n_j dV, with n_i = |ψ_i|².
    """
    g = np.asarray(couplings, dtype=float)
    dV = volume_element(states[0].grid)
    E = sum(kinetic_energy_expectation(s, hbar, m) + potential_energy_expectation(s, p)
            for s, p in zip(states, potentials))
    n = [s.probability_density for s in states]
    E += 0.5 * sum(g[i, j] * float(np.sum(n[i] * n[j]) * dV)
                   for i in range(len(states)) for j in range(len(states)))
    return float(E)


def miscibility_overlap(wf_a, wf_b) -> float:
    """
    Spatial overlap Λ = ∫ n_a n_b / sqrt(∫ n_a² ∫ n_b²) of two components:
    1 for identical density profiles, → 0 for fully separated ones.
    """
    na, nb = wf_a.probability_density, wf_b.probability_density
    return float(np.sum(na * nb) / np.sqrt(np.sum(na ** 2) * np.sum(nb ** 2)))
