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
