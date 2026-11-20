"""
testbench_interlock.py — minimal, reproducible, verifiable classical↔quantum interlocking.

Classical sub-layer (deterministic, non-unitary):
    SHA-256 hash → seed → 9-bit/512-state phase tables → emotional 5-axis vector
    → selects quantum initial conditions, potential parameters, grid only.

Quantum core (must remain unitary):
    Grid1D / WaveFunction1D / SplitStep1DSolver / GaussianBarrier
    Unitary split-step Fourier evolution, norm + energy conservation.

Quantum → Classical projection:
    Post-evolution expectation values projected onto nearest discrete phase/emotional/pellet label.
    Never replaces complex amplitudes inside the unitary step.

Run:  python testbench_interlock.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

# --- quantumlab: real API (lowercase functions, not expectation.xxx) ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "quantum-simulation-main/quantum-simulation-main"))
from quantumlab.core.grid import Grid1D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.solvers import SplitStep1DSolver
from quantumlab.potentials import GaussianBarrier
from quantumlab.observables import (
    position_expectation,
    momentum_expectation,
    total_energy_expectation,
)

# ---------- Classical deterministic sub-layer ----------

PHASE_TEMPORALS = [
    "past", "present", "future", "transition",
    "resolution", "dissolution", "crystallization", "void",
]
PHASE_POLARITIES = [
    "yin", "yang", "yao", "yin-yang",
    "yang-yin", "yin-yao", "yang-yao", "yao-yao",
]
EMOTIONAL_KEYS = ["chaos", "whimsy", "darkTone", "coherence", "voiceWeight"]
MODULI = [97, 89, 83, 79, 73]


def deterministic_seed(n: int) -> int:
    h = int(hashlib.sha256(str(n).encode()).hexdigest(), 16)
    return h % 97


def phase_from_seed(seed: int) -> dict:
    """Return the full generative phase descriptor from a deterministic seed."""
    n = seed % 512
    hexagram = n // 8 + 1
    phase_idx = n % 8
    emotional_vector = np.array([
        (seed % 97) / 97.0 * 0.12,
        ((seed // 7) % 89) / 89.0 * 0.12,
        ((seed // 13) % 83) / 83.0 * 0.12,
        ((seed // 19) % 79) / 79.0 * 0.12,
        ((seed // 23) % 73) / 73.0 * 0.12,
    ])
    return {
        "hexagram": hexagram,
        "phase_bits": phase_idx,
        "phase_temporal": PHASE_TEMPORALS[phase_idx],
        "phase_polarity": PHASE_POLARITIES[phase_idx],
        "phase_changing_lines": bin(phase_idx)[2:].zfill(3),
        "emotional_vector": emotional_vector,
        "audio_pellet_id": seed % 384,
        "vortex_rank": float(np.exp(-((seed % 64) - 32) ** 2 / 200.0)),
    }


# ---------- Quantum core ----------

def run_quantum_segment(seed: int, steps: int = 400) -> dict:
    phase = phase_from_seed(seed)

    # Classical → Quantum: seed selects initial condition + potential only.
    x0 = -20.0 + 5.0 * phase["emotional_vector"][0]
    k0 = 2.0 + 3.0 * phase["emotional_vector"][3]
    V0 = 4.0 + 4.0 * phase["vortex_rank"]

    grid = Grid1D(N=1024, x_min=-50.0, x_max=50.0)
    wf = WaveFunction1D.gaussian(grid, x0=x0, k0=k0, sigma=3.5)
    pot = GaussianBarrier(V0=V0, width=1.2, position=0.0)
    solver = SplitStep1DSolver(grid, pot, dt=0.05)

    norm0 = wf.norm()
    E0 = total_energy_expectation(wf, pot)

    for _ in range(steps):
        wf = solver.step(wf)

    norm1 = wf.norm()
    E1 = total_energy_expectation(wf, pot)
    x_exp = position_expectation(wf)
    p_exp = momentum_expectation(wf)

    # Quantum → Classical projection (after evolution, never inside it).
    projected_label = int((x_exp + 50) / 100 * 511) % 512
    return {
        "phase": phase,
        "norm_drift": abs(norm1 - norm0),
        "energy_drift": abs(E1 - E0),
        "x_exp": x_exp,
        "p_exp": p_exp,
        "projected_classical_label": projected_label,
        "audio_pellet": phase["audio_pellet_id"],
        "emotional_vector": phase["emotional_vector"],
    }


# ---------- Verification harness ----------

def test_interlock(n_seeds: int = 32) -> list:
    results = []
    for n in range(n_seeds):
        r = run_quantum_segment(n)
        results.append(r)

        # Unitarity
        assert r["norm_drift"] < 1e-10, f"Norm conservation violated at seed {n}: drift={r['norm_drift']}"
        # Energy conservation (tolerance for split-step)
        assert r["energy_drift"] < 1e-3, f"Energy conservation violated at seed {n}: drift={r['energy_drift']}"

        # Deterministic reproducibility: same seed → identical outputs
        r2 = run_quantum_segment(n)
        assert r["projected_classical_label"] == r2["projected_classical_label"], f"Non-deterministic label at seed {n}"
        assert np.allclose(r["emotional_vector"], r2["emotional_vector"]), f"Non-deterministic emotional vector at seed {n}"
        assert r["norm_drift"] == r2["norm_drift"], f"Non-deterministic norm drift at seed {n}"
        assert r["energy_drift"] == r2["energy_drift"], f"Non-deterministic energy drift at seed {n}"

    print(f"[PASS] Classical↔Quantum interlocking invariants held over {n_seeds} seeds.")
    print(f"       max norm_drift   = {max(r['norm_drift'] for r in results):.2e}")
    print(f"       max energy_drift = {max(r['energy_drift'] for r in results):.2e}")
    print(f"       projected labels: min={min(r['projected_classical_label'] for r in results)} "
          f"max={max(r['projected_classical_label'] for r in results)}")
    return results


if __name__ == "__main__":
    test_interlock()
