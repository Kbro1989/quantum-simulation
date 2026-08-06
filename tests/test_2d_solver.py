import numpy as np
import pytest
from quantumlab.core.grid import Grid1D, Grid2D
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D
from quantumlab.potentials.barriers import GaussianBarrier
from quantumlab.potentials.oscillator import HarmonicOscillator
from quantumlab.solvers.split_step_2d import SplitStep2DSolver
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.observables.expectation import total_energy_expectation

def test_2d_norm_conservation():
    """Verify that the 2D SSFM conserves norm to machine precision."""
    grid = Grid2D(128, -20.0, 20.0, 128, -20.0, 20.0)
    wf = WaveFunction2D.gaussian(grid, x0=-5.0, y0=0.0, k0_x=2.0, k0_y=0.0, sigma_x=1.0, sigma_y=1.0)
    potential = GaussianBarrier(V0=5.0, width=2.0, position=0.0)
    solver = SplitStep2DSolver(grid, potential, dt=0.02)
    assert wf.norm() == pytest.approx(1.0, abs=1e-12)
    wf_current = wf
    for _ in range(50):
        wf_current = solver.step(wf_current)
    assert wf_current.norm() == pytest.approx(1.0, abs=1e-12)

def test_2d_energy_conservation():
    """Verify energy conservation in a 2D harmonic oscillator."""
    grid = Grid2D(128, -10.0, 10.0, 128, -10.0, 10.0)
    wf = WaveFunction2D.gaussian(grid, x0=0.0, y0=0.0, k0_x=0.0, k0_y=0.0, sigma_x=1.5, sigma_y=1.5)
    potential = HarmonicOscillator(omega=1.0, m=1.0, position=0.0)
    solver = SplitStep2DSolver(grid, potential, dt=0.01)
    initial_energy = total_energy_expectation(wf, potential)
    wf_current = wf
    for _ in range(50):
        wf_current = solver.step(wf_current)
    final_energy = total_energy_expectation(wf_current, potential)
    assert final_energy == pytest.approx(initial_energy, rel=1e-05)

def test_absorbing_boundary_1d():
    """Verify that the ABL mask is 1.0 in the interior and 0.0 at edges."""
    grid = Grid1D(256, -50.0, 50.0)
    abl = AbsorbingBoundaryLayer(grid, boundary_width=10.0, order=3)
    mask = abl.mask
    center_idx = len(mask) // 2
    assert mask[center_idx] == pytest.approx(1.0, abs=1e-10)
    assert mask[0] < 0.01
    assert mask[-1] < 0.01

def test_absorbing_boundary_2d():
    """Verify that the 2D ABL mask has correct shape and boundary behavior."""
    grid = Grid2D(128, -20.0, 20.0, 128, -20.0, 20.0)
    abl = AbsorbingBoundaryLayer(grid, boundary_width=5.0, order=3)
    assert abl.mask.shape == (128, 128)
    assert abl.mask[64, 64] == pytest.approx(1.0, abs=1e-10)
    assert abl.mask[0, 0] < 0.01
    assert abl.mask[-1, -1] < 0.01

def test_abl_norm_decay():
    """Verify that applying ABL causes norm decay (absorption is working)."""
    grid = Grid1D(256, -50.0, 50.0)
    wf = WaveFunction1D.gaussian(grid, x0=47.0, k0=0.0, sigma=1.5)
    abl = AbsorbingBoundaryLayer(grid, boundary_width=10.0, order=3)
    initial_norm = wf.norm()
    abl.apply(wf)
    final_norm = wf.norm()
    assert final_norm < initial_norm
