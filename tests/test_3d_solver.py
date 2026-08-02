import numpy as np
import pytest
from quantumlab.core.grid import Grid3D
from quantumlab.core.wavefunction import WaveFunction3D
from quantumlab.potentials.barriers import Gaussian3DBarrier
from quantumlab.potentials.oscillator import HarmonicOscillator
from quantumlab.solvers.split_step_3d import SplitStep3DSolver
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.observables.expectation import total_energy_expectation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(N: int = 32, L: float = 10.0) -> Grid3D:
    return Grid3D(N, -L, L, N, -L, L, N, -L, L)


def _make_gaussian_wf(grid: Grid3D, x0: float = -3.0, k0_x: float = 1.5,
                      sigma: float = 1.0) -> WaveFunction3D:
    return WaveFunction3D.gaussian(
        grid,
        x0=x0, y0=0.0, z0=0.0,
        k0_x=k0_x, k0_y=0.0, k0_z=0.0,
        sigma_x=sigma, sigma_y=sigma, sigma_z=sigma,
    )


# ---------------------------------------------------------------------------
# Grid3D tests
# ---------------------------------------------------------------------------

def test_grid3d_shape():
    grid = _make_grid(32)
    assert grid.shape == (32, 32, 32)
    assert grid.X.shape == (32, 32, 32)
    assert grid.K_x.shape == (32, 32, 32)


def test_grid3d_frequency_zero_dc():
    """DC bin of fftfreq should be zero."""
    grid = _make_grid(32)
    assert grid.k_x[0] == pytest.approx(0.0)
    assert grid.k_y[0] == pytest.approx(0.0)
    assert grid.k_z[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# WaveFunction3D tests
# ---------------------------------------------------------------------------

def test_3d_gaussian_norm():
    """Freshly constructed Gaussian wavefunction should be normalised."""
    grid = _make_grid(32)
    wf = _make_gaussian_wf(grid)
    assert wf.norm() == pytest.approx(1.0, abs=1e-10)


def test_3d_wavefunction_slices_shape():
    grid = _make_grid(32)
    wf = _make_gaussian_wf(grid)
    assert wf.slice_xy().shape == (32, 32)
    assert wf.slice_xz().shape == (32, 32)
    assert wf.slice_yz().shape == (32, 32)


# ---------------------------------------------------------------------------
# SplitStep3DSolver — norm conservation
# ---------------------------------------------------------------------------

def test_3d_norm_conservation():
    """SSFM must exactly conserve norm (unitarity) without ABL."""
    grid = _make_grid(32)
    wf = _make_gaussian_wf(grid, x0=-3.0, k0_x=1.5)
    potential = Gaussian3DBarrier(V0=4.0, width=2.0, x0=0.0, y0=0.0, z0=0.0)
    solver = SplitStep3DSolver(grid, potential, dt=0.02)

    assert wf.norm() == pytest.approx(1.0, abs=1e-10)
    wf_current = wf
    for _ in range(30):
        wf_current = solver.step(wf_current)
    assert wf_current.norm() == pytest.approx(1.0, abs=1e-10)


def test_3d_energy_conservation_harmonic():
    """Energy must be conserved in a 3D isotropic harmonic oscillator."""
    grid = _make_grid(32, L=8.0)
    wf = WaveFunction3D.gaussian(
        grid, x0=0.0, y0=0.0, z0=0.0,
        k0_x=0.0, k0_y=0.0, k0_z=0.0,
        sigma_x=1.2, sigma_y=1.2, sigma_z=1.2,
    )
    potential = HarmonicOscillator(omega=1.0, m=1.0)
    solver = SplitStep3DSolver(grid, potential, dt=0.01)

    E0 = total_energy_expectation(wf, potential)
    wf_current = wf
    for _ in range(30):
        wf_current = solver.step(wf_current)
    E1 = total_energy_expectation(wf_current, potential)
    assert E1 == pytest.approx(E0, rel=1e-4)


# ---------------------------------------------------------------------------
# AbsorbingBoundaryLayer — 3D
# ---------------------------------------------------------------------------

def test_abl_3d_shape():
    grid = _make_grid(32)
    abl = AbsorbingBoundaryLayer(grid, boundary_width=3.0, order=3)
    assert abl.mask.shape == (32, 32, 32)


def test_abl_3d_interior_unity():
    """Mask should be 1.0 at the dead centre of the grid."""
    grid = _make_grid(32, L=20.0)
    abl = AbsorbingBoundaryLayer(grid, boundary_width=3.0, order=3)
    cx, cy, cz = 16, 16, 16
    assert abl.mask[cx, cy, cz] == pytest.approx(1.0, abs=1e-10)


def test_abl_3d_corners_absorbed():
    """All eight corners of the mask should be substantially attenuated."""
    grid = _make_grid(32, L=20.0)
    abl = AbsorbingBoundaryLayer(grid, boundary_width=3.0, order=3)
    for ix in (0, -1):
        for iy in (0, -1):
            for iz in (0, -1):
                # With bw=3 over L=40 the corner is ~5.3% — check it is well below 0.5
                assert abl.mask[ix, iy, iz] < 0.5


def test_abl_3d_norm_decay():
    """ABL applied to an edge-localised wavepacket must reduce its norm."""
    grid = _make_grid(64, L=20.0)
    wf = WaveFunction3D.gaussian(
        grid, x0=18.0, y0=0.0, z0=0.0,
        k0_x=0.0, k0_y=0.0, k0_z=0.0,
        sigma_x=1.0, sigma_y=1.0, sigma_z=1.0,
    )
    abl = AbsorbingBoundaryLayer(grid, boundary_width=4.0, order=3)
    norm_before = wf.norm()
    abl.apply(wf)
    assert wf.norm() < norm_before


# ---------------------------------------------------------------------------
# Gaussian3DBarrier
# ---------------------------------------------------------------------------

def test_gaussian3d_barrier_peak():
    """Barrier should reach V0 at its centre."""
    grid = _make_grid(32, L=10.0)
    barrier = Gaussian3DBarrier(V0=5.0, width=2.0, x0=0.0, y0=0.0, z0=0.0)
    V = barrier.evaluate(grid)
    cx = grid.N_x // 2
    cy = grid.N_y // 2
    cz = grid.N_z // 2
    assert V[cx, cy, cz] == pytest.approx(5.0, rel=1e-3)


def test_gaussian3d_barrier_symmetry():
    """Barrier must be spherically symmetric: V(r) = V(-r) for antipodal points."""
    grid = _make_grid(32, L=10.0)
    barrier = Gaussian3DBarrier(V0=5.0, width=2.0)
    V = barrier.evaluate(grid)
    # On linspace(-10, 10, 32, endpoint=False), index 11 maps to x≈-3.125
    # and index 21 maps to x≈+3.125 — exactly antipodal.
    assert V[11, 11, 11] == pytest.approx(V[21, 21, 21], rel=1e-5)
