import numpy as np
import pytest
from quantumlab.core.grid import Grid3D
from quantumlab.core.wavefunction import WaveFunction3D
from quantumlab.observables.expectation import (
    position_expectation, position_uncertainty,
    y_expectation, y_uncertainty,
    z_expectation, z_uncertainty,
    momentum_expectation, momentum_uncertainty,
)
from quantumlab.observables.coefficients import transmission_coefficient, reflection_coefficient
from quantumlab.observables.momentum import get_momentum_probability_density


@pytest.fixture
def gaussian_3d():
    grid = Grid3D(64, -20.0, 20.0, 64, -20.0, 20.0, 64, -20.0, 20.0)
    params = dict(x0=1.5, y0=-2.0, z0=0.5, k0_x=1.2, k0_y=-0.8, k0_z=0.4,
                  sigma_x=2.0, sigma_y=2.5, sigma_z=1.8)
    wf = WaveFunction3D.gaussian(grid, **params)
    return wf, params


def test_3d_position_expectation_and_uncertainty(gaussian_3d):
    wf, p = gaussian_3d
    assert position_expectation(wf) == pytest.approx(p['x0'], abs=1e-6)
    assert y_expectation(wf) == pytest.approx(p['y0'], abs=1e-6)
    assert z_expectation(wf) == pytest.approx(p['z0'], abs=1e-6)
    assert position_uncertainty(wf) == pytest.approx(p['sigma_x'], abs=1e-6)
    assert y_uncertainty(wf) == pytest.approx(p['sigma_y'], abs=1e-6)
    assert z_uncertainty(wf) == pytest.approx(p['sigma_z'], abs=1e-6)


def test_3d_momentum_expectation_and_heisenberg(gaussian_3d):
    wf, p = gaussian_3d
    hbar = 1.0
    for axis, k0, sigma, dx in [
        ('x', p['k0_x'], p['sigma_x'], position_uncertainty(wf)),
        ('y', p['k0_y'], p['sigma_y'], y_uncertainty(wf)),
        ('z', p['k0_z'], p['sigma_z'], z_uncertainty(wf)),
    ]:
        assert momentum_expectation(wf, axis, hbar) == pytest.approx(hbar * k0, abs=1e-6)
        dp = momentum_uncertainty(wf, axis, hbar)
        assert dp == pytest.approx(hbar / (2.0 * sigma), abs=1e-6)
        assert dx * dp == pytest.approx(0.5 * hbar, abs=1e-6)


def test_3d_momentum_invalid_direction(gaussian_3d):
    wf, _ = gaussian_3d
    with pytest.raises(ValueError):
        momentum_expectation(wf, 'w')


def test_3d_z_expectation_rejects_lower_dimensions():
    from quantumlab.core.grid import Grid2D
    from quantumlab.core.wavefunction import WaveFunction2D
    grid = Grid2D(32, -5.0, 5.0, 32, -5.0, 5.0)
    wf = WaveFunction2D.gaussian(grid, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        z_expectation(wf)


def test_3d_transmission_reflection_sum_to_norm(gaussian_3d):
    wf, p = gaussian_3d
    T = transmission_coefficient(wf, barrier_position=p['x0'])
    R = reflection_coefficient(wf, barrier_position=p['x0'])
    assert T + R == pytest.approx(wf.norm(), abs=1e-12)
    # Barrier at the packet centre splits it roughly in half
    assert T == pytest.approx(0.5, abs=0.05)


def test_3d_momentum_probability_density(gaussian_3d):
    wf, _ = gaussian_3d
    (kx, ky, kz), prob_k = get_momentum_probability_density(wf)
    grid = wf.grid
    assert prob_k.shape == grid.shape
    assert np.all(np.diff(kx) > 0) and np.all(np.diff(ky) > 0) and np.all(np.diff(kz) > 0)
    # Parseval: momentum-space density integrates to the same norm
    norm_k = np.sum(prob_k) * grid.dk_x * grid.dk_y * grid.dk_z
    assert norm_k == pytest.approx(wf.norm(), abs=1e-10)
