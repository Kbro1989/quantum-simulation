import numpy as np
import pytest
import yaml
from quantumlab.runner import SimulationRunner
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D, WaveFunction3D

SIMULATIONS = {
    'sim_1d': {
        'dimension': 1,
        'grid': {'N': 128, 'L': 40.0},
        'wavefunction': {'x0': -5.0, 'sigma': 1.0, 'k0': 2.0},
        'potential': {'type': 'gaussian_barrier', 'params': {'V0': 2.0, 'width': 1.0, 'position': 0.0}},
    },
    'sim_2d': {
        'dimension': 2,
        'grid': {'N': 32, 'L': 20.0},
        'wavefunction': {'x0': -3.0, 'sigma': 1.0, 'k0_x': 1.5},
        'potential': {'type': 'gaussian_2d_barrier', 'params': {'V0': 2.0, 'width': 1.0}},
    },
    'sim_3d': {
        'dimension': 3,
        'grid': {'N': 16, 'L': 20.0},
        'wavefunction': {'x0': -3.0, 'sigma': 1.5, 'k0_x': 1.0},
        'potential': {'type': 'gaussian_3d_barrier', 'params': {'V0': 2.0, 'width': 1.0}},
    },
    'sim_3d_gpu': {
        'dimension': 3,
        'gpu': True,
        'grid': {'N': 16, 'L': 20.0},
        'wavefunction': {'x0': -3.0, 'sigma': 1.5, 'k0_x': 1.0},
        'potential': {'type': 'gaussian_3d_barrier', 'params': {'V0': 2.0, 'width': 1.0}},
    },
}

WAVEFUNCTION_TYPES = {
    'sim_1d': WaveFunction1D,
    'sim_2d': WaveFunction2D,
    'sim_3d': WaveFunction3D,
    'sim_3d_gpu': WaveFunction3D,
}


@pytest.fixture
def config_path(tmp_path):
    sims = {}
    for name, cfg in SIMULATIONS.items():
        # log_interval=1 exercises the energy/<x> logging on every step
        sims[name] = dict(cfg, dt=0.01, num_steps=3, log_interval=1, snapshot_interval=1)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'output_dir': str(tmp_path / 'out'), 'simulations': sims}))
    return path


@pytest.mark.parametrize('sim_name', list(SIMULATIONS))
def test_runner_runs_each_dimension(config_path, sim_name):
    results = SimulationRunner(str(config_path)).run(sim_name)
    result = results[sim_name]
    wf = result['final_wavefunction']
    assert isinstance(wf, WAVEFUNCTION_TYPES[sim_name])
    assert wf.norm() == pytest.approx(1.0, abs=1e-10)
    assert len(result['snapshots']) == 4
    assert result['snap_times'] == pytest.approx([0.0, 0.01, 0.02, 0.03])


@pytest.mark.parametrize('dim', [1, 3])
def test_runner_split_operator_solver(tmp_path, dim):
    base = SIMULATIONS['sim_1d' if dim == 1 else 'sim_3d']
    cfg = dict(base, solver='split_operator', order=4, nonlinearity=0.5, dt=0.01, num_steps=3, log_interval=1)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'output_dir': str(tmp_path / 'out'), 'simulations': {'s': cfg}}))
    wf = SimulationRunner(str(path)).run()['s']['final_wavefunction']
    assert wf.norm() == pytest.approx(1.0, abs=1e-10)


def test_runner_rotating_damped_split_operator(tmp_path):
    cfg = dict(SIMULATIONS['sim_2d'], solver='split_operator', nonlinearity=5.0, rotation=0.5,
               damping=0.02, backend='auto', dt=0.01, num_steps=3, log_interval=1)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'output_dir': str(tmp_path / 'out'), 'simulations': {'s': cfg}}))
    wf = SimulationRunner(str(path)).run()['s']['final_wavefunction']
    assert wf.norm() == pytest.approx(1.0, abs=1e-10)


def test_runner_save_state(tmp_path):
    cfg = dict(SIMULATIONS['sim_1d'], dt=0.01, num_steps=2, save_state=True)
    path = tmp_path / 'config.yaml'
    out_dir = tmp_path / 'out'
    path.write_text(yaml.safe_dump({'output_dir': str(out_dir), 'simulations': {'s': cfg}}))
    results = SimulationRunner(str(path)).run()
    data = np.load(out_dir / 's_state.npz', allow_pickle=True)
    np.testing.assert_allclose(data['psi'], results['s']['final_wavefunction'].psi)


def test_runner_missing_keys_raises(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'output_dir': str(tmp_path / 'out'), 'simulations': {'bad': {'dimension': 1}}}))
    with pytest.raises(ValueError, match='missing required config keys'):
        SimulationRunner(str(path)).run()


def test_runner_unknown_simulation_raises(config_path):
    with pytest.raises(KeyError):
        SimulationRunner(str(config_path)).run('does_not_exist')
