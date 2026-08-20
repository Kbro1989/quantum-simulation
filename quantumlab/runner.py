"""
Deterministic batch simulation runner driven by YAML configuration.
Integrates with sovereign stack via state-address injection.
"""
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any

from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D, WaveFunction3D
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.potentials import create_potential
from quantumlab.solvers import SplitStep1DSolver, SplitStep2DSolver, SplitStep3DSolver
from quantumlab.solvers.split_step_3d_gpu import SplitStep3DGPU
from quantumlab.observables.expectation import total_energy_expectation, position_expectation
from quantumlab.visualization.animation import WavePacketAnimator


class SimulationRunner:
    """
    Sovereign simulation runner. No pseudo-RNG. All parameters are
    explicitly declared in config or derived from deterministic hash.
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.output_dir = Path(self.config.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_grid(self, dim: int, cfg: Dict[str, Any]):
        N = cfg['N']
        L = cfg['L']
        if dim == 1:
            return Grid1D(N, -L / 2, L / 2)
        elif dim == 2:
            return Grid2D(N, -L / 2, L / 2, N, -L / 2, L / 2)
        elif dim == 3:
            Nx, Ny, Nz = cfg.get('Nx', N), cfg.get('Ny', N), cfg.get('Nz', N)
            Lx, Ly, Lz = cfg.get('Lx', L), cfg.get('Ly', L), cfg.get('Lz', L)
            return Grid3D(Nx, -Lx / 2, Lx / 2, Ny, -Ly / 2, Ly / 2, Nz, -Lz / 2, Lz / 2)

    def _build_wavefunction(self, grid, cfg: Dict[str, Any]):
        dim = cfg['dimension']
        wfc = cfg['wavefunction']
        if dim == 1:
            return WaveFunction1D.gaussian(
                grid, x0=wfc['x0'], k0=wfc.get('k0', 0.0), sigma=wfc['sigma']
            )
        elif dim == 2:
            return WaveFunction2D.gaussian(
                grid, x0=wfc['x0'], y0=wfc.get('y0', 0.0),
                k0_x=wfc.get('k0_x', 0.0), k0_y=wfc.get('k0_y', 0.0),
                sigma_x=wfc['sigma'], sigma_y=wfc.get('sigma_y', wfc['sigma'])
            )
        elif dim == 3:
            return WaveFunction3D.gaussian(
                grid,
                x0=wfc['x0'], y0=wfc.get('y0', 0.0), z0=wfc.get('z0', 0.0),
                k0_x=wfc.get('k0_x', 0.0), k0_y=wfc.get('k0_y', 0.0),
                k0_z=wfc.get('k0_z', 0.0),
                sigma_x=wfc['sigma'], sigma_y=wfc.get('sigma_y', wfc['sigma']),
                sigma_z=wfc.get('sigma_z', wfc['sigma'])
            )

    def _build_solver(self, grid, potential, cfg: Dict[str, Any]):
        dim = cfg['dimension']
        dt = cfg['dt']
        hbar = cfg.get('hbar', 1.0)
        m = cfg.get('m', 1.0)
        gpu = cfg.get('gpu', False)

        if dim == 1:
            return SplitStep1DSolver(grid, potential, dt, hbar, m)
        elif dim == 2:
            return SplitStep2DSolver(grid, potential, dt, hbar, m)
        elif dim == 3:
            if gpu:
                return SplitStep3DGPU(grid, potential, dt, hbar, m)
            return SplitStep3DSolver(grid, potential, dt, hbar, m)

    def run(self, sim_name: str = None):
        sims = self.config['simulations']
        if sim_name:
            sims = {k: v for k, v in sims.items() if k == sim_name}

        results = {}
        for name, cfg in sims.items():
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print(f"{'=' * 60}")

            dim = cfg['dimension']
            grid = self._build_grid(dim, cfg['grid'])
            potential = create_potential(cfg['potential']['type'], **cfg['potential']['params'])
            wf = self._build_wavefunction(grid, cfg)
            solver = self._build_solver(grid, potential, cfg)

            abl_cfg = cfg.get('absorbing_boundary')
            abl = None
            if abl_cfg:
                abl = AbsorbingBoundaryLayer(grid, abl_cfg['width'], abl_cfg.get('order', 3))

            num_steps = cfg['num_steps']
            snapshot_interval = cfg.get('snapshot_interval', 0)
            snapshots = []
            snap_times = []

            if snapshot_interval > 0:
                snapshots.append(wf.copy())
                snap_times.append(0.0)

            wf_current = wf
            for step in range(1, num_steps + 1):
                wf_current = solver.step(wf_current)
                if abl:
                    abl.apply(wf_current)

                if snapshot_interval > 0 and step % snapshot_interval == 0:
                    snapshots.append(wf_current.copy())
                    snap_times.append(step * cfg['dt'])

                if step % cfg.get('log_interval', 100) == 0:
                    E = total_energy_expectation(wf_current, potential)
                    x = position_expectation(wf_current)
                    print(f"  Step {step:>5}/{num_steps}  E={E:.6f}  <x>={x:+.3f}  [{solver.backend}]")

            results[name] = {
                'final_wavefunction': wf_current,
                'snapshots': snapshots,
                'snap_times': snap_times,
                'config': cfg,
            }

            if cfg.get('save_state'):
                np.savez(self.output_dir / f"{name}_state.npz", psi=wf_current.psi, params=cfg)

            # Post-process: animation export
            anim_cfg = cfg.get('animation')
            if anim_cfg and len(snapshots) > 1 and dim >= 2:
                animator = WavePacketAnimator(
                    snapshots, snap_times, grid, potential=potential,
                    theme=anim_cfg.get('theme', 'dark'),
                    fps=anim_cfg.get('fps', 30)
                )
                animator.render(
                    self.output_dir / f"{name}_evolution.{anim_cfg.get('format', 'mp4')}",
                    fmt=anim_cfg.get('format', 'mp4')
                )

        return results
