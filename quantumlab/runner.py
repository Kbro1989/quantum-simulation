"""
Deterministic batch simulation runner driven by YAML configuration.
HPC Cluster Logging & Config Validation Enabled.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import yaml

from quantumlab.core.grid import Grid1D, Grid2D, Grid3D
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D, WaveFunction3D
from quantumlab.core.absorbing_boundary import AbsorbingBoundaryLayer
from quantumlab.potentials import create_potential
from quantumlab.solvers import SplitStep1DSolver, SplitStep2DSolver, SplitStep3DSolver
from quantumlab.solvers.split_step_3d_gpu import SplitStep3DGPU
from quantumlab.observables.expectation import total_energy_expectation, position_expectation
from quantumlab.visualization.animation import WavePacketAnimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("quantumlab.runner")


class SimulationRunner:
    """
    HPC Batch Simulation Runner. No pseudo-RNG. All parameters are
    explicitly declared in config or derived from deterministic hash.
    """
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Simulation config file not found: {config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        self.output_dir = Path(self.config.get("output_dir", "output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_config(self, cfg: Dict[str, Any], sim_name: str) -> None:
        """Validate required configuration parameters."""
        required = ["dimension", "grid", "wavefunction", "potential", "dt", "num_steps"]
        missing = [key for key in required if key not in cfg]
        if missing:
            raise ValueError(f"Simulation '{sim_name}' missing required config keys: {missing}")

    def _build_grid(self, dim: int, cfg: Dict[str, Any]):
        if "N" not in cfg or "L" not in cfg:
            raise KeyError("Grid configuration requires 'N' (points) and 'L' (domain length).")
        N = int(cfg["N"])
        L = float(cfg["L"])
        if dim == 1:
            return Grid1D(N, -L / 2.0, L / 2.0)
        elif dim == 2:
            return Grid2D(N, -L / 2.0, L / 2.0, N, -L / 2.0, L / 2.0)
        elif dim == 3:
            Nx, Ny, Nz = cfg.get("Nx", N), cfg.get("Ny", N), cfg.get("Nz", N)
            Lx, Ly, Lz = cfg.get("Lx", L), cfg.get("Ly", L), cfg.get("Lz", L)
            return Grid3D(Nx, -Lx / 2.0, Lx / 2.0, Ny, -Ly / 2.0, Ly / 2.0, Nz, -Lz / 2.0, Lz / 2.0)
        raise ValueError(f"Unsupported spatial dimension: {dim}")

    def _build_wavefunction(self, grid, cfg: Dict[str, Any]):
        dim = cfg["dimension"]
        wfc = cfg["wavefunction"]
        x0 = float(wfc.get("x0", 0.0))
        sigma = float(wfc.get("sigma", 1.0))

        if dim == 1:
            return WaveFunction1D.gaussian(
                grid, x0=x0, k0=float(wfc.get("k0", 0.0)), sigma=sigma
            )
        elif dim == 2:
            return WaveFunction2D.gaussian(
                grid,
                x0=x0,
                y0=float(wfc.get("y0", 0.0)),
                k0_x=float(wfc.get("k0_x", 0.0)),
                k0_y=float(wfc.get("k0_y", 0.0)),
                sigma_x=sigma,
                sigma_y=float(wfc.get("sigma_y", sigma)),
            )
        elif dim == 3:
            return WaveFunction3D.gaussian(
                grid,
                x0=x0,
                y0=float(wfc.get("y0", 0.0)),
                z0=float(wfc.get("z0", 0.0)),
                k0_x=float(wfc.get("k0_x", 0.0)),
                k0_y=float(wfc.get("k0_y", 0.0)),
                k0_z=float(wfc.get("k0_z", 0.0)),
                sigma_x=sigma,
                sigma_y=float(wfc.get("sigma_y", sigma)),
                sigma_z=float(wfc.get("sigma_z", sigma)),
            )
        raise ValueError(f"Unsupported spatial dimension: {dim}")

    def _build_solver(self, grid, potential, cfg: Dict[str, Any]):
        dim = cfg["dimension"]
        dt = float(cfg["dt"])
        hbar = float(cfg.get("hbar", 1.0))
        m = float(cfg.get("m", 1.0))
        gpu = bool(cfg.get("gpu", False))

        if dim == 1:
            return SplitStep1DSolver(grid, potential, dt, hbar, m)
        elif dim == 2:
            return SplitStep2DSolver(grid, potential, dt, hbar, m)
        elif dim == 3:
            if gpu:
                return SplitStep3DGPU(grid, potential, dt, hbar, m)
            return SplitStep3DSolver(grid, potential, dt, hbar, m)
        raise ValueError(f"Unsupported spatial dimension: {dim}")

    def run(self, sim_name: Optional[str] = None) -> Dict[str, Any]:
        sims = self.config.get("simulations", {})
        if not sims:
            raise ValueError("No simulations defined in YAML configuration.")

        if sim_name:
            if sim_name not in sims:
                raise KeyError(f"Simulation name '{sim_name}' not found in config.")
            sims = {sim_name: sims[sim_name]}

        results = {}
        for name, cfg in sims.items():
            self._validate_config(cfg, name)
            logger.info("Starting simulation: %s", name)

            dim = cfg["dimension"]
            grid = self._build_grid(dim, cfg["grid"])
            potential = create_potential(cfg["potential"]["type"], **cfg["potential"]["params"])
            wf = self._build_wavefunction(grid, cfg)
            solver = self._build_solver(grid, potential, cfg)

            abl_cfg = cfg.get("absorbing_boundary")
            abl = None
            if abl_cfg:
                abl = AbsorbingBoundaryLayer(grid, float(abl_cfg["width"]), int(abl_cfg.get("order", 3)))

            num_steps = int(cfg["num_steps"])
            snapshot_interval = int(cfg.get("snapshot_interval", 0))
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
                    snap_times.append(step * float(cfg["dt"]))

                if step % int(cfg.get("log_interval", 100)) == 0:
                    E = total_energy_expectation(wf_current, potential)
                    x = position_expectation(wf_current)
                    logger.info("  Step %5d/%d  E=%.6f  <x>=%+.3f  [%s]", step, num_steps, E, x, solver.backend)

            results[name] = {
                "final_wavefunction": wf_current,
                "snapshots": snapshots,
                "snap_times": snap_times,
                "config": cfg,
            }

            if cfg.get("save_state"):
                np.savez(self.output_dir / f"{name}_state.npz", psi=wf_current.psi, params=cfg)

            # Post-process: animation export
            anim_cfg = cfg.get("animation")
            if anim_cfg and len(snapshots) > 1 and dim >= 2:
                animator = WavePacketAnimator(
                    snapshots,
                    snap_times,
                    grid,
                    potential=potential,
                    theme=anim_cfg.get("theme", "dark"),
                    fps=int(anim_cfg.get("fps", 30)),
                )
                animator.render(
                    self.output_dir / f"{name}_evolution.{anim_cfg.get('format', 'mp4')}",
                    fmt=anim_cfg.get("format", "mp4"),
                )

        return results
