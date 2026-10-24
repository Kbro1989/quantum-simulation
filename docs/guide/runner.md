# YAML batch runner

`SimulationRunner` runs one or more simulations described in a YAML file. This is useful for parameter sweeps and batch jobs:

```python
from quantumlab.runner import SimulationRunner

results = SimulationRunner('configs/benchmark.yaml').run()          # all simulations
results = SimulationRunner('configs/benchmark.yaml').run('my_run')  # one by name
```

## Configuration

```yaml
output_dir: output/batch

simulations:
  rotating_bec:
    dimension: 2                 # 1, 2 or 3
    grid: {N: 128, L: 20.0}      # N points over [-L/2, L/2] per axis (3D also accepts Nx/Ny/Nz, Lx/Ly/Lz)
    wavefunction: {x0: 0.0, sigma: 2.0, k0_x: 0.0}
    potential:
      type: harmonic_trap        # any name from the potential registry
      params: {omegas: 1.0}
    dt: 0.005
    num_steps: 2000
    solver: split_operator       # omit for the legacy split-step solvers
    order: 2
    nonlinearity: 500.0
    rotation: 0.8
    damping: 0.05
    backend: auto                # numpy | cupy | auto
    absorbing_boundary: {width: 3.0, order: 3}
    snapshot_interval: 100
    log_interval: 500
    save_state: true             # writes <name>_state.npz
    animation: {format: mp4, fps: 24, theme: dark}   # 2D/3D only
```

`dimension`, `grid`, `wavefunction`, `potential`, `dt` and `num_steps` are required. Without `solver: split_operator` the legacy solvers are used, and in 3D `gpu: true` selects `SplitStep3DGPU`.

Each result holds the final wavefunction, the snapshots and their times, and the configuration.
