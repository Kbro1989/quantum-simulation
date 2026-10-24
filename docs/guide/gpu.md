# GPU acceleration

`SplitOperatorSolver` and `ground_state` can run on an NVIDIA GPU through [CuPy](https://cupy.dev):

```bash
pip install -e ".[gpu]"          # CuPy for CUDA 12
```

```python
solver = SplitOperatorSolver(grid, trap, dt=0.005, nonlinearity=500.0, rotation=0.8, backend="cupy")
wf = solver.evolve(wf, 10_000)   # ψ stays on the GPU; one copy back at the end
```

- `backend="auto"` uses the GPU when CuPy and a CUDA device are present and falls back to the CPU otherwise.
- Everything runs in double precision.
- During `evolve`, ψ stays in GPU memory and is copied back only for callbacks and at the end. `step` copies to and from the device on every call.
- Time-dependent potentials are evaluated on the host and copied to the device every sub-step.

## Measured performance

The timings come from Google Colab, NVIDIA Tesla T4, CuPy 14.0.1, Python 3.13. The run is a 3D rotating Gross–Pitaevskii problem (order 2, g = 200, Ω = 0.7), timed over 50 steps:

| Grid | Points | Colab CPU (2 vCPU) ms/step | T4 GPU ms/step | Speed-up | max \|ψ_GPU − ψ_CPU\| |
|---|---|---|---|---|---|
| 64³ | 262,144 | 33.1 | 4.4 | 7.6× | 2.0e-15 |
| 96³ | 884,736 | 132.9 | 12.3 | 10.8× | 2.1e-15 |
| 128³ | 2,097,152 | 363.6 | 29.2 | 12.5× | 2.6e-15 |
| 160³ | 4,096,000 | 721.5 | 78.2 | 9.2× | 1.5e-15 |

The speed-up depends on the CPU used for comparison. Colab's CPU has only 2 vCPUs. An 8-core Apple M1 takes about 87 ms/step at 128³, so the T4 is roughly 3× faster than that machine.

## How the GPU path is tested

1. **In CI** (no GPU on GitHub's runners) the device code runs through a strict emulated device. Its arrays refuse to mix with host arrays, as CuPy's do, so any host/device mix-up raises an error. The results must match NumPy to 10⁻¹³.
2. **On real hardware**, the full suite passes on a Tesla T4: 130 tests, none skipped, including the real-CuPy comparisons.
3. **You can reproduce it** with the validation notebook: [Open in Colab](https://colab.research.google.com/github/rasidi3112/quantum-simulation/blob/main/notebooks/gpu_validation.ipynb). Choose a GPU runtime and *Run all*. `python benchmarks/gpu_split_operator.py` gives the same benchmark on any CUDA machine.
