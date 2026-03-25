# Julia Backend Setup

## Requirements

- Python >= 3.10 with pyUDE[julia] installed
- Julia >= 1.10
- Internet access for the first run (Julia downloads UniversalDiffEq.jl and its dependencies)

---

## Step 1: Install Julia

Download Julia from [julialang.org/downloads](https://julialang.org/downloads). The
recommended way is via **juliaup** (the Julia version manager):

**Windows:**
```powershell
winget install julia -s msstore
```

**macOS / Linux:**
```bash
curl -fsSL https://install.julialang.org | sh
```

Verify:
```bash
julia --version
# julia version 1.11.x
```

## Step 2: Install the Python package with Julia extras

```bash
pip install "pyUDE[julia]"
```

This installs `juliacall`, the Python↔Julia bridge package backed by `PythonCall.jl`.

## Step 3: First run — Julia package resolution

The first time you construct a Julia model, pyUDE activates the bundled Julia environment
(`julia/Project.toml`) and downloads UniversalDiffEq.jl and its dependencies. This takes
2–5 minutes and requires internet access.

```python
from pyUDE import JuliaNODE
import pandas as pd
import numpy as np

# Generate minimal test data
t = np.linspace(0, 5, 30)
data = pd.DataFrame({"time": t, "x": np.exp(-0.5 * t)})

model = JuliaNODE(data)
# → Julia starts here, packages install on first run
model.train(epochs=10, verbose=True)
print("Julia backend is working.")
```

Subsequent runs skip the download step (packages are cached in the Julia depot).

## Step 4: Verify JIT compilation

Julia compiles functions on first call (JIT). The first `model.train()` call is always slower
than subsequent calls. This is normal:

```
First call:    ~60s  (download + JIT compile)
Second call:   ~5s   (JIT only)
Third+ calls:  ~1s   (fully compiled)
```

In a long analysis session, the JIT overhead is paid once per process.

---

## Troubleshooting

### `ImportError: juliacall is required`

You installed `pyUDE` without the Julia extra:
```bash
pip install "pyUDE[julia]"
```

### `RuntimeError: Julia environment directory not found`

The `julia/` directory with `Project.toml` is missing from the package installation. If
installing from source (editable install), ensure you cloned the full repository:
```bash
git clone https://github.com/19fitzna/pyUDE
cd pyUDE
pip install -e ".[julia]"
```

### `Pkg.instantiate()` fails / package not found

UniversalDiffEq.jl is a newer package and may require a specific Julia registry version.
From a Julia REPL:
```julia
using Pkg
Pkg.Registry.update()
```
Then retry from Python.

### Julia crashes with `SIGKILL` on macOS

This is a known issue with Apple Silicon (M-series) and some Julia versions. Ensure you have
Julia 1.10+ and use the ARM64 native binary (not Rosetta):
```bash
julia --version   # should show aarch64, not x86_64
```

### `PythonCall` not found in Julia

`juliacall` should install `PythonCall.jl` automatically. If it doesn't:
```bash
python -c "import juliacall; juliacall.Main.seval('using PythonCall')"
```

### Slow training with `JuliaCustomDerivatives`

When `known_dynamics` is a Python callable, Julia calls it back via PythonCall.jl on every
ODE RHS evaluation. This crosses the Julia↔Python boundary millions of times per training
run, which can be slow for complex functions.

**Mitigation:**
- Keep `known_dynamics` as simple as possible (avoid loops, prefer vectorised operations)
- Use `JuliaNODE` instead if the known structure is minimal
- Consider implementing the known dynamics in Julia directly (advanced usage)

---

## Julia Solver Names

Specify the solver at construction time via the `solver` parameter:

```python
# Non-stiff (default)
model = JuliaNODE(data, solver="Tsit5")

# High-accuracy non-stiff
model = JuliaNODE(data, solver="Vern9")

# Stiff systems
model = JuliaNODE(data, solver="Rodas5")
model = JuliaNODE(data, solver="QNDF")

# Automatic stiffness detection
model = JuliaNODE(data, solver="AutoTsit5(Rosenbrock23())")
```

See [Choosing a Backend — Julia Solver Reference](../choosing-backend.md#julia-solver-reference)
for the full table.

---

## How the Bridge Works

When you call `model.train()` on a Julia model:

1. `_env.py` starts `juliacall` and activates `julia/Project.toml`
2. `julia/pyude_bridge.jl` is loaded, defining helper functions
3. `_convert.py` converts the pandas DataFrame to Julia `Matrix{Float64}` arrays
4. `_env.py.get_julia()` returns handles to `juliacall.Main` and `UniversalDiffEq`
5. The Julia `UniversalDiffEq.NODE` (or `CustomDerivatives`) struct is constructed
6. `UniversalDiffEq.train_b` runs the optimisation loop in Julia
7. For `JuliaCustomDerivatives`, `pyude_bridge.wrap_python_dynamics()` wraps the Python
   `known_dynamics` callable so Julia's ODE solver can call it via `PythonCall.pyconvert`

On `model.forecast()`, `UniversalDiffEq.forecast` integrates forward and the result is
converted back to a pandas DataFrame by `_convert.py`.
