# Choosing a Backend: PyTorch vs Julia

pyUDE provides two backends that expose the same API:

| | PyTorch backend | Julia backend |
|---|---|---|
| **Classes** | `NODE`, `CustomDerivatives`, `CustomDifferences` | `JuliaNODE`, `JuliaCustomDerivatives`, `JuliaCustomDifferences` |
| **ODE solver** | torchdiffeq (dopri5 / rk4) | DifferentialEquations.jl (Tsit5, Rodas5, …) |
| **Neural network** | `torch.nn.Module` (any architecture) | Lux.jl MLP (auto-constructed) |
| **Install** | `pip install pyUDE` | `pip install "pyUDE[julia]"` + Julia |
| **Stiff solvers** | Limited | First-class |

---

## Use the PyTorch backend when…

### You want zero non-Python dependencies

The PyTorch backend installs with a single `pip install pyUDE`. No Julia runtime, no
package resolution, no JIT compilation on first run. This matters for:
- Containerised deployments (Docker, Kubernetes)
- Shared compute environments where you can't install Julia
- Distributing your work to colleagues who aren't Julia users

### You have an existing PyTorch model

`NODE` and `CustomDerivatives` accept any `torch.nn.Module` as the `network` argument. You can
drop in a pre-trained encoder, a convolutional network that processes spatial state, or a
transformer for sequence-valued dynamics:

```python
import torch.nn as nn

# Use a custom architecture as the unknown dynamics
class MyNetwork(nn.Module):
    def forward(self, u):
        ...

model = NODE(data, network=MyNetwork())
```

The Julia backend only supports the auto-constructed Lux.jl MLP.

### GPU training is important

PyTorch's CUDA ecosystem is more mature:
- `torch.compile` for graph-mode acceleration
- Automatic Mixed Precision (AMP) via `torch.cuda.amp`
- Multi-GPU via `DistributedDataParallel`
- Integration with PyTorch Lightning

The Julia backend runs on CPU only in the current implementation.

### You need PyTorch ecosystem integrations

- **Weights & Biases / MLflow** — log training metrics from the standard training loop
- **TorchScript / ONNX** — export the trained ODE function for deployment
- **Hugging Face** — use transformer embeddings as state representations
- **PyTorch Geometric** — graph neural networks as the unknown dynamics

### The system is non-stiff with short time horizons

For clean data and time series of moderate length (a few hundred time steps, no fast-slow
separation), torchdiffeq's dopri5 works well. The Julia backend's overhead (startup time,
bridge calls) is not justified.

### You want faster iteration during development

PyTorch error messages, `pdb`, `torch.autograd.set_detect_anomaly(True)`, and the
`torchinfo` summary tool all work directly on the ODE function. Julia stack traces are harder
to parse from Python.

---

## Use the Julia backend when…

### Performance is critical

DifferentialEquations.jl is the fastest ODE solver suite available in any language. Benchmarks
from the SciML team show 2–10× speedups over scipy and torchdiffeq on typical scientific
problems. The gains are largest when:
- The ODE has many state variables (>10)
- High accuracy is required (tight tolerances)
- The integration horizon is long

For a 2-state Lotka-Volterra system at default tolerances, expect roughly 3–5× speedup.

### The system is stiff

A *stiff* system has components evolving on very different timescales — common in:
- Biochemical reaction networks (fast binding/unbinding + slow gene expression)
- Pharmacokinetic/pharmacodynamic models (rapid absorption, slow elimination)
- Chemical engineering (fast reactions, slow diffusion)
- Electrical circuits

Stiff systems require **implicit solvers** that solve a nonlinear system at each step. The
Julia backend exposes these via the `solver` parameter:

```python
# Stiff ODE — use Rosenbrock solver
model = JuliaNODE(data, solver="Rodas5")
model = JuliaCustomDerivatives(data, f, params, solver="Rodas5")

# Other stiff options
solver="QNDF"       # Quasi-constant step-size NDF method (like MATLAB's ode15s)
solver="Rosenbrock23"  # Low-order, good for mildly stiff
```

With torchdiffeq's explicit solvers (dopri5, rk4), stiff systems either diverge or require
extremely small step sizes, making training impractically slow.

### You need long-horizon forecasts with high accuracy

Julia's solvers offer:
- **Dense output** — evaluate the solution at any time point, not just solver steps
- **Event handling** — callbacks when state crosses a threshold (e.g. bifurcations)
- **Tight error control** — absolute and relative tolerances as low as machine precision
- **Extrapolation methods** (Vern9, Vern8) — extremely high-order for smooth problems

### You need UniversalDiffEq.jl-specific features

The Julia backend gives you access to algorithms only available in UniversalDiffEq.jl:
- **Multi-time-series training** (`MultiCustomDerivatives`) — fit shared dynamics to multiple
  independent datasets simultaneously
- **Bayesian UDEs** — posterior inference over NN weights and mechanistic parameters
- **State-space estimation** — handle partially observed systems (not all states measured)
- **Derivative-matching with spline smoothing** — more robust than finite differences

### Your neural network is a small MLP

For the typical scientific ML use case (2–10 state variables, MLP with 1–3 hidden layers),
Lux.jl networks outperform PyTorch for small batches because:
- Julia compiles a single fused kernel for the whole forward pass
- No Python interpreter overhead per RHS call
- Better CPU SIMD utilisation for float64 arrays

The advantage disappears for large networks (>10k parameters) or when GPU is available.

### You already work in the SciML ecosystem

If your workflow already uses Julia tools (Turing.jl for Bayesian inference, Agents.jl for
ABM, DifferentialEquations.jl directly), the Julia backend lets you pass trained UDE models
back to Julia tools via the `_jl_model` attribute.

---

## Side-by-Side Comparison

```python
# --- PyTorch backend ---
import pyUDE as ude
import torch

def known(u, p, t):                         # u, p, t are PyTorch tensors
    return torch.stack([p["alpha"] * u[0],
                        -p["delta"] * u[1]])

model = ude.CustomDerivatives(
    data,
    known_dynamics=known,
    init_params={"alpha": 1.0, "delta": 1.5},
    solver="dopri5",                        # torchdiffeq solver
)
model.train(loss="simulation", epochs=1000)
```

```python
# --- Julia backend ---
from pyUDE import JuliaCustomDerivatives

def known(u, p, t):                         # u is list[float], p is dict, t is float
    return [p["alpha"] * u[0],
            -p["delta"] * u[1]]

model = JuliaCustomDerivatives(
    data,
    known_dynamics=known,
    init_params={"alpha": 1.0, "delta": 1.5},
    solver="Tsit5",                         # DifferentialEquations.jl solver
)
model.train(epochs=1000)
```

The API is identical except for:
1. The `known_dynamics` function signature (tensors vs plain lists)
2. The `solver` name (torchdiffeq names vs Julia solver names)
3. The `loss` parameter (Julia backend uses `"trajectory"` internally; not exposed)

---

## Julia Solver Reference

| Solver | Type | Use case |
|--------|------|----------|
| `"Tsit5"` | Explicit, 5th-order | **Default.** Non-stiff, general purpose |
| `"Vern9"` | Explicit, 9th-order | High-accuracy, smooth non-stiff problems |
| `"RK4"` | Explicit, 4th-order fixed-step | Debugging, reproducibility |
| `"Rosenbrock23"` | Implicit, 2nd-order | Mildly stiff |
| `"Rodas5"` | Implicit, 5th-order | **Stiff problems.** Chemistry, biochemistry |
| `"QNDF"` | Implicit, variable order | Stiff, matches MATLAB `ode15s` |
| `"AutoTsit5(Rosenbrock23())"` | Auto-switching | Unknown stiffness |

---

## Decision Flowchart

```
Do you have an existing PyTorch model (CNN, transformer, etc.)?
└── Yes → PyTorch backend

Is Julia installed, or are you willing to install it?
└── No → PyTorch backend

Is the system stiff (fast-slow timescales, chemistry, PK/PD)?
└── Yes → Julia backend (solver="Rodas5" or "QNDF")

Do you need GPU training or PyTorch ecosystem tools?
└── Yes → PyTorch backend

Do you need multi-series training or Bayesian UDEs?
└── Yes → Julia backend

Is performance (training speed) the top priority?
└── Yes → Julia backend

Otherwise → PyTorch backend (simpler, no extra setup)
```
