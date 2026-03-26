# pyUDE Documentation

**Universal Differential Equations in Python**

pyUDE is a Python library for building *Universal Differential Equations* (UDEs) — dynamical
systems that combine known mechanistic equations with neural networks to learn unknown terms
directly from data.

---

## What is a UDE?

A UDE is a differential equation where some terms are prescribed by domain knowledge and others
are approximated by a neural network:

```
du/dt = f_known(u, p, t)  +  NN(u)
         ↑ mechanistic        ↑ learned
```

This lets you:
- Encode what you *know* about a system (conservation laws, growth rates, decay)
- Learn what you *don't know* from data (nonlinear interactions, unknown forcing)
- Recover interpretable mechanistic parameters alongside the neural network

---

## Documentation Contents

### Getting Started
- [Installation](installation.md) — pip, optional extras, Julia setup
- [Quickstart](guides/quickstart.md) — first model in five minutes
- [Core Concepts](concepts.md) — UDEs, NODEs, adjoint methods explained

### How-To Guides
- [Choosing a Backend](choosing-backend.md) — PyTorch vs Julia: when to use each
- [Writing known_dynamics Functions](guides/known-dynamics.md) — tips and patterns
- [Julia Backend Setup](guides/julia-setup.md) — installing Julia and UniversalDiffEq.jl
- [GPU Acceleration](installation.md#gpu-support) — CUDA and MPS setup

### API Reference
- [NODE / JuliaNODE](api/node.md) — pure neural ODE
- [CustomDerivatives / JuliaCustomDerivatives](api/custom_derivatives.md) — hybrid UDE
- [CustomDifferences / JuliaCustomDifferences](api/custom_differences.md) — discrete-time UDE
- [Analysis utilities](api/analysis.md) — `forecast()`, `get_right_hand_side()`

---

## Quick Example

```python
import pandas as pd
import pyUDE as ude

data = pd.read_csv("my_timeseries.csv")   # must have a "time" column

# Pure neural ODE — learns all dynamics from data
model = ude.NODE(data)
model.train(epochs=500)
predictions = model.forecast(steps=50)
```

```python
import torch

# Hybrid UDE — known prey growth + learned predation
def known_growth(u, p, t):
    prey, pred = u[0], u[1]
    return torch.stack([p["alpha"] * prey, -p["delta"] * pred])

model = ude.CustomDerivatives(
    data,
    known_dynamics=known_growth,
    init_params={"alpha": 1.0, "delta": 1.5},
)
model.train(epochs=1000)
print(model.get_params())   # {"alpha": ..., "delta": ...}
```

---

## Package Overview

| Class | Backend | Use case |
|-------|---------|----------|
| `NODE` | PyTorch | Learn all dynamics from data |
| `CustomDerivatives` | PyTorch | Hybrid: known terms + neural network |
| `CustomDifferences` | PyTorch | Discrete-time difference equations |
| `JuliaNODE` | Julia | NODE with faster/stiff ODE solvers |
| `JuliaCustomDerivatives` | Julia | Hybrid UDE with Julia solvers |
| `JuliaCustomDifferences` | Julia | Discrete-time with Julia backend |

---

## License

MIT © Nathan Fitzpatrick
