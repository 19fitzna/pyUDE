# CustomDerivatives / JuliaCustomDerivatives

Hybrid UDE: user-supplied mechanistic dynamics plus a neural network for unknown residual
terms.

```
du/dt = f_known(u, p, t)  +  NN(u)
```

Both the neural network weights and the mechanistic parameters `p` are optimised jointly
during training.

---

## `pyUDE.CustomDerivatives`

**PyTorch backend.**

### Constructor

```python
CustomDerivatives(
    data: pd.DataFrame,
    known_dynamics: Callable,
    init_params: dict,
    network: torch.nn.Module | None = None,
    hidden_layers: int = 2,
    hidden_units: int = 32,
    time_column: str = "time",
    device: str = "cpu",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Time series data. |
| `known_dynamics` | `Callable` | — | `f(u, p, t) -> du`. See [signature details](#known_dynamics-signature-pytorch) below. Validated at construction time — an error is raised immediately if the function returns the wrong shape or raises an exception. |
| `init_params` | `dict` | — | Initial values for the mechanistic parameters. Keys are parameter names (strings); values are floats or tensors. All parameters are made trainable. |
| `network` | `nn.Module \| None` | `None` | Neural network for the unknown residual. Defaults to a small MLP. |
| `hidden_layers` | `int` | `2` | Hidden layers in the default MLP. |
| `hidden_units` | `int` | `32` | Units per hidden layer. |
| `time_column` | `str` | `"time"` | Name of the time column. |
| `device` | `str` | `"cpu"` | PyTorch device: `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`. |

### `known_dynamics` Signature (PyTorch)

```python
def f(u: torch.Tensor, p: dict, t: torch.Tensor) -> torch.Tensor:
    ...
```

| Argument | Shape / type | Description |
|----------|-------------|-------------|
| `u` | `Tensor(n_states,)`, float64 | Current state vector |
| `p` | `dict[str, Tensor]` | Trainable mechanistic parameters (scalar tensors) |
| `t` | scalar `Tensor`, float64 | Current time |
| **returns** | `Tensor(n_states,)` | Rate of change `du/dt` for the known part |

The neural network's contribution is added automatically — your function should return **only
the known terms**, not the sum.

**Example:**

```python
import torch

def lotka_volterra_known(u, p, t):
    prey, pred = u[0], u[1]
    return torch.stack([
        p["alpha"] * prey,          # prey growth (known)
        -p["delta"] * pred,         # predator death (known)
        # predation terms are unknown — learned by the NN
    ])
```

### Methods

#### `train`

```python
model.train(
    loss: str = "simulation",
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    epochs: int = 500,
    log_interval: int = 50,
    verbose: bool = True,
    solver: str = "dopri5",
    patience: int | None = None,
    max_grad_norm: float = 10.0,
    weight_decay: float | None = None,
) -> CustomDerivatives
```

Same parameters as `NODE.train()`. See [NODE API](node.md#train) for full descriptions.
Calling `train()` a second time continues from the current weights — mechanistic parameters
and network weights are preserved.

#### `forecast`

```python
model.forecast(
    steps: int,
    dt: float | None = None,
    initial_state: torch.Tensor | None = None,
) -> pd.DataFrame
```

Same as `NODE.forecast()`. See [NODE API](node.md#forecast).

#### `get_right_hand_side`

```python
model.get_right_hand_side() -> Callable
```

Returns the **full** right-hand side `f_known + NN` as a numpy-compatible `f(u, t) -> du`.

#### `get_params`

```python
model.get_params() -> dict
```

Return the current values of the learned mechanistic parameters.

```python
model.train(epochs=1000)
params = model.get_params()
# {"alpha": 0.98, "delta": 1.51}
```

Returns a `dict[str, float]`. All values are plain Python floats.

### Properties

Same as `NODE`: `data`, `is_trained`, `n_states`, `state_columns`, `time_column`.

---

## `pyUDE.JuliaCustomDerivatives`

**Julia backend.** The key distinction from `CustomDerivatives`:

- `known_dynamics` receives **plain Python lists and floats**, not PyTorch tensors, because the
  function is called from Julia via PythonCall.jl.
- The `solver` parameter selects a Julia solver from DifferentialEquations.jl.

Requires `pip install "pyUDE[julia]"`. See [Julia Backend Setup](../guides/julia-setup.md).

### Constructor

```python
JuliaCustomDerivatives(
    data: pd.DataFrame,
    known_dynamics: Callable,
    init_params: dict,
    hidden_layers: int = 2,
    hidden_units: int = 32,
    time_column: str = "time",
    solver: str = "Tsit5",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Time series data. |
| `known_dynamics` | `Callable` | — | `f(u, p, t) -> du`. See [signature details](#known_dynamics-signature-julia) below. |
| `init_params` | `dict` | — | Initial mechanistic parameter values (str → float). |
| `hidden_layers` | `int` | `2` | |
| `hidden_units` | `int` | `32` | |
| `time_column` | `str` | `"time"` | |
| `solver` | `str` | `"Tsit5"` | Julia ODE solver. See [solver table](../choosing-backend.md#julia-solver-reference). |

### `known_dynamics` Signature (Julia backend)

```python
def f(u: list, p: dict, t: float) -> list:
    ...
```

| Argument | Type | Description |
|----------|------|-------------|
| `u` | `list[float]` | Current state vector |
| `p` | `dict[str, float]` | Current mechanistic parameter values |
| `t` | `float` | Current time |
| **returns** | `list[float]` | Known part of `du/dt` |

> **Important**: do not use PyTorch operations inside this function. It is called from Julia
> during ODE integration, so only plain Python/numpy operations are valid.

**Example:**

```python
def lv_known(u, p, t):
    prey, pred = u[0], u[1]
    return [
        p["alpha"] * prey,
        -p["delta"] * pred,
    ]
```

### Methods

#### `train`

```python
model.train(
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    epochs: int = 500,
    verbose: bool = True,
) -> JuliaCustomDerivatives
```

#### `forecast`

```python
model.forecast(steps: int, dt: float | None = None) -> pd.DataFrame
```

#### `get_right_hand_side`

```python
model.get_right_hand_side() -> Callable
```

#### `get_params`

```python
model.get_params() -> dict
```

Returns `dict[str, float]` of learned mechanistic parameter values.

### Properties

Same as `NODE`: `data`, `is_trained`, `n_states`, `state_columns`, `time_column`.

---

## Examples

### Lotka-Volterra with PyTorch backend

```python
import torch
import pandas as pd
import pyUDE as ude

data = pd.read_csv("lotka_volterra.csv")   # columns: time, prey, predator

def lv_known(u, p, t):
    """Known structure: exponential prey growth + predator death."""
    prey, pred = u[0], u[1]
    return torch.stack([
        p["alpha"] * prey,
        -p["delta"] * pred,
    ])

model = ude.CustomDerivatives(
    data,
    known_dynamics=lv_known,
    init_params={"alpha": 0.8, "delta": 1.2},  # will be refined during training
    hidden_units=32,
)
model.train(epochs=2000, learning_rate=3e-3)

# Inspect recovered parameters
print(model.get_params())   # {"alpha": ≈1.0, "delta": ≈1.5}

# Forecast
forecast = model.forecast(steps=50)
```

### Same example with Julia backend

```python
from pyUDE import JuliaCustomDerivatives

def lv_known(u, p, t):                        # note: plain lists, not tensors
    prey, pred = u[0], u[1]
    return [p["alpha"] * prey, -p["delta"] * pred]

model = JuliaCustomDerivatives(
    data,
    known_dynamics=lv_known,
    init_params={"alpha": 0.8, "delta": 1.2},
    solver="Tsit5",
)
model.train(epochs=2000)
print(model.get_params())
```

### Using `get_right_hand_side` with scipy

```python
rhs = model.get_right_hand_side()

from scipy.integrate import odeint
import numpy as np

u0 = data[["prey", "predator"]].iloc[0].values
t = np.linspace(0, 20, 500)
sol = odeint(rhs, y0=u0, t=t)
```
