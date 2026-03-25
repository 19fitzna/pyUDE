# CustomDifferences / JuliaCustomDifferences

Discrete-time hybrid UDE: user-supplied known state-transition map plus a neural network for
unknown residual terms.

```
u[n+1] = g_known(u[n], p, n)  +  NN(u[n])
```

Use this when your data is naturally discrete (population censuses, daily records, financial
time series) or when you are modelling a map / difference equation directly.

---

## `pyUDE.CustomDifferences`

**PyTorch backend.**

### Constructor

```python
CustomDifferences(
    data: pd.DataFrame,
    known_map: Callable,
    init_params: dict,
    network: torch.nn.Module | None = None,
    hidden_layers: int = 2,
    hidden_units: int = 32,
    time_column: str = "time",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Discrete-time series data. |
| `known_map` | `Callable` | — | `f(u, p, t) -> u_next`. See [signature details](#known_map-signature-pytorch) below. |
| `init_params` | `dict` | — | Initial mechanistic parameter values (str → float). Made trainable. |
| `network` | `nn.Module \| None` | `None` | Neural network for unknown residual. |
| `hidden_layers` | `int` | `2` | |
| `hidden_units` | `int` | `32` | |
| `time_column` | `str` | `"time"` | Name of the time column. |

### `known_map` Signature (PyTorch)

```python
def g(u: torch.Tensor, p: dict, t: torch.Tensor) -> torch.Tensor:
    ...
```

| Argument | Shape / type | Description |
|----------|-------------|-------------|
| `u` | `Tensor(n_states,)`, float64 | Current state |
| `p` | `dict[str, Tensor]` | Trainable mechanistic parameters |
| `t` | scalar `Tensor` | Current discrete time index |
| **returns** | `Tensor(n_states,)` | Known next-state contribution |

**Example:**

```python
import torch

def logistic_known(u, p, t):
    return p["r"] * u * (1 - u)   # logistic map known structure
```

### Methods

#### `train`

```python
model.train(
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    epochs: int = 500,
    log_interval: int = 50,
    verbose: bool = True,
) -> CustomDifferences
```

Minimises one-step-ahead MSE: for each step `n`, predicts `u[n+1]` and compares to the
observed `u[n+1]`. No ODE integration is involved, so this is faster per epoch than the
continuous-time simulation loss.

Note: `loss` and `solver` parameters are not available for `CustomDifferences` — the training
procedure is fixed as one-step-ahead MSE.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | `str` | `"adam"` | `"adam"` or `"sgd"`. |
| `learning_rate` | `float` | `1e-3` | |
| `epochs` | `int` | `500` | |
| `log_interval` | `int` | `50` | Print every N epochs when `verbose=True`. |
| `verbose` | `bool` | `True` | |

#### `forecast`

```python
model.forecast(
    steps: int,
    initial_state: torch.Tensor | None = None,
) -> pd.DataFrame
```

Step the learned map forward `steps` times from the last observed state.

#### `get_params`

```python
model.get_params() -> dict
```

Return learned mechanistic parameter values as `dict[str, float]`.

### Properties

Same as `NODE`: `data`, `is_trained`, `n_states`, `state_columns`, `time_column`.

---

## `pyUDE.JuliaCustomDifferences`

**Julia backend.** Same model, delegating to UniversalDiffEq.jl's discrete-time training.

### Constructor

```python
JuliaCustomDifferences(
    data: pd.DataFrame,
    known_map: Callable,
    init_params: dict,
    hidden_layers: int = 2,
    hidden_units: int = 32,
    time_column: str = "time",
)
```

Note: `solver` is not a parameter for `JuliaCustomDifferences` (discrete maps don't use ODE
solvers).

### `known_map` Signature (Julia backend)

```python
def g(u: list, p: dict, t: float) -> list:
    ...
```

Same constraint as `JuliaCustomDerivatives`: use plain lists and floats, not PyTorch tensors.

**Example:**

```python
def logistic_known(u, p, t):
    return [p["r"] * u[0] * (1 - u[0])]
```

### Methods

#### `train`

```python
model.train(
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    epochs: int = 500,
    verbose: bool = True,
) -> JuliaCustomDifferences
```

#### `forecast`

```python
model.forecast(steps: int, dt: float | None = None) -> pd.DataFrame
```

#### `get_params`

```python
model.get_params() -> dict
```

### Properties

Same as `NODE`: `data`, `is_trained`, `n_states`, `state_columns`, `time_column`.

---

## Examples

### Logistic map with PyTorch backend

```python
import torch
import pandas as pd
import pyUDE as ude
import numpy as np

# Generate logistic map data
x, xs = 0.5, []
for _ in range(80):
    xs.append(x)
    x = 3.6 * x * (1 - x)
data = pd.DataFrame({"time": np.arange(80, dtype=float), "x": xs})

def logistic_known(u, p, t):
    return p["r"] * u * (1 - u)

model = ude.CustomDifferences(
    data,
    known_map=logistic_known,
    init_params={"r": 3.0},   # true value is 3.6
    hidden_units=16,
)
model.train(epochs=800)
print(model.get_params())   # {"r": ≈3.6}

future = model.forecast(steps=20)
```

### Population census data

```python
# Discrete annual population counts
data = pd.DataFrame({
    "time": range(30),
    "population": census_counts,
})

def beverton_holt(u, p, t):
    """Known Beverton-Holt structure."""
    return p["r"] * u / (1 + u / p["K"])

model = ude.CustomDifferences(
    data,
    known_map=beverton_holt,
    init_params={"r": 1.5, "K": 100.0},
    hidden_units=8,
)
model.train(epochs=500)
```
