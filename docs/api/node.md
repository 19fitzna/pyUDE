# NODE / JuliaNODE

Neural ODE — learns the full system dynamics `du/dt = NN(u)` from data.
No prior knowledge of the system is required.

---

## `pyUDE.NODE`

**PyTorch backend.** The neural network is a `torch.nn.Module`; the ODE is integrated with
`torchdiffeq`.

### Constructor

```python
NODE(
    data: pd.DataFrame,
    network: torch.nn.Module | None = None,
    hidden_layers: int = 2,
    hidden_units: int = 32,
    time_column: str = "time",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Time series data. Must contain a time column and at least one state column. Rows are time steps. |
| `network` | `nn.Module \| None` | `None` | Custom PyTorch network. Must accept a 1-D tensor of shape `(n_states,)` and return a tensor of the same shape. If `None`, a default MLP is built from `hidden_layers` and `hidden_units`. |
| `hidden_layers` | `int` | `2` | Number of hidden layers in the default MLP. Ignored if `network` is provided. |
| `hidden_units` | `int` | `32` | Units per hidden layer in the default MLP. |
| `time_column` | `str` | `"time"` | Name of the time column in `data`. |

The default MLP architecture is:
`[n_states → hidden_units → … → hidden_units → n_states]` with `tanh` activations between
layers and no activation on the output.

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
) -> NODE
```

Fit the model to the training data. Returns `self` for chaining.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss` | `str` | `"simulation"` | `"simulation"`: integrate ODE forward and compare to observations via MSE. `"derivative_matching"`: compare predicted derivatives to finite-difference estimates (faster but noisier). |
| `optimizer` | `str` | `"adam"` | `"adam"` or `"sgd"`. |
| `learning_rate` | `float` | `1e-3` | Optimizer learning rate. |
| `epochs` | `int` | `500` | Number of gradient steps. |
| `log_interval` | `int` | `50` | Print loss every this many epochs (when `verbose=True`). |
| `verbose` | `bool` | `True` | Print loss during training. |
| `solver` | `str` | `"dopri5"` | torchdiffeq solver name. Options: `"dopri5"` (adaptive 4/5 RK), `"rk4"` (fixed 4th-order), `"euler"`, `"midpoint"`, `"rk4"`, `"explicit_adams"`. |

#### `forecast`

```python
model.forecast(
    steps: int,
    dt: float | None = None,
    initial_state: torch.Tensor | None = None,
) -> pd.DataFrame
```

Integrate the learned dynamics forward from the last observed state.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | `int` | — | Number of future time steps to predict. |
| `dt` | `float \| None` | `None` | Step size. Defaults to the median interval in the training data. |
| `initial_state` | `Tensor \| None` | `None` | 1-D tensor of shape `(n_states,)`. Defaults to the last row of the training data. |

Returns a `pd.DataFrame` with columns `[time_column, *state_columns]`. The initial state row
is **not** included — only the `steps` forecasted rows.

#### `get_right_hand_side`

```python
model.get_right_hand_side() -> Callable
```

Return the learned dynamics as a numpy-compatible function `f(u, t) -> du`.

The returned callable accepts numpy arrays (or lists) and returns a numpy array, making it
compatible with `scipy.integrate.odeint` and similar tools.

```python
rhs = model.get_right_hand_side()

# Use with scipy
from scipy.integrate import odeint
sol = odeint(rhs, y0=u0, t=t_span)

# Evaluate directly
du = rhs(np.array([1.0, 0.5]), t=0.0)   # returns np.ndarray
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `data` | `pd.DataFrame` | The training data passed at construction. |
| `is_trained` | `bool` | `True` after `train()` completes. |
| `n_states` | `int` | Number of state variables. |
| `state_columns` | `list[str]` | Names of the state columns. |
| `time_column` | `str` | Name of the time column. |

---

## `pyUDE.JuliaNODE`

**Julia backend.** The neural network is a Lux.jl MLP auto-constructed from `hidden_layers`
and `hidden_units`; the ODE is solved by DifferentialEquations.jl.

Requires `pip install "pyUDE[julia]"` and a Julia installation. See
[Julia Backend Setup](../guides/julia-setup.md).

Julia starts lazily on the first call — constructing a `JuliaNODE` does not start Julia.
Training the model does.

### Constructor

```python
JuliaNODE(
    data: pd.DataFrame,
    hidden_layers: int = 2,
    hidden_units: int = 32,
    time_column: str = "time",
    solver: str = "Tsit5",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Same requirements as `NODE`. |
| `hidden_layers` | `int` | `2` | Hidden layers in the Lux.jl MLP. |
| `hidden_units` | `int` | `32` | Units per hidden layer. |
| `time_column` | `str` | `"time"` | Name of the time column. |
| `solver` | `str` | `"Tsit5"` | Julia ODE solver. `"Tsit5"` for non-stiff systems; `"Rodas5"` or `"QNDF"` for stiff. See [solver table](../choosing-backend.md#julia-solver-reference). |

Custom `network` is not supported in `JuliaNODE`. Use `NODE` if you need a custom architecture.

### Methods

#### `train`

```python
model.train(
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    epochs: int = 500,
    verbose: bool = True,
) -> JuliaNODE
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | `str` | `"adam"` | `"adam"` or `"sgd"`. |
| `learning_rate` | `float` | `1e-3` | Optimizer learning rate. |
| `epochs` | `int` | `500` | Number of gradient steps. |
| `verbose` | `bool` | `True` | Print loss during training. |

Note: `JuliaNODE.train()` does not expose `loss` or `solver` at training time. The solver is
set at construction; the loss function is `"trajectory"` (simulation) internally.

#### `forecast`

```python
model.forecast(
    steps: int,
    dt: float | None = None,
) -> pd.DataFrame
```

Same semantics as `NODE.forecast()`. `initial_state` is ignored — UniversalDiffEq.jl
manages initial conditions internally.

#### `get_right_hand_side`

```python
model.get_right_hand_side() -> Callable
```

Returns `f(u: list | array, t: float) -> np.ndarray`. The Julia Lux network is called via the
bridge on each evaluation.

### Properties

Same as `NODE`: `data`, `is_trained`, `n_states`, `state_columns`, `time_column`.

---

## Examples

### Basic NODE

```python
import pandas as pd
import pyUDE as ude

data = pd.read_csv("observations.csv")   # columns: time, x, y

model = ude.NODE(data, hidden_units=64, hidden_layers=3)
model.train(epochs=1000, learning_rate=1e-3, verbose=True)

future = model.forecast(steps=100)
print(future.head())
```

### Custom network architecture

```python
import torch.nn as nn
import pyUDE as ude

class ResidualMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, u):
        return self.linear(u) + u   # residual connection

model = ude.NODE(data, network=ResidualMLP(2).double())
model.train(epochs=500)
```

### Julia NODE for stiff system

```python
from pyUDE import JuliaNODE

# Fast-slow system: use Rodas5 implicit solver
model = JuliaNODE(data, solver="Rodas5")
model.train(epochs=300)
```
