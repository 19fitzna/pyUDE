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
    device: str = "cpu",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Time series data. Must contain a time column and at least one state column. Rows are time steps. |
| `network` | `nn.Module \| None` | `None` | Custom PyTorch network. Must accept a 1-D tensor of shape `(n_states,)` and return a tensor of the same shape. If `None`, a default MLP is built from `hidden_layers` and `hidden_units`. |
| `hidden_layers` | `int` | `2` | Number of hidden layers in the default MLP. Ignored if `network` is provided. |
| `hidden_units` | `int` | `32` | Units per hidden layer in the default MLP. |
| `time_column` | `str` | `"time"` | Name of the time column in `data`. |
| `device` | `str` | `"cpu"` | PyTorch device to train and run inference on. Accepts any string recognised by `torch.device`: `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"` (Apple Silicon). |

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
    patience: int | None = None,
    max_grad_norm: float = 10.0,
    weight_decay: float | None = None,
    noise_scale: float = 0.01,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> NODE
```

Fit the model to the training data. Returns `self` for chaining. Calling `train()` a second time
**continues** from the current weights — it does not reset the model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss` | `str` | `"simulation"` | `"simulation"`: integrate ODE forward and compare to observations via MSE. `"derivative_matching"`: estimate derivatives via cubic spline interpolation (falling back to finite differences if scipy is unavailable), then match predicted derivatives directly — no ODE integration required. |
| `optimizer` | `str` | `"adam"` | `"adam"` or `"sgd"`. |
| `learning_rate` | `float` | `1e-3` | Optimizer learning rate. |
| `epochs` | `int` | `500` | Number of gradient steps. |
| `log_interval` | `int` | `50` | Print loss every this many epochs (when `verbose=True`). |
| `verbose` | `bool` | `True` | Print loss during training. |
| `solver` | `str` | `"dopri5"` | torchdiffeq solver name. Options: `"dopri5"` (adaptive 4/5 RK), `"rk4"` (fixed 4th-order), `"euler"`, `"midpoint"`, `"explicit_adams"`. Only used when `loss="simulation"`. |
| `patience` | `int \| None` | `None` | Early stopping: stop if loss does not improve for this many epochs. Best weights are restored on stop. |
| `max_grad_norm` | `float` | `10.0` | Maximum gradient norm for clipping. Set to `0` to disable. Protects against exploding gradients during long-horizon integration. |
| `weight_decay` | `float \| None` | `None` | L2 regularisation applied via the optimizer. Defaults to `1e-4` when `loss="derivative_matching"` and `0.0` for simulation. Pass an explicit float to override. |
| `noise_scale` | `float` | `0.01` | Standard deviation of Gaussian noise injected into training states during derivative matching. Larger values encourage generalisation; set to `0.0` to disable. Only used when `loss="derivative_matching"`. |
| `rtol` | `float` | `1e-3` | Relative tolerance for the adaptive ODE solver. Only used with `loss="simulation"` and adaptive solvers (`dopri5`, etc.). Increase to `1e-2` for faster warm-up training. |
| `atol` | `float` | `1e-6` | Absolute tolerance for the adaptive ODE solver. Increase to `1e-4` alongside `rtol=1e-2` for faster warm-up training. |

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

#### `save`

```python
model.save(path: str) -> None
```

Save the trained model weights and metadata to disk.

```python
model.save("node_model.pt")
```

The file stores the network state dict, training data, time column name, and solver. Requires
the model to be trained first.

#### `load_weights`

```python
model.load_weights(path: str) -> NODE
```

Load weights from a checkpoint written by `save()` into this model. The model architecture
must match. After loading, `is_trained` is set to `True` and the model can be used for
forecasting or continued training.

```python
model = ude.NODE(data, hidden_units=64, hidden_layers=3)
model.load_weights("node_model.pt")
future = model.forecast(steps=50)
```

> **Note:** Julia-backed models (`JuliaNODE`, etc.) do not support `save()` or
> `load_weights()`. Trained parameters live inside a Julia struct and cannot be
> serialised via PyTorch. Use `get_params()` to extract learned parameter values
> and store them manually. See [Troubleshooting](../troubleshooting.md#julia-models-cannot-be-saved).

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

### Training on GPU

```python
import torch
import pyUDE as ude

# NVIDIA GPU
model = ude.NODE(data, device="cuda")
model.train(epochs=500)

# Apple Silicon (MPS)
model = ude.NODE(data, device="mps")
model.train(epochs=500)

# Check availability before choosing device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ude.NODE(data, device=device)
```

Forecast always returns a CPU `pd.DataFrame` regardless of device, so downstream code never
needs to change.

### Additive (incremental) training

```python
# Coarse first pass
model = ude.NODE(data, hidden_units=64)
model.train(epochs=200, learning_rate=1e-3)

# Fine-tune — continues from existing weights
model.train(epochs=500, learning_rate=1e-4)
```

### Early stopping

```python
model = ude.NODE(data)
model.train(epochs=2000, patience=100)
# Stops early if loss doesn't improve for 100 consecutive epochs
# and restores the best weights found
```

### Julia NODE for stiff system

```python
from pyUDE import JuliaNODE

# Fast-slow system: use Rodas5 implicit solver
model = JuliaNODE(data, solver="Rodas5")
model.train(epochs=300)
```
