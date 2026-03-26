# pyUDE

Universal Differential Equations in Python — a PyTorch-based library for building hybrid models that combine known mechanistic dynamics with neural networks for unknown terms.

Inspired by [UniversalDiffEq.jl](https://github.com/Jack-H-Buckner/UniversalDiffEq.jl).

## Installation

```bash
pip install pyUDE
```

With notebook support:

```bash
pip install "pyUDE[notebook]"
```

For development:

```bash
git clone https://github.com/19fitzna/pyUDE
cd pyUDE
pip install -e ".[dev,notebook]"
```

## Quick Start

### Neural ODE (NODE)

Learn dynamics entirely from data:

```python
import pandas as pd
import pyUDE as ude

# Load your time series (must have a 'time' column)
data = pd.read_csv("my_timeseries.csv")

model = ude.NODE(data, hidden_units=32, hidden_layers=2)
model.train(loss="simulation", epochs=500)

forecast = model.forecast(steps=50)
```

### Hybrid UDE — known dynamics + neural network

Encode partial knowledge and let the network learn the rest:

```python
import torch
import pyUDE as ude

def lv_known(u, p, t):
    """Known part: prey growth and predator death only."""
    prey, pred = u[0], u[1]
    return torch.stack([
        p["alpha"] * prey,          # prey grows
        -p["delta"] * pred,         # predator dies
    ])

model = ude.CustomDerivatives(
    data,
    known_dynamics=lv_known,
    init_params={"alpha": 1.0, "delta": 1.0},
)
model.train(epochs=1000)

# Inspect learned mechanistic parameters
print(model.get_params())  # {"alpha": ..., "delta": ...}

forecast = model.forecast(steps=50)
```

### Discrete-time UDE (CustomDifferences)

For maps and difference equations:

```python
def logistic_map(u, p, t):
    return p["r"] * u * (1 - u)

model = ude.CustomDifferences(
    data,
    known_map=logistic_map,
    init_params={"r": 2.5},
)
model.train(epochs=500)
```

### Inspect learned dynamics

```python
rhs = model.get_right_hand_side()  # returns numpy-compatible f(u, t)

from scipy.integrate import odeint
sol = odeint(rhs, y0=[1.0, 0.5], t=t_span)
```

## API Reference

### Models

| Class | Description |
|-------|-------------|
| `NODE(data, ...)` | Full neural ODE — network learns all dynamics |
| `CustomDerivatives(data, known_dynamics, init_params, ...)` | Hybrid UDE — known + NN |
| `CustomDifferences(data, known_map, init_params, ...)` | Discrete-time hybrid |

### Common model methods

| Method | Description |
|--------|-------------|
| `model.train(loss, optimizer, learning_rate, epochs, ...)` | Fit to data |
| `model.forecast(steps, dt)` | Integrate forward, returns DataFrame |
| `model.get_right_hand_side()` | Extract learned dynamics as a callable |

### `train()` options

| Parameter | Values | Default |
|-----------|--------|---------|
| `loss` | `"simulation"`, `"derivative_matching"` | `"simulation"` |
| `optimizer` | `"adam"`, `"sgd"` | `"adam"` |
| `solver` | any torchdiffeq solver name | `"dopri5"` |
| `epochs` | int | `500` |
| `learning_rate` | float | `1e-3` |
| `patience` | int or `None` | `None` (disabled) |
| `max_grad_norm` | float | `10.0` |
| `weight_decay` | float or `None` | `None` (auto: `1e-4` for derivative_matching, `0.0` for simulation) |

`loss="simulation"` integrates the ODE forward and requires `torchdiffeq`.
`loss="derivative_matching"` estimates derivatives via cubic spline interpolation (requires
`scipy`; falls back to finite differences without it) and does **not** require `torchdiffeq`.
It is faster per epoch but should be followed by simulation fine-tuning for best forecast
accuracy.

Repeated `train()` calls continue from existing weights (additive training). The optimizer
resets each call, allowing you to change `learning_rate` between phases.

## GPU Acceleration

All PyTorch models accept a `device` parameter:

```python
model = ude.NODE(data, device="cuda")      # NVIDIA GPU
model = ude.NODE(data, device="mps")       # Apple Silicon
model = ude.NODE(data, device="cpu")       # CPU (default)
```

Auto-select the best available device:

```python
import torch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = ude.NODE(data, device=device)
```

`forecast()` always returns a CPU `pd.DataFrame` regardless of the training device.

## Running Tests

```bash
pytest tests/unit/           # fast, no torchdiffeq needed
pytest tests/integration/    # requires torchdiffeq
```

## License

MIT © Nathan Fitzpatrick
