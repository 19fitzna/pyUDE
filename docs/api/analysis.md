# Analysis Utilities

These functions can be called either as **standalone functions** (passing the model as the
first argument) or as **methods** on any model object. Both forms are equivalent.

```python
# Method form
predictions = model.forecast(steps=50)
rhs = model.get_right_hand_side()

# Standalone form (useful in pipelines)
from pyUDE import forecast, get_right_hand_side
predictions = forecast(model, steps=50)
rhs = get_right_hand_side(model)
```

---

## `forecast`

```python
pyUDE.forecast(
    model,
    steps: int,
    dt: float | None = None,
    initial_state: torch.Tensor | None = None,
) -> pd.DataFrame
```

Integrate (or step) the learned model forward from the last observed state.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | any trained model | — | A trained `NODE`, `CustomDerivatives`, `CustomDifferences`, or Julia equivalent. |
| `steps` | `int` | — | Number of future time steps to predict. |
| `dt` | `float \| None` | `None` | Step size. Defaults to the median interval in the training data. For `CustomDifferences`, this is the number of discrete steps. |
| `initial_state` | `Tensor \| None` | `None` | Starting state for the forecast. Defaults to the last observed state in the training data. Ignored by Julia backend models. |

### Returns

`pd.DataFrame` with columns `[time_column, *state_columns]`. Contains exactly `steps` rows.
The initial state is **not** included.

### Notes

- For continuous-time models (`NODE`, `CustomDerivatives`, `JuliaNODE`,
  `JuliaCustomDerivatives`), the ODE is integrated forward using the solver that was specified
  at training time.
- For discrete-time models (`CustomDifferences`, `JuliaCustomDifferences`), the map is stepped
  forward.
- Forecasted times are evenly spaced at `dt`-second intervals starting from
  `t_last + dt`.

### Example

```python
model.train(epochs=500)

# Forecast 100 steps ahead at the default time spacing
fc = model.forecast(steps=100)

# Forecast at a custom step size
fc = model.forecast(steps=50, dt=0.1)

# Forecast from a custom initial condition
import torch
u0 = torch.tensor([8.0, 3.0], dtype=torch.float64)
fc = model.forecast(steps=50, initial_state=u0)
```

---

## `get_right_hand_side`

```python
pyUDE.get_right_hand_side(model) -> Callable
```

Extract the learned dynamics as a numpy-compatible function.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | any trained continuous-time model | `NODE`, `CustomDerivatives`, `JuliaNODE`, or `JuliaCustomDerivatives`. Not applicable to discrete-time models. |

### Returns

A callable `f(u, t) -> du` where:
- `u`: numpy array or list of length `n_states`
- `t`: scalar float
- returns: numpy array of length `n_states`

This function is **compatible with `scipy.integrate.odeint`** and similar tools.

### Example

```python
rhs = model.get_right_hand_side()

# Evaluate at a single point
import numpy as np
du = rhs(np.array([10.0, 5.0]), t=0.0)

# Integrate with scipy
from scipy.integrate import odeint
t_span = np.linspace(0, 20, 500)
u0 = np.array([10.0, 5.0])
sol = odeint(rhs, y0=u0, t=t_span)

# Plot phase portrait
import matplotlib.pyplot as plt
plt.plot(sol[:, 0], sol[:, 1])
plt.xlabel("prey"); plt.ylabel("predator")

# Integrate with solve_ivp (different argument order!)
from scipy.integrate import solve_ivp
def rhs_ivp(t, u):          # scipy.solve_ivp uses (t, u) not (u, t)
    return rhs(u, t)
sol = solve_ivp(rhs_ivp, t_span=[0, 20], y0=u0, dense_output=True)
```

> **Argument order**: `scipy.integrate.odeint` uses `f(u, t)` (state first),
> while `scipy.integrate.solve_ivp` uses `f(t, u)` (time first). The function returned by
> `get_right_hand_side()` follows the `odeint` convention `f(u, t)`.
