# Troubleshooting

Common errors and how to fix them.

---

## Construction errors

### `known_dynamics raised an error on test input: ...`

pyUDE probes your `known_dynamics` (or `known_map`) function with zero inputs at
construction time to catch signature and shape mistakes early.

**Common causes:**

- The function signature doesn't match `f(u, p, t)` — check argument order.
- You're indexing `u` with named keys (e.g. `u["prey"]`) instead of integer indices
  (`u[0]`). In the PyTorch backend `u` is a 1-D tensor.
- An operation inside the function doesn't support the `torch.float64` dtype.
- For the Julia backend `u` is a list of floats, not a tensor — don't call
  `.item()` or tensor methods on it.

**Fix:** Test your function manually before passing it to the model:

```python
import torch
u_test = torch.zeros(n_states, dtype=torch.float64)
p_test = {k: torch.tensor(float(v), dtype=torch.float64) for k, v in init_params.items()}
t_test = torch.tensor(0.0, dtype=torch.float64)
print(your_known_dynamics(u_test, p_test, t_test))  # should print a tensor of shape (n_states,)
```

### `known_dynamics must return shape (N,), got ...`

Your function returns the wrong number of state variables. The return shape must
exactly match the number of non-time columns in your DataFrame.

---

## Training errors

### `NaN loss detected at epoch N. Stopping training.`

The loss became numerically unstable. Common causes:

- **Learning rate too high** — try `learning_rate=1e-4` or lower.
- **Outliers in the data** — large state values produce large derivatives. Normalise
  or clip the data before training.
- **Gradient explosion** — increase gradient clipping: `max_grad_norm=1.0`.
- **Bad initial parameters** — `init_params` far from the true values can produce
  unstable gradients at the start. Start closer to plausible values.

### Loss immediately collapses to zero or stops improving

- **Simulation loss with `rk4`** on oscillatory systems (e.g. Lotka-Volterra) often
  produces inaccurate gradients. Switch to `solver="dopri5"`.
- **Straight-line forecasts with `derivative_matching`** — always follow up
  derivative matching with a simulation fine-tune:

```python
model.train(loss="derivative_matching", epochs=500, learning_rate=1e-3)
model.train(loss="simulation", epochs=1000, learning_rate=1e-4, patience=100)
```

---

## Data validation errors

### `Duplicate values found in time column 'time'`

Two or more rows share the same timestamp. This causes division-by-zero in derivative
estimation.

**Fix:** Average repeated observations or drop duplicates:

```python
data = data.groupby("time").mean().reset_index()
```

### `Time column 'time' must be numeric, got dtype object`

The time column contains strings or a mixed dtype.

**Fix:**

```python
data["time"] = pd.to_numeric(data["time"])
```

### `Non-numeric state columns found: [...]`

The DataFrame contains string or categorical columns that can't be used as state
variables.

**Fix:** Drop them before constructing the model:

```python
data = data.drop(columns=["label", "category"])
```

### `Time column 'time' must be monotonically increasing`

Rows are not sorted by time.

**Fix:**

```python
data = data.sort_values("time").reset_index(drop=True)
```

---

## Forecast errors

### `steps must be a positive integer, got 0`

`forecast(steps=0)` or `forecast(steps=-1)` returns no data. Pass a positive integer.

### `Model has not been trained yet. Call model.train() first.`

You called `forecast()`, `get_params()`, or `get_right_hand_side()` before training.
Call `.train()` first, or load weights with `.load_weights()`.

### `torchdiffeq is required for forecasting`

`forecast()` on continuous-time models (NODE, CustomDerivatives) requires torchdiffeq:

```bash
pip install torchdiffeq
```

---

## GPU / device errors

### `CUDA not available` despite having an NVIDIA GPU

`pip install pyUDE` installs the CPU-only build of PyTorch by default. Reinstall with CUDA support:

```bash
pip uninstall torch torchdiffeq -y
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install torchdiffeq
```

See [Installation — GPU Support](installation.md#gpu-support) for full instructions.

### `RuntimeError: Expected all tensors to be on the same device`

You passed an `initial_state` tensor on a different device than the model. Either pass
a CPU tensor (pyUDE will move it automatically) or match devices explicitly:

```python
u0 = u0.to(model._device)
```

### MPS (Apple Silicon) not working

PyTorch MPS does not support `float64`. pyUDE uses float64 throughout. Use `device="cpu"`
on Apple Silicon.

---

## Persistence errors

### `save()` on a `CustomDifferences` model — loading gives wrong shapes

`CustomDifferences.save()` and `load_weights()` require the model architecture
(hidden layers, units, `init_params` keys) to match exactly between the saving and
loading instances.

### Julia models cannot be saved

The Julia backend stores trained parameters inside a Julia struct in process memory.
`save()` and `load_weights()` are not available for `JuliaNODE`,
`JuliaCustomDerivatives`, or `JuliaCustomDifferences`. To persist a Julia-backed
model, re-train it or export the learned parameters with `get_params()` and store them
separately.
