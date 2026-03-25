# Writing `known_dynamics` Functions

The `known_dynamics` (or `known_map`) function is the most important part of a
`CustomDerivatives` or `CustomDifferences` model. This guide covers patterns, pitfalls, and
tips for writing it well.

---

## The Basic Contract

Your function defines the **mechanistic part** of the dynamics. The neural network handles
everything else. The two are combined additively:

```
du/dt = known_dynamics(u, p, t)  +  NN(u)
```

So your function should return **only the terms you know** — not the full dynamics.

---

## PyTorch Backend Signature

```python
def f(u: torch.Tensor, p: dict, t: torch.Tensor) -> torch.Tensor:
    ...
```

- `u` is a 1-D `float64` tensor of shape `(n_states,)`
- `p` is a dict of `{str: scalar tensor}` — these are trainable
- `t` is a scalar `float64` tensor
- Return a tensor of the same shape as `u`

### Rules

1. **Use `torch` operations** — no `numpy`, no Python math. The function must be
   differentiable so that gradients can flow through `p`.
2. **Use `torch.stack` to build the output**, not `torch.tensor(list)`.
3. **Index with integers** `u[0]`, `u[1]`, not slices — slices return views that can cause
   in-place mutation issues.

---

## Julia Backend Signature

```python
def f(u: list, p: dict, t: float) -> list:
    ...
```

- `u` is a plain Python `list[float]`
- `p` is a plain Python `dict[str, float]`
- `t` is a plain Python `float`
- Return a plain `list[float]`

### Rules

1. **No PyTorch** inside this function — it runs on the Julia side via PythonCall.jl.
2. **Plain Python arithmetic only** — `u[0] * p["r"]` is fine; `torch.exp(u[0])` is not.
3. **Numpy is allowed** but introduces overhead; prefer plain lists.

---

## Common Patterns

### Linear terms

```python
# PyTorch
def linear(u, p, t):
    return torch.stack([p["a"] * u[0], p["b"] * u[1]])

# Julia
def linear(u, p, t):
    return [p["a"] * u[0], p["b"] * u[1]]
```

### Nonlinear terms (exponential, logistic)

```python
# PyTorch
import torch

def growth(u, p, t):
    return torch.stack([
        p["r"] * u[0] * (1 - u[0] / p["K"]),  # logistic growth
    ])

# Julia
import math

def growth(u, p, t):
    return [p["r"] * u[0] * (1 - u[0] / p["K"])]
```

### Time-varying terms

```python
# PyTorch — seasonal forcing
def seasonal(u, p, t):
    forcing = p["A"] * torch.sin(2 * torch.pi * t / p["period"])
    return torch.stack([p["r"] * u[0] + forcing])

# Julia
import math

def seasonal(u, p, t):
    forcing = p["A"] * math.sin(2 * math.pi * t / p["period"])
    return [p["r"] * u[0] + forcing]
```

### Conservation constraints

If you know that the sum of states is conserved, you can encode this:

```python
# PyTorch — SIR model (S + I + R = const)
def sir_known(u, p, t):
    S, I, R = u[0], u[1], u[2]
    infection = p["beta"] * S * I
    # recovery is unknown — learned by NN
    return torch.stack([
        -infection,
        infection,
        torch.tensor(0.0, dtype=torch.float64),  # R's known contribution: 0
    ])
```

### Partial specification (only some states known)

You don't have to know every term. Return zero for states where you have no knowledge:

```python
# PyTorch — only prey dynamics are known
def partial_lv(u, p, t):
    prey, pred = u[0], u[1]
    return torch.stack([
        p["alpha"] * prey - p["beta"] * prey * pred,  # fully known
        torch.zeros(1, dtype=torch.float64).squeeze(),  # predator: unknown
    ])
```

---

## Parameter Initialisation

`init_params` sets the starting point for optimisation. Good initialisations converge faster
and to better solutions.

```python
# Too far from truth: slow convergence or divergence
init_params={"alpha": 10.0, "delta": 0.01}

# Close to expected range: good
init_params={"alpha": 1.0, "delta": 1.5}
```

**Tips:**
- Use domain knowledge to set plausible ranges.
- For rate parameters, orders of magnitude matter more than exact values.
- If training diverges, try reducing `learning_rate` and bringing `init_params` closer to
  expected values.
- All parameters are unconstrained floats during optimisation. If a parameter must be
  positive (e.g. a rate constant), you may want to apply `torch.exp(p["r"])` inside your
  function and initialise with `{"r": 0.0}` (so `exp(0) = 1.0`).

### Constraining parameters to positive values

```python
# PyTorch — constrain r to be positive via exp reparameterisation
def logistic(u, p, t):
    r = torch.exp(p["log_r"])     # always positive
    K = torch.exp(p["log_K"])
    return torch.stack([r * u[0] * (1 - u[0] / K)])

model = ude.CustomDerivatives(
    data, logistic,
    init_params={"log_r": 0.0, "log_K": 5.0},   # exp(0)=1, exp(5)≈148
)
```

---

## Debugging

### Check the function independently

Before training, verify that your function returns the right shape:

```python
import torch

u_test = torch.tensor([10.0, 5.0], dtype=torch.float64)
p_test = {k: torch.tensor(v, dtype=torch.float64) for k, v in init_params.items()}
t_test = torch.tensor(0.0, dtype=torch.float64)

du = lv_known(u_test, p_test, t_test)
print(du.shape)   # should be (2,) for a 2-state system
print(du)
```

### Check the function runs without NaN

```python
# Check for NaN/Inf in output
assert torch.isfinite(du).all(), f"known_dynamics returned non-finite values: {du}"
```

### Common errors

| Error | Likely cause |
|-------|-------------|
| `RuntimeError: Expected all tensors to be on the same device` | Mixing CPU/GPU tensors |
| `RuntimeError: inplace operation` | Using `u[0] += ...` instead of `u[0] + ...` |
| `ValueError: grad can be implicitly created only for scalar outputs` | Returning wrong shape |
| `NaN loss after first epoch` | Parameters or learning rate too large |
| Output shape mismatch | Returning wrong number of state derivatives |
