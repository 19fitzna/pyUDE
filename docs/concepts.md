# Core Concepts

## Universal Differential Equations

A **Universal Differential Equation** (UDE) is a differential equation in which one or more
terms are replaced by a universal function approximator — typically a neural network. The
concept was introduced by Rackauckas et al. (2020) in
[Universal Differential Equations for Scientific Machine Learning](https://arxiv.org/abs/2001.04385).

The general form is:

```
du/dt = f(u, p, t, NN_θ(u))
```

where:
- `u` is the state vector (e.g. prey and predator populations)
- `p` are the mechanistic parameters (e.g. growth rates)
- `t` is time
- `NN_θ` is a neural network with learned weights `θ`

The key insight is that you can **mix** known physics/biology/chemistry with learned unknowns.
This is more data-efficient than a pure neural ODE and produces more interpretable models than
a black-box network.

---

## The Three Model Types

### NODE — Neural ODE

```
du/dt = NN(u)
```

The neural network *is* the dynamics. Nothing is known in advance. The ODE is integrated
forward in time and the network weights are adjusted to match the observations.

Use when: no prior knowledge of the system exists, or as a baseline before adding structure.

### CustomDerivatives — Hybrid UDE

```
du/dt = f_known(u, p, t)  +  NN(u)
```

You supply `f_known` — the terms you understand. The network learns the residual: everything
your mechanistic model fails to capture. Both the neural network weights `θ` *and* the
mechanistic parameters `p` are optimised jointly.

Use when: you have partial knowledge of the dynamics (e.g. you know growth is exponential but
don't know the interaction structure).

### CustomDifferences — Discrete-time UDE

```
u[n+1] = g_known(u[n], p, n)  +  NN(u[n])
```

The discrete-time equivalent of CustomDerivatives. Instead of integrating a differential
equation, the model steps forward in discrete time using a known map plus a learned residual.

Use when: your data is naturally discrete (population censuses, daily observations, financial
time series), or you are modelling a map/difference equation directly.

---

## How Training Works

### Simulation loss (continuous-time)

1. Integrate the ODE forward from the first observed state using an adaptive solver (Tsit5 or
   dopri5).
2. Compute MSE between the predicted trajectory and the observations.
3. Differentiate through the ODE solver using the **adjoint method** — a memory-efficient
   technique that avoids storing intermediate solver states.
4. Update parameters with Adam.

The adjoint method solves a second ODE backwards in time to compute gradients. This is why
`torchdiffeq.odeint_adjoint` (PyTorch backend) or `SciMLSensitivity.jl` (Julia backend) is
used rather than direct automatic differentiation through the solver steps.

### Derivative-matching loss (PyTorch backend only)

1. Estimate derivatives from the data using **cubic spline interpolation** (via
   `scipy.interpolate.CubicSpline`). Falls back to central finite differences if scipy is not
   installed.
2. At each epoch, perturb the observed states with small Gaussian noise before evaluating the
   ODE right-hand side. This encourages the network to generalise to nearby states, not just
   the exact observed points — which prevents straight-line forecasts when integrating.
3. Compute MSE between predicted and estimated derivatives — **no ODE integration required**.
4. Update with Adam + L2 weight decay (default `1e-4`; pass `weight_decay=0` to disable).

This is faster per epoch than simulation loss and does not require `torchdiffeq`. It is often
useful for a quick first fit before switching to simulation loss. For best results, install
scipy: `pip install scipy`.

### Discrete-time loss

1. Apply the map `u[n+1] = g_known(u[n], p, n) + NN(u[n])` at each observed state.
2. Compute one-step-ahead MSE against `u[n+1]`.
3. Update with Adam.

### Training safeguards

All training loops include:

- **Gradient clipping** (`max_grad_norm=10.0`): caps the gradient norm before each optimizer
  step, preventing gradient explosions during long-horizon ODE integration.
- **NaN detection**: if the loss becomes `NaN`, training stops immediately with a warning rather
  than continuing to corrupt model weights.
- **Early stopping** (`patience=N`): stops training if the loss does not improve for `N`
  consecutive epochs and restores the best weights seen so far.

### Additive training

Calling `train()` more than once on the same model **continues training from the current
weights** — the model is not re-initialised. This is useful for staged training strategies:

```python
model.train(epochs=200, learning_rate=1e-3)   # coarse pass
model.train(epochs=500, learning_rate=1e-4)   # fine-tune
```

Each call creates a fresh optimizer, so `learning_rate` and `optimizer` can be changed freely
between calls.

---

## Adjoint Sensitivity Analysis

Standard automatic differentiation would differentiate through every solver step, creating a
computation graph proportional to the number of steps. The adjoint method instead solves the
*adjoint ODE* backwards in time:

```
d(loss)/d(θ) = ∫ a(t)ᵀ ∂f/∂θ dt
```

where `a(t)` is the adjoint state satisfying `da/dt = -a(t)ᵀ ∂f/∂u`.

The result: gradient computation costs the same regardless of the number of ODE solver steps,
making long-horizon integrations practical.

In pyUDE:
- **PyTorch backend**: uses `torchdiffeq.odeint_adjoint`
- **Julia backend**: uses `SciMLSensitivity.jl` via UniversalDiffEq.jl

---

## The `known_dynamics` / `known_map` Callable

The function you supply to `CustomDerivatives` or `CustomDifferences` must follow a specific
signature. See [Writing known_dynamics Functions](guides/known-dynamics.md) for details and
patterns.

### PyTorch backend signature

```python
def f(u: torch.Tensor, p: dict, t: torch.Tensor) -> torch.Tensor:
    # u: shape (n_states,), dtype float64
    # p: dict mapping str → scalar torch.Tensor (trainable)
    # t: scalar tensor
    # returns: shape (n_states,), dtype float64
    ...
```

### Julia backend signature

```python
def f(u: list, p: dict, t: float) -> list:
    # u: list of floats
    # p: dict mapping str → float
    # t: float
    # returns: list of floats
    ...
```

The Julia backend accepts plain Python lists/floats because the function crosses the
Python↔Julia bridge via PythonCall.jl. PyTorch tensors cannot cross this boundary.

---

## GPU Acceleration

PyTorch models support GPU training via the `device` constructor parameter:

```python
model = ude.NODE(data, device="cuda")   # NVIDIA GPU
model = ude.NODE(data, device="mps")    # Apple Silicon (Metal Performance Shaders)
model = ude.NODE(data, device="cpu")    # CPU (default)
```

When `device` is set, the neural network parameters and all training tensors (`t`, `u`) are
moved to the device automatically. `forecast()` always returns a CPU `pd.DataFrame`.

The adjoint ODE solver (`torchdiffeq.odeint_adjoint`) operates natively on GPU tensors, so
the full forward integration and backward adjoint pass run on the accelerator.

Julia backend models use Julia's GPU packages (CUDA.jl, Metal.jl) independently and are not
affected by the `device` parameter.

---

## Further Reading

- [Rackauckas et al. (2020) — Universal Differential Equations for Scientific Machine Learning](https://arxiv.org/abs/2001.04385)
- [UniversalDiffEq.jl documentation](https://jack-h-buckner.github.io/UniversalDiffEq.jl/dev/)
- [torchdiffeq — adjoint sensitivity for PyTorch](https://github.com/rtqichen/torchdiffeq)
- [DifferentialEquations.jl documentation](https://docs.sciml.ai/DiffEqDocs/stable/)
