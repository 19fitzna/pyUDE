# Quickstart

This guide walks through building your first UDE in five minutes. We use synthetic data so
you can run everything without any external files.

## 1. Install

```bash
pip install pyUDE
```

## 2. Generate synthetic data

```python
import numpy as np
import pandas as pd

# Simulate Lotka-Volterra (predator-prey)
def simulate_lv(t_end=10, n=80, alpha=1.0, beta=0.1, delta=0.075, gamma=1.5):
    t = np.linspace(0, t_end, n)
    dt = t[1] - t[0]
    prey, pred = 10.0, 5.0
    rows = []
    for ti in t:
        rows.append((ti, prey, pred))
        prey = max(prey + dt * (alpha*prey - beta*prey*pred), 1e-3)
        pred = max(pred + dt * (delta*prey*pred - gamma*pred), 1e-3)
    return pd.DataFrame(rows, columns=["time", "prey", "predator"])

data = simulate_lv()
print(data.head())
#    time   prey  predator
# 0   0.0  10.00      5.00
# 1   0.13   9.97      4.92
# ...
```

## 3. Train a Neural ODE (no prior knowledge)

Use `NODE` when you have no idea what the dynamics look like:

```python
import pyUDE as ude

model = ude.NODE(data, hidden_units=32)
model.train(epochs=500, verbose=True)
# Epoch   50/500  loss=0.421832
# Epoch  100/500  loss=0.183401
# ...
```

## 4. Forecast and visualise

```python
import matplotlib.pyplot as plt

forecast = model.forecast(steps=30)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col in zip(axes, ["prey", "predator"]):
    ax.plot(data["time"], data[col], "k.", ms=4, label="Training data")
    ax.plot(forecast["time"], forecast[col], "r--", lw=2, label="Forecast")
    ax.axvline(data["time"].iloc[-1], color="gray", ls=":")
    ax.set(xlabel="time", ylabel=col, title=col)
    ax.legend()
plt.tight_layout()
plt.show()
```

## 5. Hybrid UDE — encoding known structure

If you know that prey grow exponentially and predators die without prey, encode that:

```python
import torch

def lv_known(u, p, t):
    prey, pred = u[0], u[1]
    return torch.stack([
        p["alpha"] * prey,    # known: exponential growth
        -p["delta"] * pred,   # known: linear death
        # unknown: predation interaction → learned by NN
    ])

hybrid = ude.CustomDerivatives(
    data,
    known_dynamics=lv_known,
    init_params={"alpha": 0.8, "delta": 1.2},   # initial guesses
    hidden_units=16,
)
hybrid.train(epochs=1500, learning_rate=3e-3)

# Check recovered parameters (true: alpha=1.0, delta=1.5)
print(hybrid.get_params())
```

## 6. Extract the learned dynamics

Use the learned model as a standard ODE with any Python solver:

```python
from scipy.integrate import odeint

rhs = hybrid.get_right_hand_side()    # f(u, t) -> du

t_span = np.linspace(0, 15, 400)
u0 = data[["prey", "predator"]].iloc[0].values
sol = odeint(rhs, y0=u0, t=t_span)

plt.plot(sol[:, 0], sol[:, 1])
plt.xlabel("prey"); plt.ylabel("predator")
plt.title("Phase portrait of learned dynamics")
plt.show()
```

## Next steps

- [Core Concepts](../concepts.md) — understand adjoint methods and loss functions
- [Choosing a Backend](../choosing-backend.md) — when to use Julia for better performance
- [Writing known_dynamics Functions](known-dynamics.md) — tips for structuring your model
- [Notebook example](../../notebooks/01_quickstart.ipynb) — interactive version of this guide
