"""Training routines for UDE models."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pyUDE.core.base import UDEModel
    from pyUDE.core.custom_differences import CustomDifferences


def _make_optimizer(params, name: str, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unknown optimizer '{name}'. Choose 'adam' or 'sgd'.")


# ---------------------------------------------------------------------------
# Continuous-time (ODE) training
# ---------------------------------------------------------------------------

def train_model(
    model: "UDEModel",
    loss: str = "simulation",
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    epochs: int = 500,
    log_interval: int = 50,
    verbose: bool = True,
    solver: str = "dopri5",
    **kwargs,
) -> None:
    """
    Train a continuous-time UDE model (NODE or CustomDerivatives).

    Parameters
    ----------
    model : UDEModel
        The model to train. ``model._ode_func`` must already be constructed.
    loss : {"simulation", "derivative_matching"}
    optimizer_name : str
    learning_rate : float
    epochs : int
    log_interval : int
    verbose : bool
    solver : str
        torchdiffeq solver name.
    """
    try:
        from torchdiffeq import odeint_adjoint as odeint
    except ImportError as e:
        raise ImportError(
            "torchdiffeq is required for continuous-time UDE training. "
            "Install it with: pip install torchdiffeq"
        ) from e

    ode_func = model._ode_func
    t, u_obs, _ = model._get_training_tensors()

    optimizer = _make_optimizer(ode_func.parameters(), optimizer_name, learning_rate)
    loss_fn = nn.MSELoss()

    if loss == "simulation":
        _train_simulation(ode_func, t, u_obs, optimizer, loss_fn, odeint,
                          solver, epochs, log_interval, verbose)
    elif loss == "derivative_matching":
        _train_derivative_matching(ode_func, t, u_obs, optimizer, loss_fn,
                                   epochs, log_interval, verbose)
    else:
        raise ValueError(f"Unknown loss '{loss}'. Choose 'simulation' or 'derivative_matching'.")


def _train_simulation(ode_func, t, u_obs, optimizer, loss_fn, odeint,
                      solver, epochs, log_interval, verbose):
    """Integrate ODE forward and compare to observations."""
    u0 = u_obs[0]
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        u_pred = odeint(ode_func, u0, t, method=solver, adjoint_params=tuple(ode_func.parameters()))
        loss = loss_fn(u_pred, u_obs)
        loss.backward()
        optimizer.step()

        if verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch:>5d}/{epochs}  loss={loss.item():.6f}")


def _train_derivative_matching(ode_func, t, u_obs, optimizer, loss_fn,
                                epochs, log_interval, verbose):
    """
    Compare predicted derivatives against finite-difference estimates.
    Faster than simulation but noisier on coarsely-sampled data.
    """
    # Central finite differences for interior points; forward/backward at edges
    dt = t[1:] - t[:-1]
    du_fd = torch.zeros_like(u_obs)
    du_fd[1:-1] = (u_obs[2:] - u_obs[:-2]) / (t[2:] - t[:-2]).unsqueeze(1)
    du_fd[0]    = (u_obs[1]  - u_obs[0])  / dt[0]
    du_fd[-1]   = (u_obs[-1] - u_obs[-2]) / dt[-1]

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        du_pred = torch.stack([ode_func(t[i], u_obs[i]) for i in range(len(t))])
        loss = loss_fn(du_pred, du_fd)
        loss.backward()
        optimizer.step()

        if verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch:>5d}/{epochs}  loss={loss.item():.6f}")


# ---------------------------------------------------------------------------
# Discrete-time training
# ---------------------------------------------------------------------------

def train_differences(
    model: "CustomDifferences",
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    epochs: int = 500,
    log_interval: int = 50,
    verbose: bool = True,
) -> None:
    """Train a discrete-time UDE model by minimising one-step-ahead MSE."""
    t, u_obs, _ = model._get_training_tensors()
    param_dict = model._param_dict
    network = model._network_module
    known_map = model._known_map

    all_params = list(param_dict.parameters()) + list(network.parameters())
    optimizer = _make_optimizer(all_params, optimizer_name, learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        p = {k: v for k, v in param_dict.items()}
        u_next_pred = torch.stack([
            known_map(u_obs[i], p, t[i]) + network(u_obs[i])
            for i in range(len(t) - 1)
        ])
        loss = loss_fn(u_next_pred, u_obs[1:])
        loss.backward()
        optimizer.step()

        if verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch:>5d}/{epochs}  loss={loss.item():.6f}")
