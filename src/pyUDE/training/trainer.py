"""Training routines for UDE models."""

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pyUDE.core.base import UDEModel
    from pyUDE.core.custom_differences import CustomDifferences


def _make_optimizer(params, name: str, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
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
    patience: Optional[int] = None,
    max_grad_norm: float = 10.0,
    weight_decay: float = 0.0,
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
    patience : int, optional
        Stop if loss does not improve for this many epochs (early stopping).
    max_grad_norm : float
        Maximum norm for gradient clipping. Set to 0 to disable.
    weight_decay : float
        L2 regularisation for the optimizer.
    """
    ode_func = model._ode_func
    t, u_obs, _ = model._get_training_tensors()
    device = model._device
    t = t.to(device)
    u_obs = u_obs.to(device)

    optimizer = _make_optimizer(ode_func.parameters(), optimizer_name, learning_rate,
                                weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    if loss == "simulation":
        try:
            from torchdiffeq import odeint_adjoint as odeint
        except ImportError as e:
            raise ImportError(
                "torchdiffeq is required for simulation loss. "
                "Install with: pip install torchdiffeq"
            ) from e
        _train_simulation(ode_func, t, u_obs, optimizer, loss_fn, odeint,
                          solver, epochs, log_interval, verbose, patience, max_grad_norm)
    elif loss == "derivative_matching":
        _train_derivative_matching(ode_func, t, u_obs, optimizer, loss_fn,
                                   epochs, log_interval, verbose, patience, max_grad_norm)
    else:
        raise ValueError(f"Unknown loss '{loss}'. Choose 'simulation' or 'derivative_matching'.")


def _train_simulation(ode_func, t, u_obs, optimizer, loss_fn, odeint,
                      solver, epochs, log_interval, verbose, patience, max_grad_norm):
    """Integrate ODE forward and compare to observations."""
    # adjoint_params captures all trainable parameters for the adjoint method
    n_params = sum(1 for _ in ode_func.parameters())
    assert n_params > 0, "No trainable parameters found in ODE function"

    u0 = u_obs[0]
    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        u_pred = odeint(ode_func, u0, t, method=solver, adjoint_params=tuple(ode_func.parameters()))
        loss = loss_fn(u_pred, u_obs)

        if torch.isnan(loss):
            if verbose:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in ode_func.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience and epochs_no_improve >= patience:
            ode_func.load_state_dict(best_state)
            if verbose:
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.6f}")
            break

        if verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}")


def _estimate_derivatives(t, u_obs):
    """Estimate du/dt using cubic spline interpolation. Falls back to finite differences if scipy is unavailable."""
    try:
        from scipy.interpolate import CubicSpline
        t_np = t.detach().cpu().numpy()
        u_np = u_obs.detach().cpu().numpy()
        du = np.zeros_like(u_np)
        for col in range(u_np.shape[1]):
            cs = CubicSpline(t_np, u_np[:, col])
            du[:, col] = cs(t_np, 1)
        return torch.tensor(du, dtype=u_obs.dtype, device=u_obs.device)
    except ImportError:
        # Fallback: central finite differences
        dt = t[1:] - t[:-1]
        du_fd = torch.zeros_like(u_obs)
        du_fd[1:-1] = (u_obs[2:] - u_obs[:-2]) / (t[2:] - t[:-2]).unsqueeze(1)
        du_fd[0]    = (u_obs[1]  - u_obs[0])  / dt[0]
        du_fd[-1]   = (u_obs[-1] - u_obs[-2]) / dt[-1]
        return du_fd


def _train_derivative_matching(ode_func, t, u_obs, optimizer, loss_fn,
                                epochs, log_interval, verbose, patience, max_grad_norm,
                                noise_scale=0.01):
    """
    Compare predicted derivatives against cubic-spline derivative estimates.
    Injects Gaussian noise into training states to encourage generalisation
    beyond exact data points.
    """
    du_target = _estimate_derivatives(t, u_obs)

    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Perturb observed states so the network generalizes beyond exact data points
        u_input = u_obs + noise_scale * torch.randn_like(u_obs)

        du_pred = torch.stack([ode_func(t[i], u_input[i]) for i in range(len(t))])
        loss = loss_fn(du_pred, du_target)

        if torch.isnan(loss):
            if verbose:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in ode_func.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience and epochs_no_improve >= patience:
            ode_func.load_state_dict(best_state)
            if verbose:
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.6f}")
            break

        if verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}")


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
    patience: Optional[int] = None,
    max_grad_norm: float = 10.0,
) -> None:
    """Train a discrete-time UDE model by minimising one-step-ahead MSE."""
    t, u_obs, _ = model._get_training_tensors()
    device = model._device
    t = t.to(device)
    u_obs = u_obs.to(device)
    param_dict = model._param_dict
    network = model._network_module
    known_map = model._known_map

    all_params = list(param_dict.parameters()) + list(network.parameters())
    optimizer = _make_optimizer(all_params, optimizer_name, learning_rate)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_param_state = None
    best_net_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        p = {k: v for k, v in param_dict.items()}
        u_next_pred = torch.stack([
            known_map(u_obs[i], p, t[i]) + network(u_obs[i])
            for i in range(len(t) - 1)
        ])
        loss = loss_fn(u_next_pred, u_obs[1:])

        if torch.isnan(loss):
            if verbose:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)

        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_param_state = {k: v.clone() for k, v in param_dict.state_dict().items()}
            best_net_state = {k: v.clone() for k, v in network.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience and epochs_no_improve >= patience:
            param_dict.load_state_dict(best_param_state)
            network.load_state_dict(best_net_state)
            if verbose:
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.6f}")
            break

        if verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}")
