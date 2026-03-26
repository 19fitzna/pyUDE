"""Training routines for UDE models."""

from typing import TYPE_CHECKING, Dict, List, Optional

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


def _clamp_params(param_dict: nn.ParameterDict, param_bounds: dict) -> None:
    """Clamp mechanistic parameters to their specified bounds (in-place, no_grad)."""
    with torch.no_grad():
        for name, (lo, hi) in param_bounds.items():
            p = param_dict[name]
            if lo is not None and hi is not None:
                p.clamp_(lo, hi)
            elif lo is not None:
                p.clamp_(min=lo)
            elif hi is not None:
                p.clamp_(max=hi)


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
    noise_scale: float = 0.01,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    val_t: Optional[torch.Tensor] = None,
    val_u: Optional[torch.Tensor] = None,
    val_interval: int = 1,
    lambda_l1: float = 0.0,
    **kwargs,
) -> Dict[str, list]:
    """Train a continuous-time UDE model (NODE or CustomDerivatives).

    Returns
    -------
    dict with keys ``"train_loss"``, ``"val_loss"`` (empty if no val data),
    and ``"val_epochs"`` (empty if no val data).
    """
    ode_func = model._ode_func
    t, u_obs = model._get_training_tensors()
    device = model._device
    t = t.to(device)
    u_obs = u_obs.to(device)

    if val_t is not None:
        val_t = val_t.to(device)
        val_u = val_u.to(device)

    # Param bounds for CustomDerivatives (accessed via ode_func.params)
    param_bounds = getattr(model, '_param_bounds', None)

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
        return _train_simulation(
            ode_func, t, u_obs, optimizer, loss_fn, odeint,
            solver, epochs, log_interval, verbose, patience, max_grad_norm,
            rtol=rtol, atol=atol,
            val_t=val_t, val_u=val_u, val_interval=val_interval,
            lambda_l1=lambda_l1, param_bounds=param_bounds,
        )
    elif loss == "derivative_matching":
        return _train_derivative_matching(
            ode_func, t, u_obs, optimizer, loss_fn,
            epochs, log_interval, verbose, patience, max_grad_norm,
            noise_scale=noise_scale,
            val_t=val_t, val_u=val_u, val_interval=val_interval,
            lambda_l1=lambda_l1, param_bounds=param_bounds,
        )
    else:
        raise ValueError(f"Unknown loss '{loss}'. Choose 'simulation' or 'derivative_matching'.")


def _train_simulation(
    ode_func, t, u_obs, optimizer, loss_fn, odeint,
    solver, epochs, log_interval, verbose, patience, max_grad_norm,
    rtol=1e-3, atol=1e-6,
    val_t=None, val_u=None, val_interval=1,
    lambda_l1=0.0, param_bounds=None,
) -> Dict[str, list]:
    """Integrate ODE forward and compare to observations."""
    n_params = sum(1 for _ in ode_func.parameters())
    assert n_params > 0, "No trainable parameters found in ODE function"

    u0 = u_obs[0]
    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    val_epochs_history: List[int] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        u_pred = odeint(ode_func, u0, t, method=solver,
                        rtol=rtol, atol=atol,
                        adjoint_params=tuple(ode_func.parameters()))
        loss = loss_fn(u_pred, u_obs)

        if lambda_l1 > 0:
            l1_penalty = sum(p.abs().sum() for p in ode_func.parameters())
            loss = loss + lambda_l1 * l1_penalty

        if torch.isnan(loss):
            if verbose:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        if param_bounds and hasattr(ode_func, 'params'):
            _clamp_params(ode_func.params, param_bounds)

        loss_val = loss.item()
        train_loss_history.append(loss_val)

        # Validation loss
        val_loss_val = None
        if val_t is not None and epoch % val_interval == 0:
            with torch.no_grad():
                t_combined = torch.cat([t[-1:], val_t])
                u_extended = odeint(ode_func, u_obs[-1], t_combined, method=solver,
                                    rtol=rtol, atol=atol)
                u_val_pred = u_extended[1:]
                val_loss_val = loss_fn(u_val_pred, val_u).item()
            val_loss_history.append(val_loss_val)
            val_epochs_history.append(epoch)

        # Early stopping: use val loss if available this epoch, else train loss
        monitor = val_loss_val if val_t is not None else loss_val
        if monitor is not None and patience:
            if monitor < best_loss:
                best_loss = monitor
                best_state = {k: v.clone() for k, v in ode_func.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if best_state is not None:
                    ode_func.load_state_dict(best_state)
                if verbose:
                    label = "val" if val_t is not None else "train"
                    print(f"Early stopping at epoch {epoch}. Best {label} loss: {best_loss:.6f}")
                break

        if verbose and epoch % log_interval == 0:
            msg = f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}"
            if val_loss_val is not None:
                msg += f"  val_loss={val_loss_val:.6f}"
            print(msg)

    return {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "val_epochs": val_epochs_history,
    }


def _estimate_derivatives(t, u_obs):
    """Estimate du/dt using cubic spline interpolation.

    Falls back to central finite differences if scipy is unavailable.
    """
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
        # (t[2:] - t[:-2]) has shape (T-2,); unsqueeze(1) broadcasts over n_states
        dt = t[1:] - t[:-1]
        du_fd = torch.zeros_like(u_obs)
        du_fd[1:-1] = (u_obs[2:] - u_obs[:-2]) / (t[2:] - t[:-2]).unsqueeze(1)
        du_fd[0]    = (u_obs[1]  - u_obs[0])  / dt[0]
        du_fd[-1]   = (u_obs[-1] - u_obs[-2]) / dt[-1]
        return du_fd


def _train_derivative_matching(
    ode_func, t, u_obs, optimizer, loss_fn,
    epochs, log_interval, verbose, patience, max_grad_norm,
    noise_scale=0.01,
    val_t=None, val_u=None, val_interval=1,
    lambda_l1=0.0, param_bounds=None,
) -> Dict[str, list]:
    """Compare predicted derivatives against cubic-spline estimates.

    Injects Gaussian noise into training states to encourage generalisation
    beyond exact data points.
    """
    du_target = _estimate_derivatives(t, u_obs)

    val_du_target = None
    if val_t is not None:
        val_du_target = _estimate_derivatives(val_t, val_u)

    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    val_epochs_history: List[int] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Perturb observed states so the network generalizes beyond exact data points
        u_input = u_obs + noise_scale * torch.randn_like(u_obs)

        try:
            # Vectorised forward pass — ~T× faster than the Python loop below
            du_pred = torch.vmap(ode_func)(t, u_input)
        except Exception:
            # Fallback: known_dynamics uses control flow not supported by vmap
            du_pred = torch.stack([ode_func(t[i], u_input[i]) for i in range(len(t))])
        loss = loss_fn(du_pred, du_target)

        if lambda_l1 > 0:
            l1_penalty = sum(p.abs().sum() for p in ode_func.parameters())
            loss = loss + lambda_l1 * l1_penalty

        if torch.isnan(loss):
            if verbose:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        if param_bounds and hasattr(ode_func, 'params'):
            _clamp_params(ode_func.params, param_bounds)

        loss_val = loss.item()
        train_loss_history.append(loss_val)

        # Validation loss
        val_loss_val = None
        if val_t is not None and epoch % val_interval == 0:
            ode_func.eval()
            with torch.no_grad():
                try:
                    du_val_pred = torch.vmap(ode_func)(val_t, val_u)
                except Exception:
                    du_val_pred = torch.stack([ode_func(val_t[i], val_u[i]) for i in range(len(val_t))])
                val_loss_val = loss_fn(du_val_pred, val_du_target).item()
            ode_func.train()
            val_loss_history.append(val_loss_val)
            val_epochs_history.append(epoch)

        # Early stopping
        monitor = val_loss_val if val_t is not None else loss_val
        if monitor is not None and patience:
            if monitor < best_loss:
                best_loss = monitor
                best_state = {k: v.clone() for k, v in ode_func.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if best_state is not None:
                    ode_func.load_state_dict(best_state)
                if verbose:
                    label = "val" if val_t is not None else "train"
                    print(f"Early stopping at epoch {epoch}. Best {label} loss: {best_loss:.6f}")
                break

        if verbose and epoch % log_interval == 0:
            msg = f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}"
            if val_loss_val is not None:
                msg += f"  val_loss={val_loss_val:.6f}"
            print(msg)

    return {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "val_epochs": val_epochs_history,
    }


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
    weight_decay: float = 0.0,
    val_t: Optional[torch.Tensor] = None,
    val_u: Optional[torch.Tensor] = None,
    val_interval: int = 1,
    lambda_l1: float = 0.0,
) -> Dict[str, list]:
    """Train a discrete-time UDE model by minimising one-step-ahead MSE.

    Returns
    -------
    dict with keys ``"train_loss"``, ``"val_loss"``, ``"val_epochs"``.
    """
    t, u_obs = model._get_training_tensors()
    device = model._device
    t = t.to(device)
    u_obs = u_obs.to(device)

    if val_t is not None:
        val_t = val_t.to(device)
        val_u = val_u.to(device)

    param_dict = model._param_dict
    network = model._network_module
    known_map = model._known_map
    param_bounds = getattr(model, '_param_bounds', None)

    all_params = list(param_dict.parameters()) + list(network.parameters())
    optimizer = _make_optimizer(all_params, optimizer_name, learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_param_state = None
    best_net_state = None
    epochs_no_improve = 0

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    val_epochs_history: List[int] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        p = {k: v for k, v in param_dict.items()}
        u_next_pred = torch.stack([
            known_map(u_obs[i], p, t[i]) + network(u_obs[i])
            for i in range(len(t) - 1)
        ])
        loss = loss_fn(u_next_pred, u_obs[1:])

        if lambda_l1 > 0:
            l1_penalty = sum(p_.abs().sum() for p_ in all_params)
            loss = loss + lambda_l1 * l1_penalty

        if torch.isnan(loss):
            if verbose:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)

        optimizer.step()

        if param_bounds:
            _clamp_params(param_dict, param_bounds)

        loss_val = loss.item()
        train_loss_history.append(loss_val)

        # Validation loss
        val_loss_val = None
        if val_t is not None and epoch % val_interval == 0:
            with torch.no_grad():
                u_cur = u_obs[-1]
                preds = []
                p_val = {k: v for k, v in param_dict.items()}
                for i in range(len(val_u)):
                    u_next = known_map(u_cur, p_val, val_t[i]) + network(u_cur)
                    preds.append(u_next)
                    u_cur = u_next
                u_val_pred = torch.stack(preds)
                val_loss_val = loss_fn(u_val_pred, val_u).item()
            val_loss_history.append(val_loss_val)
            val_epochs_history.append(epoch)

        # Early stopping
        monitor = val_loss_val if val_t is not None else loss_val
        if monitor is not None and patience:
            if monitor < best_loss:
                best_loss = monitor
                best_param_state = {k: v.clone() for k, v in param_dict.state_dict().items()}
                best_net_state = {k: v.clone() for k, v in network.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if best_param_state is not None:
                    param_dict.load_state_dict(best_param_state)
                    network.load_state_dict(best_net_state)
                if verbose:
                    label = "val" if val_t is not None else "train"
                    print(f"Early stopping at epoch {epoch}. Best {label} loss: {best_loss:.6f}")
                break

        if verbose and epoch % log_interval == 0:
            msg = f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}"
            if val_loss_val is not None:
                msg += f"  val_loss={val_loss_val:.6f}"
            print(msg)

    return {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "val_epochs": val_epochs_history,
    }
