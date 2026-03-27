"""Training routines for UDE models."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pyUDE.core.base import UDEModel
    from pyUDE.core.custom_differences import CustomDifferences


@dataclass
class TrainResult:
    """Metadata returned after a training run.

    Attributes
    ----------
    loss_history : list[float]
        Loss value recorded at the end of each epoch.
    best_loss : float
        Lowest loss observed during training.
    best_epoch : int
        Epoch (1-indexed) at which ``best_loss`` was recorded.
    epochs_run : int
        Total number of epochs executed (may be less than requested if
        early stopping triggered or NaN was detected).
    stopped_early : bool
        ``True`` if training ended due to the early-stopping criterion.
    """

    loss_history: List[float] = field(default_factory=list)
    best_loss: float = float("inf")
    best_epoch: int = 0
    epochs_run: int = 0
    stopped_early: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer(params, name: str, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Choose 'adam' or 'sgd'.")


def _make_scheduler(optimizer, scheduler, epochs):
    """Build an LR scheduler from a string shorthand or return a user-supplied instance.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    scheduler : str, torch.optim.lr_scheduler._LRScheduler, or None
    epochs : int
    """
    if scheduler is None:
        return None
    if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
        return scheduler
    if not isinstance(scheduler, str):
        raise TypeError(f"scheduler must be a string or LRScheduler instance, got {type(scheduler)}")
    name = scheduler.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    raise ValueError(f"Unknown scheduler '{name}'. Choose 'cosine', 'plateau', or pass an LRScheduler.")


def _step_scheduler(scheduler, loss_val):
    """Step the scheduler, handling ReduceLROnPlateau specially."""
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(loss_val)
    else:
        scheduler.step()


def _epoch_iter(epochs, progress_bar, desc="Training"):
    """Return an iterable over epoch numbers, optionally wrapped with tqdm."""
    rng = range(1, epochs + 1)
    if not progress_bar:
        return rng
    try:
        from tqdm.auto import tqdm
        return tqdm(rng, desc=desc, unit="epoch")
    except ImportError:
        import warnings
        warnings.warn(
            "tqdm not installed; falling back to print logging. "
            "Install with: pip install tqdm",
            stacklevel=3,
        )
        return rng


def _batched_ode_call(ode_func, t_batch, u_batch):
    """Evaluate ode_func over a batch of (t, u) pairs, using vmap when possible."""
    try:
        from torch.func import vmap
        # Use functional_call through vmap for stateful modules
        params = dict(ode_func.named_parameters())
        buffers = dict(ode_func.named_buffers())

        def fn(t_i, u_i):
            return torch.func.functional_call(ode_func, (params, buffers), (t_i, u_i))

        return vmap(fn, in_dims=(0, 0))(t_batch, u_batch)
    except Exception:
        # Fallback: user-supplied known_dynamics may use operations
        # that vmap doesn't support (e.g., data-dependent control flow).
        return torch.stack([ode_func(t_batch[i], u_batch[i]) for i in range(len(t_batch))])


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
    scheduler: Optional[Union[str, torch.optim.lr_scheduler.LRScheduler]] = None,
    progress_bar: bool = False,
    **kwargs,
) -> TrainResult:
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
    scheduler : str or LRScheduler, optional
        Learning rate scheduler. ``"cosine"`` or ``"plateau"`` for built-in
        schedules, or pass a pre-built ``torch.optim.lr_scheduler`` instance.
    progress_bar : bool
        Show a tqdm progress bar instead of print-based logging.
    """
    ode_func = model._ode_func
    t, u_obs, _ = model._get_training_tensors()
    device = model._device
    t = t.to(device)
    u_obs = u_obs.to(device)

    optimizer = _make_optimizer(ode_func.parameters(), optimizer_name, learning_rate,
                                weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    sched = _make_scheduler(optimizer, scheduler, epochs)

    if loss == "simulation":
        try:
            from torchdiffeq import odeint_adjoint as odeint
        except ImportError as e:
            raise ImportError(
                "torchdiffeq is required for simulation loss. "
                "Install with: pip install torchdiffeq"
            ) from e
        return _train_simulation(ode_func, t, u_obs, optimizer, loss_fn, odeint,
                                 solver, epochs, log_interval, verbose, patience,
                                 max_grad_norm, sched, progress_bar)
    elif loss == "derivative_matching":
        return _train_derivative_matching(ode_func, t, u_obs, optimizer, loss_fn,
                                          epochs, log_interval, verbose, patience,
                                          max_grad_norm, sched, progress_bar)
    else:
        raise ValueError(f"Unknown loss '{loss}'. Choose 'simulation' or 'derivative_matching'.")


def _train_simulation(ode_func, t, u_obs, optimizer, loss_fn, odeint,
                      solver, epochs, log_interval, verbose, patience,
                      max_grad_norm, scheduler, progress_bar):
    """Integrate ODE forward and compare to observations."""
    n_params = sum(1 for _ in ode_func.parameters())
    assert n_params > 0, "No trainable parameters found in ODE function"

    u0 = u_obs[0]
    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    result = TrainResult()

    epoch_range = _epoch_iter(epochs, progress_bar, desc="Simulation")

    for epoch in epoch_range:
        optimizer.zero_grad()
        u_pred = odeint(ode_func, u0, t, method=solver, adjoint_params=tuple(ode_func.parameters()))
        loss = loss_fn(u_pred, u_obs)

        if torch.isnan(loss):
            if verbose and not progress_bar:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        _step_scheduler(scheduler, loss.item())

        loss_val = loss.item()
        result.loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in ode_func.state_dict().items()}
            result.best_loss = best_loss
            result.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience and epochs_no_improve >= patience:
            ode_func.load_state_dict(best_state)
            result.stopped_early = True
            if verbose and not progress_bar:
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.6f}")
            break

        if verbose and not progress_bar and epoch % log_interval == 0:
            print(f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}")

        if hasattr(epoch_range, 'set_postfix'):
            epoch_range.set_postfix(loss=f"{loss_val:.6f}")

    result.epochs_run = len(result.loss_history)
    return result


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
                                epochs, log_interval, verbose, patience,
                                max_grad_norm, scheduler, progress_bar,
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
    result = TrainResult()

    epoch_range = _epoch_iter(epochs, progress_bar, desc="Deriv. matching")

    for epoch in epoch_range:
        optimizer.zero_grad()

        # Perturb observed states so the network generalizes beyond exact data points
        u_input = u_obs + noise_scale * torch.randn_like(u_obs)

        du_pred = _batched_ode_call(ode_func, t, u_input)
        loss = loss_fn(du_pred, du_target)

        if torch.isnan(loss):
            if verbose and not progress_bar:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        _step_scheduler(scheduler, loss.item())

        loss_val = loss.item()
        result.loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in ode_func.state_dict().items()}
            result.best_loss = best_loss
            result.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience and epochs_no_improve >= patience:
            ode_func.load_state_dict(best_state)
            result.stopped_early = True
            if verbose and not progress_bar:
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.6f}")
            break

        if verbose and not progress_bar and epoch % log_interval == 0:
            print(f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}")

        if hasattr(epoch_range, 'set_postfix'):
            epoch_range.set_postfix(loss=f"{loss_val:.6f}")

    result.epochs_run = len(result.loss_history)
    return result


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
    scheduler: Optional[Union[str, torch.optim.lr_scheduler.LRScheduler]] = None,
    progress_bar: bool = False,
) -> TrainResult:
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
    sched = _make_scheduler(optimizer, scheduler, epochs)

    best_loss = float('inf')
    best_param_state = None
    best_net_state = None
    epochs_no_improve = 0
    result = TrainResult()

    # Pre-slice inputs for the one-step-ahead prediction
    u_in = u_obs[:-1]
    t_in = t[:-1]
    u_target = u_obs[1:]

    epoch_range = _epoch_iter(epochs, progress_bar, desc="Discrete")

    for epoch in epoch_range:
        optimizer.zero_grad()
        p = {k: v for k, v in param_dict.items()}

        # Try vectorised forward pass; fall back to loop if vmap can't trace
        try:
            from torch.func import vmap
            params = dict(network.named_parameters())
            buffers = dict(network.named_buffers())

            def step_fn(u_i, t_i):
                net_out = torch.func.functional_call(network, (params, buffers), (u_i,))
                return known_map(u_i, p, t_i) + net_out

            u_next_pred = vmap(step_fn, in_dims=(0, 0))(u_in, t_in)
        except Exception:
            u_next_pred = torch.stack([
                known_map(u_in[i], p, t_in[i]) + network(u_in[i])
                for i in range(len(u_in))
            ])

        loss = loss_fn(u_next_pred, u_target)

        if torch.isnan(loss):
            if verbose and not progress_bar:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)

        optimizer.step()
        _step_scheduler(sched, loss.item())

        loss_val = loss.item()
        result.loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_param_state = {k: v.clone() for k, v in param_dict.state_dict().items()}
            best_net_state = {k: v.clone() for k, v in network.state_dict().items()}
            result.best_loss = best_loss
            result.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience and epochs_no_improve >= patience:
            param_dict.load_state_dict(best_param_state)
            network.load_state_dict(best_net_state)
            result.stopped_early = True
            if verbose and not progress_bar:
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.6f}")
            break

        if verbose and not progress_bar and epoch % log_interval == 0:
            print(f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}")

        if hasattr(epoch_range, 'set_postfix'):
            epoch_range.set_postfix(loss=f"{loss_val:.6f}")

    result.epochs_run = len(result.loss_history)
    return result
