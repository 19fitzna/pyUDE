"""Training routines for UDE models."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import math

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
        Training loss recorded at the end of each epoch.
    val_loss_history : list[float]
        Validation loss (empty when no validation data is provided).
    val_epochs : list[int]
        Epochs at which validation loss was recorded.
    best_loss : float
        Lowest monitored loss observed during training.
    best_epoch : int
        Epoch (1-indexed) at which ``best_loss`` was recorded.
    epochs_run : int
        Total number of epochs executed (may be less than requested if
        early stopping triggered or NaN was detected).
    stopped_early : bool
        ``True`` if training ended due to the early-stopping criterion.
    """

    loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    val_epochs: List[int] = field(default_factory=list)
    best_loss: float = float("inf")
    best_epoch: int = 0
    epochs_run: int = 0
    stopped_early: bool = False

    def to_dict(self) -> Dict[str, list]:
        """Convert to the dict format used by ``_merge_history``."""
        return {
            "train_loss": list(self.loss_history),
            "val_loss": list(self.val_loss_history),
            "val_epochs": list(self.val_epochs),
        }


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


def _make_scheduler(optimizer, scheduler, epochs):
    """Build an LR scheduler from a string shorthand or return a user-supplied instance.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    scheduler : str, torch.optim.lr_scheduler.LRScheduler, or None
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
    noise_scale: float = 0.01,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    val_t: Optional[torch.Tensor] = None,
    val_u: Optional[torch.Tensor] = None,
    val_interval: int = 1,
    lambda_l1: float = 0.0,
    scheduler: Optional[Union[str, torch.optim.lr_scheduler.LRScheduler]] = None,
    progress_bar: bool = False,
    n_shooting_segments: int = 10,
    pred_length: Optional[int] = None,
    proc_weight: float = 1.0,
    obs_weight: float = 1.0,
    obs_cov: Optional[torch.Tensor] = None,
    proc_cov: Optional[torch.Tensor] = None,
    **kwargs,
) -> TrainResult:
    """Train a continuous-time UDE model (NODE or CustomDerivatives).

    Returns
    -------
    TrainResult with loss history, validation loss, best epoch, etc.
    """
    ode_func = model._ode_func
    t, u_obs = model._get_training_tensors()
    device = model._device
    t = t.to(device)
    u_obs = u_obs.to(device)

    if val_t is not None:
        val_t = val_t.to(device)
        val_u = val_u.to(device)

    if obs_cov is not None:
        obs_cov = obs_cov.to(device)
    if proc_cov is not None:
        proc_cov = proc_cov.to(device)

    # Param bounds for CustomDerivatives (accessed via ode_func.params)
    param_bounds = getattr(model, '_param_bounds', None)

    loss_fn = nn.MSELoss()

    valid_losses = ("simulation", "derivative_matching", "multiple_shooting",
                    "conditional_likelihood")

    if loss == "simulation":
        optimizer = _make_optimizer(ode_func.parameters(), optimizer_name,
                                    learning_rate, weight_decay=weight_decay)
        sched = _make_scheduler(optimizer, scheduler, epochs)
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
            sched, progress_bar,
            rtol=rtol, atol=atol,
            val_t=val_t, val_u=val_u, val_interval=val_interval,
            lambda_l1=lambda_l1, param_bounds=param_bounds,
            obs_weight=obs_weight,
        )
    elif loss == "derivative_matching":
        optimizer = _make_optimizer(ode_func.parameters(), optimizer_name,
                                    learning_rate, weight_decay=weight_decay)
        sched = _make_scheduler(optimizer, scheduler, epochs)
        return _train_derivative_matching(
            ode_func, t, u_obs, optimizer, loss_fn,
            epochs, log_interval, verbose, patience, max_grad_norm,
            sched, progress_bar,
            noise_scale=noise_scale,
            val_t=val_t, val_u=val_u, val_interval=val_interval,
            lambda_l1=lambda_l1, param_bounds=param_bounds,
        )
    elif loss == "multiple_shooting":
        try:
            from torchdiffeq import odeint_adjoint as odeint
        except ImportError as e:
            raise ImportError(
                "torchdiffeq is required for multiple shooting loss. "
                "Install with: pip install torchdiffeq"
            ) from e
        # Resolve n_segments from pred_length if provided
        if pred_length is not None:
            n_segments = max(1, math.ceil(len(t) / pred_length))
        else:
            n_segments = n_shooting_segments
        return _train_multiple_shooting(
            ode_func, t, u_obs, optimizer_name, learning_rate, weight_decay,
            loss_fn, odeint,
            solver, epochs, log_interval, verbose, patience, max_grad_norm,
            scheduler, progress_bar,
            n_segments=n_segments,
            proc_weight=proc_weight, obs_weight=obs_weight,
            rtol=rtol, atol=atol,
            val_t=val_t, val_u=val_u, val_interval=val_interval,
            lambda_l1=lambda_l1, param_bounds=param_bounds,
        )
    elif loss == "conditional_likelihood":
        if obs_cov is None or proc_cov is None:
            raise ValueError(
                "conditional_likelihood requires both obs_covariance and "
                "proc_covariance. Set them on the model constructor or pass "
                "them to train()."
            )
        optimizer = _make_optimizer(ode_func.parameters(), optimizer_name,
                                    learning_rate, weight_decay=weight_decay)
        sched = _make_scheduler(optimizer, scheduler, epochs)
        try:
            from torchdiffeq import odeint_adjoint as odeint
        except ImportError as e:
            raise ImportError(
                "torchdiffeq is required for conditional likelihood loss. "
                "Install with: pip install torchdiffeq"
            ) from e
        result = _train_conditional_likelihood(
            model, ode_func, t, u_obs, optimizer, odeint,
            solver, epochs, log_interval, verbose, patience, max_grad_norm,
            sched, progress_bar,
            obs_cov=obs_cov, proc_cov=proc_cov,
            proc_weight=proc_weight, obs_weight=obs_weight,
            rtol=rtol, atol=atol,
            val_t=val_t, val_u=val_u, val_interval=val_interval,
            lambda_l1=lambda_l1, param_bounds=param_bounds,
        )
        return result
    else:
        raise ValueError(
            f"Unknown loss '{loss}'. Choose from: {valid_losses}"
        )


def _train_simulation(
    ode_func, t, u_obs, optimizer, loss_fn, odeint,
    solver, epochs, log_interval, verbose, patience, max_grad_norm,
    scheduler, progress_bar,
    rtol=1e-3, atol=1e-6,
    val_t=None, val_u=None, val_interval=1,
    lambda_l1=0.0, param_bounds=None,
    obs_weight=1.0,
) -> TrainResult:
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
        u_pred = odeint(ode_func, u0, t, method=solver,
                        rtol=rtol, atol=atol,
                        adjoint_params=tuple(ode_func.parameters()))
        loss = obs_weight * loss_fn(u_pred, u_obs)

        if lambda_l1 > 0:
            l1_penalty = sum(p.abs().sum() for p in ode_func.parameters())
            loss = loss + lambda_l1 * l1_penalty

        if torch.isnan(loss):
            if verbose and not progress_bar:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        _step_scheduler(scheduler, loss.item())

        if param_bounds and hasattr(ode_func, 'params'):
            _clamp_params(ode_func.params, param_bounds)

        loss_val = loss.item()
        result.loss_history.append(loss_val)

        # Validation loss
        val_loss_val = None
        if val_t is not None and epoch % val_interval == 0:
            with torch.no_grad():
                t_combined = torch.cat([t[-1:], val_t])
                u_extended = odeint(ode_func, u_obs[-1], t_combined, method=solver,
                                    rtol=rtol, atol=atol)
                u_val_pred = u_extended[1:]
                val_loss_val = loss_fn(u_val_pred, val_u).item()
            result.val_loss_history.append(val_loss_val)
            result.val_epochs.append(epoch)

        # Early stopping: use val loss if available this epoch, else train loss
        monitor = val_loss_val if val_t is not None else loss_val
        if monitor is not None and patience:
            if monitor < best_loss:
                best_loss = monitor
                best_state = {k: v.clone() for k, v in ode_func.state_dict().items()}
                result.best_loss = best_loss
                result.best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if best_state is not None:
                    ode_func.load_state_dict(best_state)
                result.stopped_early = True
                if verbose and not progress_bar:
                    label = "val" if val_t is not None else "train"
                    print(f"Early stopping at epoch {epoch}. Best {label} loss: {best_loss:.6f}")
                break
        elif not patience:
            # Track best even without early stopping
            if loss_val < best_loss:
                best_loss = loss_val
                result.best_loss = best_loss
                result.best_epoch = epoch

        if verbose and not progress_bar and epoch % log_interval == 0:
            msg = f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}"
            if val_loss_val is not None:
                msg += f"  val_loss={val_loss_val:.6f}"
            print(msg)

        if hasattr(epoch_range, 'set_postfix'):
            epoch_range.set_postfix(loss=f"{loss_val:.6f}")

    result.epochs_run = len(result.loss_history)
    return result


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
    scheduler, progress_bar,
    noise_scale=0.01,
    val_t=None, val_u=None, val_interval=1,
    lambda_l1=0.0, param_bounds=None,
) -> TrainResult:
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
    result = TrainResult()

    epoch_range = _epoch_iter(epochs, progress_bar, desc="Deriv. matching")

    for epoch in epoch_range:
        optimizer.zero_grad()

        # Perturb observed states so the network generalizes beyond exact data points
        u_input = u_obs + noise_scale * torch.randn_like(u_obs)

        du_pred = _batched_ode_call(ode_func, t, u_input)
        loss = loss_fn(du_pred, du_target)

        if lambda_l1 > 0:
            l1_penalty = sum(p.abs().sum() for p in ode_func.parameters())
            loss = loss + lambda_l1 * l1_penalty

        if torch.isnan(loss):
            if verbose and not progress_bar:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        _step_scheduler(scheduler, loss.item())

        if param_bounds and hasattr(ode_func, 'params'):
            _clamp_params(ode_func.params, param_bounds)

        loss_val = loss.item()
        result.loss_history.append(loss_val)

        # Validation loss
        val_loss_val = None
        if val_t is not None and epoch % val_interval == 0:
            ode_func.eval()
            with torch.no_grad():
                du_val_pred = _batched_ode_call(ode_func, val_t, val_u)
                val_loss_val = loss_fn(du_val_pred, val_du_target).item()
            ode_func.train()
            result.val_loss_history.append(val_loss_val)
            result.val_epochs.append(epoch)

        # Early stopping
        monitor = val_loss_val if val_t is not None else loss_val
        if monitor is not None and patience:
            if monitor < best_loss:
                best_loss = monitor
                best_state = {k: v.clone() for k, v in ode_func.state_dict().items()}
                result.best_loss = best_loss
                result.best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if best_state is not None:
                    ode_func.load_state_dict(best_state)
                result.stopped_early = True
                if verbose and not progress_bar:
                    label = "val" if val_t is not None else "train"
                    print(f"Early stopping at epoch {epoch}. Best {label} loss: {best_loss:.6f}")
                break
        elif not patience:
            if loss_val < best_loss:
                best_loss = loss_val
                result.best_loss = best_loss
                result.best_epoch = epoch

        if verbose and not progress_bar and epoch % log_interval == 0:
            msg = f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}"
            if val_loss_val is not None:
                msg += f"  val_loss={val_loss_val:.6f}"
            print(msg)

        if hasattr(epoch_range, 'set_postfix'):
            epoch_range.set_postfix(loss=f"{loss_val:.6f}")

    result.epochs_run = len(result.loss_history)
    return result


# ---------------------------------------------------------------------------
# Multiple shooting
# ---------------------------------------------------------------------------

def _smooth_trajectory(t, u_obs):
    """Return a spline-smoothed trajectory evaluated at times ``t``.

    Falls back to the raw observations if scipy is unavailable.
    """
    try:
        from scipy.interpolate import CubicSpline
        t_np = t.detach().cpu().numpy()
        u_np = u_obs.detach().cpu().numpy()
        smoothed = np.zeros_like(u_np)
        for col in range(u_np.shape[1]):
            cs = CubicSpline(t_np, u_np[:, col])
            smoothed[:, col] = cs(t_np)
        return torch.tensor(smoothed, dtype=u_obs.dtype, device=u_obs.device)
    except ImportError:
        return u_obs.clone()


def _train_multiple_shooting(
    ode_func, t, u_obs,
    optimizer_name, learning_rate, weight_decay,
    loss_fn, odeint,
    solver, epochs, log_interval, verbose, patience, max_grad_norm,
    scheduler_spec, progress_bar,
    n_segments=10,
    proc_weight=1.0, obs_weight=1.0,
    rtol=1e-3, atol=1e-6,
    val_t=None, val_u=None, val_interval=1,
    lambda_l1=0.0, param_bounds=None,
) -> TrainResult:
    """Divide trajectory into segments with learnable initial conditions.

    Each segment is integrated independently.  The total loss combines
    observation fit and continuity (segments must connect).  Continuity
    loss is normalised by ``n_segments`` to prevent domination.
    """
    n_params = sum(1 for _ in ode_func.parameters())
    assert n_params > 0, "No trainable parameters found in ODE function"

    T = len(t)
    n_segments = min(n_segments, T - 1)  # can't have more segments than gaps

    # --- build segments ------------------------------------------------
    # Each segment is a contiguous slice [start, end) of the time series.
    seg_size = max(2, T // n_segments)
    seg_starts = list(range(0, T, seg_size))
    if seg_starts[-1] >= T - 1:
        seg_starts = seg_starts[:-1]  # last segment must have ≥2 points
    segments = []
    for i, s in enumerate(seg_starts):
        e = seg_starts[i + 1] if i + 1 < len(seg_starts) else T
        segments.append((s, e))

    n_seg = len(segments)

    # --- shooting initial conditions -----------------------------------
    # Initialise from cubic-spline-smoothed trajectory for robustness on
    # noisy data; fallback to raw observations if scipy unavailable.
    u_smooth = _smooth_trajectory(t, u_obs)
    shooting_x0 = nn.ParameterList([
        nn.Parameter(u_smooth[s].clone().detach())
        for s, _e in segments
    ])

    # --- optimizer (includes shooting parameters) ----------------------
    all_params = list(ode_func.parameters()) + list(shooting_x0.parameters())
    optimizer = _make_optimizer(all_params, optimizer_name, learning_rate,
                                weight_decay=weight_decay)
    sched = _make_scheduler(optimizer, scheduler_spec, epochs)

    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    result = TrainResult()

    epoch_range = _epoch_iter(epochs, progress_bar, desc="Multi-shoot")

    for epoch in epoch_range:
        optimizer.zero_grad()

        total_obs_loss = torch.tensor(0.0, dtype=torch.float64, device=t.device)
        total_cont_loss = torch.tensor(0.0, dtype=torch.float64, device=t.device)

        for k, (s, e) in enumerate(segments):
            t_seg = t[s:e]
            u_seg = u_obs[s:e]
            x0_k = shooting_x0[k]

            u_pred_k = odeint(ode_func, x0_k, t_seg, method=solver,
                              rtol=rtol, atol=atol,
                              adjoint_params=tuple(ode_func.parameters()) + (x0_k,))

            # Observation loss for this segment
            total_obs_loss = total_obs_loss + loss_fn(u_pred_k, u_seg)

            # Continuity loss: end of this segment must match start of next
            if k + 1 < n_seg:
                x0_next = shooting_x0[k + 1]
                total_cont_loss = total_cont_loss + loss_fn(
                    u_pred_k[-1].unsqueeze(0), x0_next.unsqueeze(0)
                )

        loss = obs_weight * total_obs_loss + (proc_weight / n_seg) * total_cont_loss

        if lambda_l1 > 0:
            l1_penalty = sum(p.abs().sum() for p in ode_func.parameters())
            loss = loss + lambda_l1 * l1_penalty

        if torch.isnan(loss):
            if verbose and not progress_bar:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)

        optimizer.step()
        _step_scheduler(sched, loss.item())

        if param_bounds and hasattr(ode_func, 'params'):
            _clamp_params(ode_func.params, param_bounds)

        loss_val = loss.item()
        result.loss_history.append(loss_val)

        # Validation loss (single-shooting from last training point)
        val_loss_val = None
        if val_t is not None and epoch % val_interval == 0:
            with torch.no_grad():
                t_combined = torch.cat([t[-1:], val_t])
                u_extended = odeint(ode_func, u_obs[-1], t_combined, method=solver,
                                    rtol=rtol, atol=atol)
                u_val_pred = u_extended[1:]
                val_loss_val = loss_fn(u_val_pred, val_u).item()
            result.val_loss_history.append(val_loss_val)
            result.val_epochs.append(epoch)

        # Early stopping
        monitor = val_loss_val if val_t is not None else loss_val
        if monitor is not None and patience:
            if monitor < best_loss:
                best_loss = monitor
                best_state = {k: v.clone() for k, v in ode_func.state_dict().items()}
                result.best_loss = best_loss
                result.best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if best_state is not None:
                    ode_func.load_state_dict(best_state)
                result.stopped_early = True
                if verbose and not progress_bar:
                    label = "val" if val_t is not None else "train"
                    print(f"Early stopping at epoch {epoch}. Best {label} loss: {best_loss:.6f}")
                break
        elif not patience:
            if loss_val < best_loss:
                best_loss = loss_val
                result.best_loss = best_loss
                result.best_epoch = epoch

        if verbose and not progress_bar and epoch % log_interval == 0:
            msg = f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}"
            if val_loss_val is not None:
                msg += f"  val_loss={val_loss_val:.6f}"
            print(msg)

        if hasattr(epoch_range, 'set_postfix'):
            epoch_range.set_postfix(loss=f"{loss_val:.6f}")

    result.epochs_run = len(result.loss_history)
    return result


# ---------------------------------------------------------------------------
# Conditional likelihood (Extended Kalman Filter)
# ---------------------------------------------------------------------------

def _train_conditional_likelihood(
    model, ode_func, t, u_obs, optimizer, odeint,
    solver, epochs, log_interval, verbose, patience, max_grad_norm,
    scheduler, progress_bar,
    obs_cov, proc_cov,
    proc_weight=1.0, obs_weight=1.0,
    rtol=1e-3, atol=1e-6,
    val_t=None, val_u=None, val_interval=1,
    lambda_l1=0.0, param_bounds=None,
) -> TrainResult:
    """EKF-based conditional likelihood loss.

    At each time step the Extended Kalman Filter predicts the state via ODE
    integration, computes the innovation (observation minus prediction), and
    updates using the Kalman gain.  The loss is the negative log-likelihood
    of the innovations under the predicted covariance.

    Computational cost is O(T * n_states^3) per epoch due to Jacobian and
    matrix operations.  For high-dimensional systems (50+ states) prefer
    ``multiple_shooting`` or ``simulation`` loss.
    """
    n_params = sum(1 for _ in ode_func.parameters())
    assert n_params > 0, "No trainable parameters found in ODE function"

    T = len(t)
    n = u_obs.shape[1]
    device = t.device

    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    result = TrainResult()

    epoch_range = _epoch_iter(epochs, progress_bar, desc="Cond. likelihood")

    I_n = torch.eye(n, dtype=torch.float64, device=device)

    for epoch in epoch_range:
        optimizer.zero_grad()

        # Initial state estimate and covariance
        x_hat = u_obs[0].clone()
        P = obs_cov.clone()  # P_0 = Σ_obs

        total_nll = torch.tensor(0.0, dtype=torch.float64, device=device)
        state_estimates = [x_hat.detach().clone()]

        for k in range(1, T):
            # 1. Predict: integrate ODE from t_{k-1} to t_k
            t_seg = torch.stack([t[k - 1], t[k]])
            x_pred_traj = odeint(ode_func, x_hat, t_seg, method=solver,
                                 rtol=rtol, atol=atol,
                                 adjoint_params=tuple(ode_func.parameters()))
            x_pred = x_pred_traj[-1]  # predicted state at t_k

            # 2. Jacobian of ode_func w.r.t. state (for covariance propagation)
            with torch.enable_grad():
                x_for_jac = x_hat.detach().requires_grad_(True)
                F_k = torch.autograd.functional.jacobian(
                    lambda x: ode_func(t[k - 1], x), x_for_jac,
                    vectorize=True,
                )

            # Approximate state transition matrix: I + F * dt
            dt_k = t[k] - t[k - 1]
            Phi_k = I_n + F_k * dt_k

            # 3. Covariance predict
            P_pred = Phi_k @ P @ Phi_k.T + proc_cov

            # 4. Innovation
            v_k = u_obs[k] - x_pred

            # 5. Innovation covariance
            S_k = P_pred + obs_cov

            # 6. Kalman gain via solve (numerically stable)
            K_k = torch.linalg.solve(S_k.T, P_pred.T).T

            # 7. State update
            x_hat = x_pred + K_k @ v_k

            # 8. Covariance update (Joseph form for numerical stability)
            IKH = I_n - K_k
            P = IKH @ P_pred @ IKH.T + K_k @ obs_cov @ K_k.T

            # 9. Negative log-likelihood contribution
            # -log N(v; 0, S) = 0.5 * (v^T S^{-1} v + log|S|) + const
            v_Sinv_v = v_k @ torch.linalg.solve(S_k, v_k)
            _, log_det_S = torch.linalg.slogdet(S_k)
            nll_k = 0.5 * (v_Sinv_v + log_det_S)

            total_nll = total_nll + nll_k
            state_estimates.append(x_hat.detach().clone())

        loss = obs_weight * total_nll

        if lambda_l1 > 0:
            l1_penalty = sum(p.abs().sum() for p in ode_func.parameters())
            loss = loss + lambda_l1 * l1_penalty

        if torch.isnan(loss):
            if verbose and not progress_bar:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        _step_scheduler(scheduler, loss.item())

        if param_bounds and hasattr(ode_func, 'params'):
            _clamp_params(ode_func.params, param_bounds)

        loss_val = loss.item()
        result.loss_history.append(loss_val)

        # Validation loss (simulation-based for simplicity)
        val_loss_val = None
        if val_t is not None and epoch % val_interval == 0:
            with torch.no_grad():
                t_combined = torch.cat([t[-1:], val_t])
                u_extended = odeint(ode_func, u_obs[-1], t_combined, method=solver,
                                    rtol=rtol, atol=atol)
                u_val_pred = u_extended[1:]
                val_loss_val = nn.MSELoss()(u_val_pred, val_u).item()
            result.val_loss_history.append(val_loss_val)
            result.val_epochs.append(epoch)

        # Early stopping
        monitor = val_loss_val if val_t is not None else loss_val
        if monitor is not None and patience:
            if monitor < best_loss:
                best_loss = monitor
                best_state = {k_: v.clone() for k_, v in ode_func.state_dict().items()}
                result.best_loss = best_loss
                result.best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if best_state is not None:
                    ode_func.load_state_dict(best_state)
                result.stopped_early = True
                if verbose and not progress_bar:
                    label = "val" if val_t is not None else "train"
                    print(f"Early stopping at epoch {epoch}. Best {label} loss: {best_loss:.6f}")
                break
        elif not patience:
            if loss_val < best_loss:
                best_loss = loss_val
                result.best_loss = best_loss
                result.best_epoch = epoch

        if verbose and not progress_bar and epoch % log_interval == 0:
            msg = f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}"
            if val_loss_val is not None:
                msg += f"  val_loss={val_loss_val:.6f}"
            print(msg)

        if hasattr(epoch_range, 'set_postfix'):
            epoch_range.set_postfix(loss=f"{loss_val:.6f}")

    # Store final state estimates on the model
    model._state_estimates = torch.stack(state_estimates)

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
    weight_decay: float = 0.0,
    val_t: Optional[torch.Tensor] = None,
    val_u: Optional[torch.Tensor] = None,
    val_interval: int = 1,
    lambda_l1: float = 0.0,
    scheduler: Optional[Union[str, torch.optim.lr_scheduler.LRScheduler]] = None,
    progress_bar: bool = False,
    obs_weight: float = 1.0,
) -> TrainResult:
    """Train a discrete-time UDE model by minimising one-step-ahead MSE.

    Returns
    -------
    TrainResult with loss history, validation loss, best epoch, etc.
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

        loss = obs_weight * loss_fn(u_next_pred, u_target)

        if lambda_l1 > 0:
            l1_penalty = sum(p_.abs().sum() for p_ in all_params)
            loss = loss + lambda_l1 * l1_penalty

        if torch.isnan(loss):
            if verbose and not progress_bar:
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            break

        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)

        optimizer.step()
        _step_scheduler(sched, loss.item())

        if param_bounds:
            _clamp_params(param_dict, param_bounds)

        loss_val = loss.item()
        result.loss_history.append(loss_val)

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
            result.val_loss_history.append(val_loss_val)
            result.val_epochs.append(epoch)

        # Early stopping
        monitor = val_loss_val if val_t is not None else loss_val
        if monitor is not None and patience:
            if monitor < best_loss:
                best_loss = monitor
                best_param_state = {k: v.clone() for k, v in param_dict.state_dict().items()}
                best_net_state = {k: v.clone() for k, v in network.state_dict().items()}
                result.best_loss = best_loss
                result.best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if best_param_state is not None:
                    param_dict.load_state_dict(best_param_state)
                    network.load_state_dict(best_net_state)
                result.stopped_early = True
                if verbose and not progress_bar:
                    label = "val" if val_t is not None else "train"
                    print(f"Early stopping at epoch {epoch}. Best {label} loss: {best_loss:.6f}")
                break
        elif not patience:
            if loss_val < best_loss:
                best_loss = loss_val
                result.best_loss = best_loss
                result.best_epoch = epoch

        if verbose and not progress_bar and epoch % log_interval == 0:
            msg = f"Epoch {epoch:>5d}/{epochs}  loss={loss_val:.6f}"
            if val_loss_val is not None:
                msg += f"  val_loss={val_loss_val:.6f}"
            print(msg)

        if hasattr(epoch_range, 'set_postfix'):
            epoch_range.set_postfix(loss=f"{loss_val:.6f}")

    result.epochs_run = len(result.loss_history)
    return result
