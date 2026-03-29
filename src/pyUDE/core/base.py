from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pyUDE.utils.validation import validate_dataframe
from pyUDE.utils.data import dataframe_to_tensors, tensors_to_dataframe


def _normalize_covariance(
    cov: Optional[Union[float, torch.Tensor]],
    n: int,
) -> Optional[torch.Tensor]:
    """Normalize a covariance specification to an ``(n, n)`` matrix.

    Parameters
    ----------
    cov : float, 1-D Tensor, 2-D Tensor, or None
        * ``None`` → ``None``
        * scalar float → ``cov * I_n``  (interpreted as **variance**)
        * 1-D Tensor of length *n* → ``diag(cov)``  (elements are **variances**)
        * 2-D Tensor of shape *(n, n)* → returned as-is
    n : int
        State dimension.
    """
    if cov is None:
        return None
    if isinstance(cov, (int, float)):
        return float(cov) * torch.eye(n, dtype=torch.float64)
    if not isinstance(cov, torch.Tensor):
        raise TypeError(
            f"Covariance must be a float, 1-D tensor, or 2-D tensor, got {type(cov).__name__}"
        )
    cov = cov.to(dtype=torch.float64)
    if cov.ndim == 1:
        if cov.shape[0] != n:
            raise ValueError(
                f"1-D covariance vector has length {cov.shape[0]}, expected {n}"
            )
        return torch.diag(cov)
    if cov.ndim == 2:
        if cov.shape != (n, n):
            raise ValueError(
                f"2-D covariance matrix has shape {tuple(cov.shape)}, expected ({n}, {n})"
            )
        return cov
    raise ValueError(f"Covariance tensor must be 1-D or 2-D, got {cov.ndim}-D")


class UDEModel(ABC):
    """
    Abstract base class for all Universal Differential Equation models.

    Subclasses implement ``_build_ode_func`` to define the right-hand side
    of the differential equation as a callable ``f(t, u) -> du``.

    Parameters
    ----------
    data : pd.DataFrame
        Training data with a time column and state columns.
    time_column : str
        Name of the time column.
    device : str
        PyTorch device.
    observation_error : float, Tensor, or None
        Observation noise covariance Σ_obs.  Scalar and 1-D inputs are
        interpreted as **variances** (diagonal elements), not standard
        deviations.  Required for ``loss="conditional_likelihood"``.
    process_error : float, Tensor, or None
        Process noise covariance Σ_proc.  Same conventions as
        ``observation_error``.
    proc_weight : float
        Weight for process-model terms in state-space losses (default 1.0).
    obs_weight : float
        Weight for observation terms in state-space losses (default 1.0).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_column: str = "time",
        device: str = "cpu",
        observation_error: Optional[Union[float, torch.Tensor]] = None,
        process_error: Optional[Union[float, torch.Tensor]] = None,
        proc_weight: float = 1.0,
        obs_weight: float = 1.0,
    ):
        validate_dataframe(data, time_column)
        self._data = data
        self._time_column = time_column
        self._state_columns: List[str] = [c for c in data.columns if c != time_column]
        self._n_states: int = len(self._state_columns)
        self._is_trained: bool = False
        self._device: torch.device = torch.device(device)

        # Noise covariances
        self._obs_err = _normalize_covariance(observation_error, self._n_states)
        self._proc_err = _normalize_covariance(process_error, self._n_states)
        self._proc_weight: float = float(proc_weight)
        self._obs_weight: float = float(obs_weight)

        # Populated by train()
        self._ode_func: Optional[nn.Module] = None
        self._solver: Optional[str] = None
        self._state_estimates: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_ode_func(self) -> nn.Module:
        """Return an nn.Module whose forward(t, u) gives du/dt."""

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        loss: str = "simulation",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 500,
        log_interval: int = 50,
        verbose: bool = True,
        solver: str = "dopri5",
        patience: Optional[int] = None,
        max_grad_norm: float = 10.0,
        weight_decay: Optional[float] = None,
        noise_scale: float = 0.01,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        val_data: Optional[pd.DataFrame] = None,
        val_interval: int = 1,
        lambda_l1: float = 0.0,
        scheduler: Optional[Union[str, torch.optim.lr_scheduler.LRScheduler]] = None,
        progress_bar: bool = False,
        n_shooting_segments: int = 10,
        pred_length: Optional[int] = None,
        proc_weight: Optional[float] = None,
        obs_weight: Optional[float] = None,
        **kwargs,
    ) -> "UDEModel":
        """
        Fit the model to the training data.

        Parameters
        ----------
        loss : {"simulation", "derivative_matching", "multiple_shooting", "conditional_likelihood"}
            Loss function.

            * ``"simulation"`` — integrates the ODE forward and compares to
              observations (single shooting).
            * ``"derivative_matching"`` — compares predicted derivatives
              against cubic-spline estimates.
            * ``"multiple_shooting"`` — divides the trajectory into segments
              with learnable initial conditions; combines observation and
              continuity losses.
            * ``"conditional_likelihood"`` — Extended Kalman Filter-based
              loss that separates observation noise from process noise.
              Requires ``observation_error`` and ``process_error``.
        optimizer : str
            Optimizer name (``"adam"`` or ``"sgd"``).
        learning_rate : float
        epochs : int
        log_interval : int
            Print loss every this many epochs when ``verbose=True``.
        verbose : bool
        solver : str
            ODE solver passed to torchdiffeq (e.g. ``"dopri5"``, ``"rk4"``).
        patience : int, optional
            Stop if loss does not improve for this many epochs (early stopping).
            Uses validation loss when ``val_data`` is provided.
        max_grad_norm : float
            Maximum norm for gradient clipping. Set to 0 to disable.
        weight_decay : float, optional
            L2 regularisation. Defaults to ``1e-4`` for derivative_matching
            and ``0.0`` for simulation.
        noise_scale : float
            Standard deviation of Gaussian noise injected into training states
            during derivative matching. Only used when ``loss="derivative_matching"``.
        rtol : float
            Relative tolerance for the adaptive ODE solver.
        atol : float
            Absolute tolerance for the adaptive ODE solver.
        val_data : pd.DataFrame, optional
            Held-out validation set.
        val_interval : int
            Compute validation loss every this many epochs. Default ``1``.
        lambda_l1 : float
            L1 penalty coefficient. Default ``0.0`` (disabled).
        scheduler : str or LRScheduler, optional
            Learning rate scheduler.
        progress_bar : bool
            Show a tqdm progress bar instead of print-based logging.
        n_shooting_segments : int
            Number of segments for ``"multiple_shooting"`` loss. Default 10.
        pred_length : int, optional
            Time points per segment for ``"multiple_shooting"``. If provided,
            takes precedence over ``n_shooting_segments``.
        proc_weight : float, optional
            Process-model weight. Overrides the model-level ``proc_weight``
            for this training run.
        obs_weight : float, optional
            Observation weight. Overrides the model-level ``obs_weight``
            for this training run.

        Returns
        -------
        self (fluent interface)
        """
        from pyUDE.training.trainer import train_model

        # Clear stale state estimates from a previous training run
        self._state_estimates = None

        if weight_decay is None:
            weight_decay = 1e-4 if loss == "derivative_matching" else 0.0

        # Resolve per-call vs model-level weights
        eff_proc_weight = proc_weight if proc_weight is not None else self._proc_weight
        eff_obs_weight = obs_weight if obs_weight is not None else self._obs_weight

        val_t = val_u = None
        if val_data is not None:
            validate_dataframe(val_data, self._time_column)
            val_t, val_u, _ = dataframe_to_tensors(val_data, self._time_column)

        if self._ode_func is None:
            self._ode_func = self._build_ode_func()

        self._ode_func = self._ode_func.to(self._device)
        self._solver = solver

        self.train_result_ = train_model(
            model=self,
            loss=loss,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            epochs=epochs,
            log_interval=log_interval,
            verbose=verbose,
            solver=solver,
            patience=patience,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            noise_scale=noise_scale,
            rtol=rtol,
            atol=atol,
            val_t=val_t,
            val_u=val_u,
            val_interval=val_interval,
            lambda_l1=lambda_l1,
            scheduler=scheduler,
            progress_bar=progress_bar,
            n_shooting_segments=n_shooting_segments,
            pred_length=pred_length,
            proc_weight=eff_proc_weight,
            obs_weight=eff_obs_weight,
            obs_cov=self._obs_err,
            proc_cov=self._proc_err,
            **kwargs,
        )
        self._is_trained = True
        self._merge_history(self.train_result_.to_dict())
        return self

    def _merge_history(self, history: Dict[str, list]) -> None:
        """Append a training history dict into ``self.train_history_``."""
        if not hasattr(self, 'train_history_') or self.train_history_ is None:
            self.train_history_: Dict[str, list] = {
                "train_loss": [],
                "val_loss": [],
                "val_epochs": [],
            }
        epoch_offset = len(self.train_history_["train_loss"])
        self.train_history_["train_loss"].extend(history.get("train_loss", []))
        if history.get("val_loss"):
            self.train_history_["val_loss"].extend(history["val_loss"])
            self.train_history_["val_epochs"].extend(
                e + epoch_offset for e in history.get("val_epochs", [])
            )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def forecast(
        self,
        steps: int,
        dt: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> pd.DataFrame:
        """
        Integrate the learned dynamics forward from the last observed state.

        Parameters
        ----------
        steps : int
            Number of time steps to forecast.
        dt : float, optional
            Step size. Defaults to the median time step in the training data.
        initial_state : Tensor, optional
            Starting state. Defaults to the last row of the training data.

        Returns
        -------
        pd.DataFrame with columns [time_column, *state_columns]
        """
        from pyUDE.analysis.forecast import forecast as _forecast

        self._require_trained()
        return _forecast(self, steps=steps, dt=dt, initial_state=initial_state)

    def get_right_hand_side(self) -> Callable:
        """
        Return a callable ``f(u, t) -> du`` representing the learned dynamics.

        The returned function accepts numpy arrays or tensors and returns a
        numpy array, making it easy to use with scipy or matplotlib.
        """
        from pyUDE.analysis.dynamics import get_right_hand_side as _get_rhs

        self._require_trained()
        return _get_rhs(self)

    def get_state_estimates(self) -> pd.DataFrame:
        """
        Return state estimates over the training period.

        When trained with ``loss="conditional_likelihood"``, returns the
        Kalman-filtered state estimates (denoised trajectory).  Otherwise,
        returns the ODE trajectory integrated from the first observation.

        Returns
        -------
        pd.DataFrame with columns ``[time_column, *state_columns]``
        """
        self._require_trained()
        t, u_obs = self._get_training_tensors()

        if self._state_estimates is not None:
            u_est = self._state_estimates.detach().cpu()
        else:
            try:
                from torchdiffeq import odeint
            except ImportError as e:
                raise ImportError(
                    "torchdiffeq is required for state estimation. "
                    "Install with: pip install torchdiffeq"
                ) from e
            solver = self._solver or "dopri5"
            self._ode_func.eval()
            with torch.no_grad():
                u_est = odeint(self._ode_func, u_obs[0], t, method=solver)
            self._ode_func.train()

        df = pd.DataFrame(u_est.numpy(), columns=self._state_columns)
        df.insert(0, self._time_column, t.cpu().numpy())
        return df

    def get_predictions(self) -> pd.DataFrame:
        """
        Return one-step-ahead predicted vs observed changes.

        Evaluates the ODE right-hand side at each observed state to get
        predicted derivatives, and compares against finite-difference observed
        changes.  Mirrors UniversalDiffEq.jl's ``plot_predictions``.

        Returns
        -------
        pd.DataFrame
            Columns: ``[time, pred_<state>, obs_<state>, ...]`` for each state.
        """
        self._require_trained()
        t, u_obs = self._get_training_tensors()

        self._ode_func.eval()
        with torch.no_grad():
            du_pred = torch.stack([
                self._ode_func(t[i], u_obs[i]) for i in range(len(t))
            ])
        self._ode_func.train()

        # Finite-difference observed changes
        dt = t[1:] - t[:-1]
        du_obs = (u_obs[1:] - u_obs[:-1]) / dt.unsqueeze(1)

        # Align: use interior points (drop last predicted, drop first observed)
        t_mid = t[:-1].cpu().numpy()
        du_pred_np = du_pred[:-1].cpu().numpy()
        du_obs_np = du_obs.cpu().numpy()

        columns = {}
        columns[self._time_column] = t_mid
        for i, col in enumerate(self._state_columns):
            columns[f"pred_{col}"] = du_pred_np[:, i]
            columns[f"obs_{col}"] = du_obs_np[:, i]
        return pd.DataFrame(columns)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        path : str
            File path (e.g. ``"model.pt"``).
        """
        self._require_trained()
        torch.save({
            'ode_func_state': self._ode_func.state_dict(),
            'data': self._data,
            'time_column': self._time_column,
            'solver': self._solver,
            'class': self.__class__.__name__,
            'obs_cov': self._obs_err,
            'proc_cov': self._proc_err,
            'proc_weight': self._proc_weight,
            'obs_weight': self._obs_weight,
        }, path)

    def load_weights(self, path: str) -> "UDEModel":
        """
        Load weights from a checkpoint into this model.

        The model must have the same architecture as when ``save()`` was called.
        After loading, the model is marked as trained and can be used for
        forecasting or continued training.

        Parameters
        ----------
        path : str
            Path to a checkpoint written by ``save()``.

        Returns
        -------
        self (fluent interface)
        """
        checkpoint = torch.load(path, weights_only=False, map_location=self._device)
        if self._ode_func is None:
            self._ode_func = self._build_ode_func().to(self._device)
        self._ode_func.load_state_dict(checkpoint['ode_func_state'])
        self._solver = checkpoint.get('solver', 'dopri5')
        # Restore covariance settings from checkpoint (backwards-compatible)
        if checkpoint.get('obs_cov') is not None:
            self._obs_err = checkpoint['obs_cov']
        if checkpoint.get('proc_cov') is not None:
            self._proc_err = checkpoint['proc_cov']
        self._proc_weight = checkpoint.get('proc_weight', 1.0)
        self._obs_weight = checkpoint.get('obs_weight', 1.0)
        self._is_trained = True
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained yet. Call model.train() first."
            )

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (
            f"{self.__class__.__name__}("
            f"states={self._n_states}, "
            f"columns={self._state_columns}, "
            f"device='{self._device}', "
            f"{status})"
        )

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def state_columns(self) -> List[str]:
        return list(self._state_columns)

    @property
    def time_column(self) -> str:
        return self._time_column

    @property
    def observation_error(self) -> Optional[torch.Tensor]:
        """Observation noise covariance matrix, or ``None``."""
        return self._obs_err

    @property
    def process_error(self) -> Optional[torch.Tensor]:
        """Process noise covariance matrix, or ``None``."""
        return self._proc_err

    def _get_training_tensors(self):
        """Return ``(t, u)`` tensors from the training DataFrame."""
        t, u, _ = dataframe_to_tensors(self._data, self._time_column)
        return t, u
