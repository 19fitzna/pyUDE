from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import pandas as pd
import torch
import torch.nn as nn

from pyUDE.utils.validation import validate_dataframe
from pyUDE.utils.data import dataframe_to_tensors, tensors_to_dataframe


class UDEModel(ABC):
    """
    Abstract base class for all Universal Differential Equation models.

    Subclasses implement ``_build_ode_func`` to define the right-hand side
    of the differential equation as a callable ``f(t, u) -> du``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_column: str = "time",
        device: str = "cpu",
    ):
        validate_dataframe(data, time_column)
        self._data = data
        self._time_column = time_column
        self._state_columns: List[str] = [c for c in data.columns if c != time_column]
        self._n_states: int = len(self._state_columns)
        self._is_trained: bool = False
        self._device: torch.device = torch.device(device)

        # Populated by train()
        self._ode_func: Optional[nn.Module] = None
        self._solver: Optional[str] = None

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
        **kwargs,
    ) -> "UDEModel":
        """
        Fit the model to the training data.

        Parameters
        ----------
        loss : {"simulation", "derivative_matching"}
            Loss function. ``"simulation"`` integrates the ODE forward and
            compares to observations; ``"derivative_matching"`` compares
            predicted derivatives against cubic-spline estimates.
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
        max_grad_norm : float
            Maximum norm for gradient clipping. Set to 0 to disable.
        weight_decay : float, optional
            L2 regularisation. Defaults to ``1e-4`` for derivative_matching
            and ``0.0`` for simulation.

        Returns
        -------
        self (fluent interface)
        """
        from pyUDE.training.trainer import train_model

        if weight_decay is None:
            weight_decay = 1e-4 if loss == "derivative_matching" else 0.0

        if self._ode_func is None:
            self._ode_func = self._build_ode_func()

        self._ode_func = self._ode_func.to(self._device)
        self._solver = solver

        train_model(
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
            **kwargs,
        )
        self._is_trained = True
        return self

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
        }, path)

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

    def _get_training_tensors(self):
        """Return (t, u) tensors from the training DataFrame."""
        return dataframe_to_tensors(self._data, self._time_column)
