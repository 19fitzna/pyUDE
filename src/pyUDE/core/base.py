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
    ):
        validate_dataframe(data, time_column)
        self._data = data
        self._time_column = time_column
        self._state_columns: List[str] = [c for c in data.columns if c != time_column]
        self._n_states: int = len(self._state_columns)
        self._is_trained: bool = False

        # Populated by train()
        self._ode_func: Optional[nn.Module] = None

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
        **kwargs,
    ) -> "UDEModel":
        """
        Fit the model to the training data.

        Parameters
        ----------
        loss : {"simulation", "derivative_matching"}
            Loss function. ``"simulation"`` integrates the ODE forward and
            compares to observations; ``"derivative_matching"`` compares
            predicted derivatives against finite-difference estimates.
        optimizer : str
            Optimizer name (``"adam"`` or ``"sgd"``).
        learning_rate : float
        epochs : int
        log_interval : int
            Print loss every this many epochs when ``verbose=True``.
        verbose : bool
        solver : str
            ODE solver passed to torchdiffeq (e.g. ``"dopri5"``, ``"rk4"``).

        Returns
        -------
        self (fluent interface)
        """
        from pyUDE.training.trainer import train_model

        if self._ode_func is None:
            self._ode_func = self._build_ode_func()

        train_model(
            model=self,
            loss=loss,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            epochs=epochs,
            log_interval=log_interval,
            verbose=verbose,
            solver=solver,
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
    # Helpers
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained yet. Call model.train() first."
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
