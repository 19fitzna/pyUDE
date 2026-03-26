from typing import Callable, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn

from pyUDE.core.base import UDEModel
from pyUDE.core.node import _default_mlp
from pyUDE.utils.validation import validate_dataframe
from pyUDE.utils.data import dataframe_to_tensors, tensors_to_dataframe


class CustomDifferences(UDEModel):
    """
    Universal Difference Equation — discrete-time variant of CustomDerivatives.

    Models state transitions as:
        u[t+1] = known_map(u[t], p, t) + network(u[t])

    Equivalent to ``UniversalDiffEq.CustomDifferences`` in Julia.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data.
    known_map : callable
        ``f(u, p, t) -> u_next`` defining the known part of the state transition.
    init_params : dict
        Initial values for the mechanistic parameters.
    network : nn.Module, optional
        Neural network for unknown residual. Defaults to a small MLP.
    hidden_layers : int
    hidden_units : int
    time_column : str
    """

    def __init__(
        self,
        data: pd.DataFrame,
        known_map: Callable,
        init_params: Dict,
        network: Optional[nn.Module] = None,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        time_column: str = "time",
        device: str = "cpu",
    ):
        super().__init__(data, time_column, device)
        self._known_map = known_map
        self._init_params = init_params
        self._network = network
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._validate_known_map()

    def _validate_known_map(self) -> None:
        """Probe known_map with a zero input to catch shape/signature errors early."""
        u_test = torch.zeros(self._n_states, dtype=torch.float64)
        p_test = {k: torch.tensor(float(v), dtype=torch.float64) for k, v in self._init_params.items()}
        t_test = torch.tensor(0.0, dtype=torch.float64)
        try:
            result = self._known_map(u_test, p_test, t_test)
        except Exception as e:
            raise ValueError(f"known_map raised an error on test input: {e}") from e
        if result.shape != (self._n_states,):
            raise ValueError(
                f"known_map must return shape ({self._n_states},), got {result.shape}"
            )

    def _build_ode_func(self) -> None:
        # Discrete-time models have no continuous ODE function
        return None

    def get_right_hand_side(self):
        raise NotImplementedError(
            "Discrete-time models do not have a continuous right-hand side. "
            "Use forecast() to step the learned map forward."
        )

    def train(
        self,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 500,
        log_interval: int = 50,
        verbose: bool = True,
        patience: Optional[int] = None,
        max_grad_norm: float = 10.0,
        **kwargs,
    ) -> "CustomDifferences":
        """Train by minimising one-step-ahead MSE across the time series."""
        from pyUDE.training.trainer import train_differences

        if not hasattr(self, '_param_dict'):
            self._param_dict = nn.ParameterDict({
                k: nn.Parameter(torch.tensor(float(v), dtype=torch.float64))
                for k, v in self._init_params.items()
            }).to(self._device)

        if not hasattr(self, '_network_module'):
            if self._network is None:
                self._network_module = _default_mlp(
                    in_dim=self._n_states,
                    out_dim=self._n_states,
                    hidden_layers=self._hidden_layers,
                    hidden_units=self._hidden_units,
                ).double().to(self._device)
            else:
                self._network_module = self._network.to(self._device)

        train_differences(
            model=self,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            epochs=epochs,
            log_interval=log_interval,
            verbose=verbose,
            patience=patience,
            max_grad_norm=max_grad_norm,
        )
        self._is_trained = True
        return self

    def forecast(self, steps: int, initial_state=None, **kwargs):
        """Step the discrete map forward from the last observed state."""
        from pyUDE.analysis.forecast import forecast_differences

        self._require_trained()
        return forecast_differences(self, steps=steps, initial_state=initial_state)

    def get_params(self) -> Dict:
        """Return current mechanistic parameter values."""
        self._require_trained()
        return {k: v.item() for k, v in self._param_dict.items()}
