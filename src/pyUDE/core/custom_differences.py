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
    ):
        super().__init__(data, time_column)
        self._known_map = known_map
        self._init_params = init_params
        self._network = network
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units

    def _build_ode_func(self) -> nn.Module:
        # Not used for discrete models — overriding train() instead
        return None

    def train(
        self,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 500,
        log_interval: int = 50,
        verbose: bool = True,
        **kwargs,
    ) -> "CustomDifferences":
        """Train by minimising one-step-ahead MSE across the time series."""
        from pyUDE.training.trainer import train_differences

        self._param_dict = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(float(v), dtype=torch.float64))
            for k, v in self._init_params.items()
        })

        if self._network is None:
            self._network_module = _default_mlp(
                in_dim=self._n_states,
                out_dim=self._n_states,
                hidden_layers=self._hidden_layers,
                hidden_units=self._hidden_units,
            ).double()
        else:
            self._network_module = self._network

        train_differences(
            model=self,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            epochs=epochs,
            log_interval=log_interval,
            verbose=verbose,
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
