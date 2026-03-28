from typing import Callable, Dict, Optional, Union

import pandas as pd
import torch
import torch.nn as nn

from pyUDE.core.base import UDEModel
from pyUDE.core.node import _default_mlp


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
        dropout: float = 0.0,
        param_bounds: Optional[Dict] = None,
    ):
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}.")
        if param_bounds is not None:
            extra = set(param_bounds) - set(init_params)
            if extra:
                raise ValueError(
                    f"param_bounds contains keys not in init_params: {sorted(extra)}"
                )
        super().__init__(data, time_column, device)
        self._known_map = known_map
        self._init_params = init_params
        self._network = network
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._dropout = dropout
        self._param_bounds = param_bounds
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
        weight_decay: float = 0.0,
        val_data: Optional[pd.DataFrame] = None,
        val_interval: int = 1,
        lambda_l1: float = 0.0,
        scheduler: Optional[Union[str, torch.optim.lr_scheduler.LRScheduler]] = None,
        progress_bar: bool = False,
        obs_weight: Optional[float] = None,
        loss: str = "one_step",
        **kwargs,
    ) -> "CustomDifferences":
        """Train by minimising one-step-ahead MSE across the time series."""
        if loss in ("conditional_likelihood", "multiple_shooting"):
            raise ValueError(
                f"loss='{loss}' requires ODE integration and is not supported "
                f"for discrete-time models. Use loss='one_step' (default)."
            )

        from pyUDE.training.trainer import train_differences

        # Clear stale state estimates
        self._state_estimates = None

        eff_obs_weight = obs_weight if obs_weight is not None else self._obs_weight

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
                    dropout=self._dropout,
                ).double().to(self._device)
            else:
                self._network_module = self._network.double().to(self._device)

        val_t = val_u = None
        if val_data is not None:
            from pyUDE.utils.validation import validate_dataframe
            from pyUDE.utils.data import dataframe_to_tensors
            validate_dataframe(val_data, self._time_column)
            val_t, val_u, _ = dataframe_to_tensors(val_data, self._time_column)

        self.train_result_ = train_differences(
            model=self,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            epochs=epochs,
            log_interval=log_interval,
            verbose=verbose,
            patience=patience,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            val_t=val_t,
            val_u=val_u,
            val_interval=val_interval,
            lambda_l1=lambda_l1,
            scheduler=scheduler,
            progress_bar=progress_bar,
            obs_weight=eff_obs_weight,
        )
        self._is_trained = True
        self._merge_history(self.train_result_.to_dict())
        return self

    def forecast(self, steps: int, initial_state=None, **kwargs):
        """Step the discrete map forward from the last observed state."""
        from pyUDE.analysis.forecast import forecast_differences

        self._require_trained()
        return forecast_differences(self, steps=steps, initial_state=initial_state)

    def save(self, path: str) -> None:
        """Save the trained discrete-time model to disk."""
        self._require_trained()
        torch.save({
            'param_dict_state': self._param_dict.state_dict(),
            'network_state': self._network_module.state_dict(),
            'data': self._data,
            'time_column': self._time_column,
            'class': self.__class__.__name__,
        }, path)

    def load_weights(self, path: str) -> "CustomDifferences":
        """
        Load weights from a checkpoint into this model.

        The model must have the same architecture as when ``save()`` was called.

        Parameters
        ----------
        path : str
            Path to a checkpoint written by ``save()``.

        Returns
        -------
        self (fluent interface)
        """
        checkpoint = torch.load(path, weights_only=False, map_location=self._device)
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
                    dropout=self._dropout,
                ).double().to(self._device)
            else:
                self._network_module = self._network.double().to(self._device)
        self._param_dict.load_state_dict(checkpoint['param_dict_state'])
        self._network_module.load_state_dict(checkpoint['network_state'])
        self._is_trained = True
        return self

    def get_params(self) -> Dict:
        """Return current mechanistic parameter values."""
        self._require_trained()
        return {k: v.item() for k, v in self._param_dict.items()}
