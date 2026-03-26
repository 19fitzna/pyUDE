from typing import Optional

import pandas as pd
import torch
import torch.nn as nn

from pyUDE.core.base import UDEModel


def _default_mlp(in_dim: int, out_dim: int, hidden_layers: int, hidden_units: int) -> nn.Sequential:
    """Build a simple MLP with tanh activations."""
    layers = [nn.Linear(in_dim, hidden_units), nn.Tanh()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
    layers.append(nn.Linear(hidden_units, out_dim))
    return nn.Sequential(*layers)


class _NODEFunc(nn.Module):
    """ODE right-hand side: du/dt = network(u)."""

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.network(u)


class NODE(UDEModel):
    """
    Neural ODE — learns the full dynamics from data.

    Equivalent to ``UniversalDiffEq.NODE`` in Julia.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with a time column and one or more state columns.
    network : nn.Module, optional
        Custom PyTorch network. Must accept a tensor of shape ``(n_states,)``
        and return a tensor of the same shape. If ``None``, a default MLP is
        constructed from ``hidden_layers`` and ``hidden_units``.
    hidden_layers : int
        Number of hidden layers in the default MLP (ignored if ``network``
        is provided).
    hidden_units : int
        Units per hidden layer in the default MLP.
    time_column : str
        Name of the time column in ``data``.

    Examples
    --------
    >>> model = NODE(data)
    >>> model.train(epochs=500)
    >>> predictions = model.forecast(steps=50)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        network: Optional[nn.Module] = None,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        time_column: str = "time",
        device: str = "cpu",
    ):
        super().__init__(data, time_column, device)
        self._network = network
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units

    def _build_ode_func(self) -> nn.Module:
        if self._network is not None:
            net = self._network
        else:
            net = _default_mlp(
                in_dim=self._n_states,
                out_dim=self._n_states,
                hidden_layers=self._hidden_layers,
                hidden_units=self._hidden_units,
            ).double()  # match float64 data tensors
        return _NODEFunc(net)
