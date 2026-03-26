from typing import Callable, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn

from pyUDE.core.base import UDEModel
from pyUDE.core.node import _default_mlp


class _CustomDerivativesFunc(nn.Module):
    """
    ODE right-hand side: du/dt = known_dynamics(u, p, t) + network(u).

    Parameters
    ----------
    known_dynamics : callable
        ``f(u, p, t) -> du`` where ``p`` is a dict of named parameters.
    network : nn.Module
    params : nn.ParameterDict
        Mechanistic parameters made trainable via ``nn.Parameter``.
    """

    def __init__(
        self,
        known_dynamics: Callable,
        network: nn.Module,
        params: nn.ParameterDict,
    ):
        super().__init__()
        self.known_dynamics = known_dynamics
        self.network = network
        self.params = params

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        p = {k: v for k, v in self.params.items()}
        du_known = self.known_dynamics(u, p, t)
        du_nn = self.network(u)
        return du_known + du_nn


class CustomDerivatives(UDEModel):
    """
    Hybrid UDE: user-supplied known dynamics + neural network for unknown terms.

    Equivalent to ``UniversalDiffEq.CustomDerivatives`` in Julia.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data.
    known_dynamics : callable
        ``f(u, p, t) -> du`` where:

        - ``u`` is a 1-D tensor of state values,
        - ``p`` is a dict mapping parameter names to scalar tensors,
        - ``t`` is a scalar tensor.

        The function should return a tensor of the same shape as ``u``.
    init_params : dict
        Initial values for the mechanistic parameters in ``known_dynamics``.
        Values can be floats or tensors; they will be made trainable.
    network : nn.Module, optional
        Neural network for unknown residual dynamics. Defaults to a small MLP.
    hidden_layers : int
    hidden_units : int
    time_column : str

    Examples
    --------
    >>> def lv(u, p, t):
    ...     prey, pred = u[0], u[1]
    ...     return torch.stack([
    ...         p["alpha"] * prey - p["beta"] * prey * pred,
    ...         -p["delta"] * pred,
    ...     ])
    >>> model = CustomDerivatives(data, lv, {"alpha": 1.0, "beta": 0.1, "delta": 1.0})
    >>> model.train(epochs=1000)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        known_dynamics: Callable,
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
        self._known_dynamics = known_dynamics
        self._init_params = init_params
        self._network = network
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._dropout = dropout
        self._param_bounds = param_bounds
        self._validate_known_dynamics()

    def _validate_known_dynamics(self) -> None:
        """Probe known_dynamics with a zero input to catch shape/signature errors early."""
        u_test = torch.zeros(self._n_states, dtype=torch.float64)
        p_test = {k: torch.tensor(float(v), dtype=torch.float64) for k, v in self._init_params.items()}
        t_test = torch.tensor(0.0, dtype=torch.float64)
        try:
            result = self._known_dynamics(u_test, p_test, t_test)
        except Exception as e:
            raise ValueError(f"known_dynamics raised an error on test input: {e}") from e
        if result.shape != (self._n_states,):
            raise ValueError(
                f"known_dynamics must return shape ({self._n_states},), got {result.shape}"
            )

    def _build_ode_func(self) -> nn.Module:
        # Store as instance variable so learned values survive if _ode_func is reset
        if not hasattr(self, '_param_dict') or self._param_dict is None:
            self._param_dict = nn.ParameterDict({
                k: nn.Parameter(torch.tensor(float(v), dtype=torch.float64))
                for k, v in self._init_params.items()
            })

        if self._network is not None:
            net = self._network
        else:
            net = _default_mlp(
                in_dim=self._n_states,
                out_dim=self._n_states,
                hidden_layers=self._hidden_layers,
                hidden_units=self._hidden_units,
                dropout=self._dropout,
            ).double()

        return _CustomDerivativesFunc(self._known_dynamics, net, self._param_dict)

    def get_params(self) -> Dict:
        """Return the current values of the mechanistic parameters."""
        self._require_trained()
        return {k: v.item() for k, v in self._param_dict.items()}
