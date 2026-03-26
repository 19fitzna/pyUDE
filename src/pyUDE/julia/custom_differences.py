"""JuliaCustomDifferences — Discrete-time hybrid UDE backed by UniversalDiffEq.jl."""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from pyUDE.utils.validation import validate_dataframe
from pyUDE.julia._base import JuliaModelBase
from pyUDE.julia._env import get_julia
from pyUDE.julia._convert import df_to_julia, params_dict_to_julia


class JuliaCustomDifferences(JuliaModelBase):
    """
    Discrete-time hybrid UDE: user-supplied known map + Lux.jl neural network,
    backed by UniversalDiffEq.jl.

    Models state transitions as:
        u[t+1] = known_map(u[t], p, t) + network(u[t])

    Equivalent to ``pyUDE.CustomDifferences`` but uses UniversalDiffEq.jl's
    discrete-time training routines.

    Parameters
    ----------
    data : pd.DataFrame
    known_map : callable
        ``f(u, p, t) -> u_next`` — the known part of the state transition.
        Signature matches Julia's ``CustomDifferences`` convention:
        - ``u`` : list of floats
        - ``p`` : dict of parameter name → float
        - ``t`` : float
        - returns : list or numpy array
    init_params : dict
    hidden_layers : int
    hidden_units : int
    time_column : str
    """

    def __init__(
        self,
        data: pd.DataFrame,
        known_map: Callable,
        init_params: Dict,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        time_column: str = "time",
    ):
        validate_dataframe(data, time_column)
        self._data = data
        self._time_column = time_column
        self._state_columns: List[str] = [c for c in data.columns if c != time_column]
        self._n_states: int = len(self._state_columns)
        self._known_map = known_map
        self._init_params = init_params
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._jl_model = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 500,
        verbose: bool = True,
        **kwargs,
    ) -> "JuliaCustomDifferences":
        """Fit the discrete-time UDE. Returns self."""
        jl, UDE = get_julia()

        if self._jl_model is None:
            jl_map = jl.wrap_python_dynamics(self._known_map)
            nn = jl.make_lux_mlp(
                self._n_states, self._n_states,
                self._hidden_units, self._hidden_layers,
            )
            t_jl, data_jl, _ = df_to_julia(self._data, self._time_column)
            init_params_jl = params_dict_to_julia(self._init_params, jl)

            self._jl_model = UDE.CustomDifferences(
                data_jl, t_jl, jl_map, init_params_jl,
                neural_network=nn,
            )

        jl_opt = jl.make_optimizer(optimizer, float(learning_rate))
        UDE.train_b(
            self._jl_model,
            optimiser=jl_opt,
            n_epochs=int(epochs),
            verbose=verbose,
        )
        self._is_trained = True
        return self

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def forecast(self, steps: int, dt: Optional[float] = None, **kwargs) -> pd.DataFrame:
        """Step the discrete map forward ``steps`` times from the last observed state."""
        self._require_trained()
        jl, UDE = get_julia()

        jl_forecast = UDE.forecast(self._jl_model, steps)
        t_py = list(jl_forecast.time)
        state_arrays = {col: list(getattr(jl_forecast, col)) for col in self._state_columns}
        df = pd.DataFrame(state_arrays)
        df.insert(0, self._time_column, t_py)
        return df

    def get_params(self) -> Dict:
        """Return current mechanistic parameter values."""
        self._require_trained()
        jl, _ = get_julia()
        try:
            jl_params = self._jl_model.parameters
            return {str(k): float(v) for k, v in zip(jl_params._fields, jl_params)}
        except Exception as e:
            raise RuntimeError(
                "Failed to extract parameters from Julia model. "
                "The UniversalDiffEq.jl parameter storage format may have changed. "
                f"Original error: {e}"
            ) from e

