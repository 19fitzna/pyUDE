"""JuliaCustomDerivatives — Hybrid UDE backed by UniversalDiffEq.jl."""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from pyUDE.utils.validation import validate_dataframe
from pyUDE.julia._base import JuliaModelBase
from pyUDE.julia._env import get_julia
from pyUDE.julia._convert import df_to_julia, julia_forecast_to_df, params_dict_to_julia


class JuliaCustomDerivatives(JuliaModelBase):
    """
    Hybrid UDE: user-supplied known dynamics + Lux.jl neural network,
    backed by UniversalDiffEq.jl.

    Equivalent to ``pyUDE.CustomDerivatives`` but uses Julia's ODE solvers.
    Mirrors the API of UniversalDiffEq.CustomDerivatives exactly.

    The key technical detail: ``known_dynamics`` is a **Python callable**.
    It is wrapped by ``pyude_bridge.wrap_python_dynamics()`` so Julia's ODE
    solver can call it from within the integration loop via PythonCall.jl.

    Parameters
    ----------
    data : pd.DataFrame
    known_dynamics : callable
        ``f(u, p, t) -> du`` where:
        - ``u`` : list of floats (state vector)
        - ``p`` : dict mapping parameter names → float values
        - ``t`` : float (current time)
        - returns : list or numpy array of shape ``(n_states,)``
    init_params : dict
        Initial values of mechanistic parameters in ``known_dynamics``.
        Optimised jointly with the neural network weights.
    hidden_layers : int
    hidden_units : int
    time_column : str
    solver : str
        Julia ODE solver.  ``"Tsit5"`` (non-stiff, default) or
        ``"Rodas5"`` / ``"QNDF"`` for stiff systems.

    Examples
    --------
    >>> def lv_known(u, p, t):
    ...     prey, pred = u[0], u[1]
    ...     return [p["alpha"] * prey, -p["delta"] * pred]
    >>>
    >>> model = JuliaCustomDerivatives(data, lv_known, {"alpha": 1.0, "delta": 1.5})
    >>> model.train(epochs=1000)
    >>> print(model.get_params())
    """

    def __init__(
        self,
        data: pd.DataFrame,
        known_dynamics: Callable,
        init_params: Dict,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        time_column: str = "time",
        solver: str = "Tsit5",
    ):
        validate_dataframe(data, time_column)
        self._data = data
        self._time_column = time_column
        self._state_columns: List[str] = [c for c in data.columns if c != time_column]
        self._n_states: int = len(self._state_columns)
        self._known_dynamics = known_dynamics
        self._init_params = init_params
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._solver = solver
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
    ) -> "JuliaCustomDerivatives":
        """
        Fit the hybrid UDE to training data.

        The Python ``known_dynamics`` callable is wrapped so Julia's ODE solver
        can call it via PythonCall.jl on every RHS evaluation.

        Returns self (fluent interface).
        """
        jl, UDE = get_julia()

        if self._jl_model is None:
            # Wrap Python known_dynamics as a Julia-callable function
            jl_dynamics = jl.wrap_python_dynamics(self._known_dynamics)

            # Build Lux MLP for unknown residual dynamics
            nn = jl.make_lux_mlp(
                self._n_states, self._n_states,
                self._hidden_units, self._hidden_layers,
            )

            # Convert training data and initial parameters
            t_jl, data_jl, _ = df_to_julia(self._data, self._time_column)
            init_params_jl = params_dict_to_julia(self._init_params, jl)

            # Construct UniversalDiffEq.CustomDerivatives Julia struct
            self._jl_model = UDE.CustomDerivatives(
                data_jl, t_jl, jl_dynamics, init_params_jl,
                neural_network=nn,
            )

        # Build optimizer and train
        jl_opt = jl.make_optimizer(optimizer, float(learning_rate))
        UDE.train_b(
            self._jl_model,
            optimiser=jl_opt,
            loss_function="trajectory",
            n_epochs=int(epochs),
            verbose=verbose,
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
        initial_state=None,
    ) -> pd.DataFrame:
        """Integrate forward from the last observed state."""
        self._require_trained()
        if dt is None:
            t = self._data[self._time_column].to_numpy()
            dt = float(np.median(np.diff(t)))
        return julia_forecast_to_df(
            self._jl_model, steps, dt,
            self._state_columns, self._time_column,
        )

    def get_right_hand_side(self) -> Callable:
        """Return numpy-compatible ``f(u, t) -> du`` for the full learned dynamics."""
        self._require_trained()
        jl, UDE = get_julia()
        jl_rhs = UDE.get_right_hand_side(self._jl_model)

        def rhs(u, t):
            py_u = list(np.asarray(u, dtype=np.float64))
            result = jl_rhs(py_u, float(t))
            return np.array(list(result), dtype=np.float64)

        return rhs

    def get_params(self) -> Dict:
        """
        Return the current values of the learned mechanistic parameters.

        Returns
        -------
        dict  mapping parameter name → float
        """
        self._require_trained()
        jl, UDE = get_julia()
        try:
            jl_params = self._jl_model.parameters
            return {str(k): float(v) for k, v in zip(jl_params._fields, jl_params)}
        except Exception as e:
            raise RuntimeError(
                "Failed to extract parameters from Julia model. "
                "The UniversalDiffEq.jl parameter storage format may have changed. "
                f"Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (
            f"{self.__class__.__name__}("
            f"states={self._n_states}, "
            f"columns={self._state_columns}, "
            f"solver='{self._solver}', "
            f"{status})"
        )
