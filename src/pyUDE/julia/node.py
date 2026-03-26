"""JuliaNODE — Neural ODE backed by UniversalDiffEq.jl."""

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from pyUDE.utils.validation import validate_dataframe
from pyUDE.julia._env import get_julia
from pyUDE.julia._convert import df_to_julia, julia_forecast_to_df


class JuliaNODE:
    """
    Neural ODE using UniversalDiffEq.jl as the backend.

    Learns the full system dynamics ``du/dt = NN(u)`` from data.
    The neural network is a Lux.jl MLP; the ODE is solved by
    DifferentialEquations.jl.

    Compared to the PyTorch ``NODE``:
    - Uses Julia's Tsit5/Vern9 adaptive solvers (2-10× faster for most systems)
    - Has first-class stiff solvers (Rodas5, QNDF) for chemistry/biology
    - Delegates optimisation to UniversalDiffEq.jl's training routines

    Parameters
    ----------
    data : pd.DataFrame
        Time series data. Must have a time column and at least one state column.
    hidden_layers : int
        Number of hidden layers in the default Lux MLP.
    hidden_units : int
        Units per hidden layer.
    time_column : str
        Name of the time column in ``data``.
    solver : str
        Julia ODE solver name.  ``"Tsit5"`` for non-stiff systems (default);
        ``"Rodas5"`` or ``"QNDF"`` for stiff systems.

    Examples
    --------
    >>> model = JuliaNODE(data)
    >>> model.train(epochs=500)
    >>> predictions = model.forecast(steps=50)
    """

    def __init__(
        self,
        data: pd.DataFrame,
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
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._solver = solver
        self._jl_model = None   # Julia-side UniversalDiffEq model struct
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
    ) -> "JuliaNODE":
        """
        Fit the NODE to the training data.

        Parameters
        ----------
        optimizer : {"adam", "sgd"}
        learning_rate : float
        epochs : int
        verbose : bool

        Returns
        -------
        self (fluent interface)
        """
        jl, UDE = get_julia()

        if self._jl_model is None:
            # Build Lux MLP
            nn = jl.make_lux_mlp(
                self._n_states, self._n_states,
                self._hidden_units, self._hidden_layers,
            )

            # Convert data
            t_jl, data_jl, _ = df_to_julia(self._data, self._time_column)

            # Construct UniversalDiffEq.NODE Julia struct
            self._jl_model = UDE.NODE(data_jl, t_jl, neural_network=nn)

        # Build optimizer
        jl_opt = jl.make_optimizer(optimizer, float(learning_rate))

        # Train
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
        """
        Integrate the learned dynamics forward from the last observed state.

        Parameters
        ----------
        steps : int
        dt : float, optional
            Step size. Defaults to the median training interval.
        initial_state : ignored (UniversalDiffEq handles initial conditions internally)

        Returns
        -------
        pd.DataFrame  [time_column, *state_columns]
        """
        self._require_trained()

        if dt is None:
            t = self._data[self._time_column].to_numpy()
            dt = float(np.median(np.diff(t)))

        return julia_forecast_to_df(
            self._jl_model, steps, dt,
            self._state_columns, self._time_column,
        )

    def get_right_hand_side(self) -> Callable:
        """
        Return a numpy-compatible callable ``f(u, t) -> du``.

        The returned function wraps the Julia-side Lux network so that it can
        be used with ``scipy.integrate.odeint`` or other Python ODE solvers.

        Returns
        -------
        callable  f(u: array-like, t: float) -> np.ndarray
        """
        self._require_trained()
        jl, UDE = get_julia()
        jl_rhs = UDE.get_right_hand_side(self._jl_model)

        def rhs(u, t):
            import numpy as np
            py_u = list(np.asarray(u, dtype=np.float64))
            result = jl_rhs(py_u, float(t))
            return np.array(list(result), dtype=np.float64)

        return rhs

    # ------------------------------------------------------------------
    # Properties (mirror pyUDE.core.base.UDEModel)
    # ------------------------------------------------------------------

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

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained yet. Call model.train() first."
            )
