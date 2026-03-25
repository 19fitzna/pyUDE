"""
Data conversion utilities between pandas DataFrames and Julia arrays.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from pyUDE.julia._env import get_julia


def df_to_julia(
    df: pd.DataFrame,
    time_column: str = "time",
):
    """
    Convert a pandas DataFrame to (t_jl, data_jl) Julia arrays.

    Returns
    -------
    t_jl : Julia Vector{Float64}, shape (T,)
    data_jl : Julia Matrix{Float64}, shape (T, n_states)
        Rows are time steps; columns are state variables in DataFrame order.
    state_columns : list of str
    """
    jl, _ = get_julia()
    state_cols = [c for c in df.columns if c != time_column]
    t_np = df[time_column].to_numpy(dtype=np.float64)
    data_np = df[state_cols].to_numpy(dtype=np.float64)

    t_jl = jl.py_vector_to_julia(t_np.tolist())
    data_jl = jl.py_matrix_to_julia(data_np.tolist())
    return t_jl, data_jl, state_cols


def julia_forecast_to_df(
    jl_model,
    steps: int,
    dt: float,
    state_columns: List[str],
    time_column: str = "time",
) -> pd.DataFrame:
    """
    Run UniversalDiffEq.forecast on a trained Julia model and convert the
    result to a pandas DataFrame.

    Parameters
    ----------
    jl_model : Julia UniversalDiffEq model struct (trained)
    steps : int
    dt : float
    state_columns : list of str
    time_column : str

    Returns
    -------
    pd.DataFrame  with columns [time_column, *state_columns]
    """
    jl, UDE = get_julia()
    # UniversalDiffEq.forecast returns a DataFrame-like Julia object;
    # we ask it to simulate `steps` additional time steps
    jl_forecast = UDE.forecast(jl_model, steps)

    # Extract time and state arrays
    t_py = list(jl_forecast.time)
    state_arrays = {col: list(getattr(jl_forecast, col)) for col in state_columns}

    df = pd.DataFrame(state_arrays)
    df.insert(0, time_column, t_py)
    return df


def params_dict_to_julia(params: dict, jl):
    """Convert a Python dict of float init_params to a Julia NamedTuple."""
    # Build as Julia NamedTuple: (alpha=1.0, delta=1.5)
    items = ", ".join(f"{k}={float(v)}" for k, v in params.items())
    return jl.seval(f"(; {items})")
