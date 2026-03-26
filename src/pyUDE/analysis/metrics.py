"""Evaluation metrics for UDE model predictions."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def _to_array(x, state_columns=None):
    """Convert a DataFrame or ndarray to ``(array, column_names)``."""
    if isinstance(x, pd.DataFrame):
        cols = list(state_columns) if state_columns is not None else list(x.columns)
        return x[cols].values, cols
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        cols = (
            list(state_columns)
            if state_columns is not None
            else [f"state_{i}" for i in range(arr.shape[1])]
        )
        return arr, cols


def _check_shapes(obs, pred):
    if obs.shape != pred.shape:
        raise ValueError(
            f"observed and predicted shapes do not match: {obs.shape} vs {pred.shape}"
        )


def mse(observed, predicted, state_columns=None) -> Dict[str, float]:
    """Mean squared error per state variable.

    Parameters
    ----------
    observed, predicted : pd.DataFrame or np.ndarray
        Must have matching shapes. When DataFrames are passed the time
        column is excluded automatically via ``state_columns``.
    state_columns : list[str], optional
        Column names to evaluate. If ``None``, all columns are used.

    Returns
    -------
    dict mapping column name → MSE, plus ``"mean"`` key with the average.
    """
    obs_arr, cols = _to_array(observed, state_columns)
    pred_arr, _ = _to_array(predicted, cols)
    _check_shapes(obs_arr, pred_arr)
    result = {col: float(np.mean((obs_arr[:, i] - pred_arr[:, i]) ** 2))
              for i, col in enumerate(cols)}
    result["mean"] = float(np.mean(list(result.values())))
    return result


def rmse(observed, predicted, state_columns=None) -> Dict[str, float]:
    """Root mean squared error per state variable.

    Same signature as :func:`mse`. Returns ``{"col": rmse_val, ..., "mean": ...}``.
    """
    return {k: float(np.sqrt(v)) for k, v in mse(observed, predicted, state_columns).items()}


def mae(observed, predicted, state_columns=None) -> Dict[str, float]:
    """Mean absolute error per state variable.

    Same signature as :func:`mse`.
    """
    obs_arr, cols = _to_array(observed, state_columns)
    pred_arr, _ = _to_array(predicted, cols)
    _check_shapes(obs_arr, pred_arr)
    result = {col: float(np.mean(np.abs(obs_arr[:, i] - pred_arr[:, i])))
              for i, col in enumerate(cols)}
    result["mean"] = float(np.mean(list(result.values())))
    return result


def r2_score(observed, predicted, state_columns=None) -> Dict[str, float]:
    """Coefficient of determination (R²) per state variable.

    Same signature as :func:`mse`. Returns 1.0 for a state with zero variance.
    """
    obs_arr, cols = _to_array(observed, state_columns)
    pred_arr, _ = _to_array(predicted, cols)
    _check_shapes(obs_arr, pred_arr)
    result = {}
    for i, col in enumerate(cols):
        ss_res = np.sum((obs_arr[:, i] - pred_arr[:, i]) ** 2)
        ss_tot = np.sum((obs_arr[:, i] - np.mean(obs_arr[:, i])) ** 2)
        result[col] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
    result["mean"] = float(np.mean(list(result.values())))
    return result


def score(
    observed,
    predicted,
    metric: str = "rmse",
    state_columns=None,
) -> Dict[str, float]:
    """Compute a named metric between observed and predicted values.

    Parameters
    ----------
    observed, predicted : pd.DataFrame or np.ndarray
    metric : {"mse", "rmse", "mae", "r2_score", "r2"}
    state_columns : list[str], optional

    Returns
    -------
    dict mapping column name → metric value, plus ``"mean"`` key.
    """
    _dispatch = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2_score,
        "r2": r2_score,
    }
    if metric not in _dispatch:
        raise ValueError(
            f"Unknown metric {metric!r}. Choose from {list(_dispatch.keys())}."
        )
    return _dispatch[metric](observed, predicted, state_columns)
