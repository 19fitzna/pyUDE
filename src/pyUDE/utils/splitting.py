"""Temporal train/test splitting and time-series cross-validation."""

from typing import List, Optional, Tuple

import pandas as pd


def train_test_split(
    data: pd.DataFrame,
    test_fraction: float = 0.2,
    time_column: str = "time",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame into train and test sets.

    Temporal order is always preserved — rows are never shuffled.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with a time column and at least one state column.
    test_fraction : float
        Fraction of rows to assign to the test set. Must be in ``(0, 1)``.
    time_column : str
        Name of the time column (used only for validation; not required to
        be present for the split itself).

    Returns
    -------
    (train_data, test_data) : tuple of pd.DataFrame
        Both DataFrames have reset integer indices and are valid inputs to
        any model constructor.

    Raises
    ------
    ValueError
        If ``test_fraction`` is not in ``(0, 1)``, or if either split would
        have fewer than 2 rows.
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(
            f"test_fraction must be in (0, 1), got {test_fraction!r}."
        )

    n = len(data)
    n_test = max(1, int(n * test_fraction))
    n_train = n - n_test

    if n_train < 2:
        raise ValueError(
            f"Training split would have {n_train} row(s) — at least 2 required. "
            f"Reduce test_fraction or provide more data."
        )
    if n_test < 2:
        raise ValueError(
            f"Test split would have {n_test} row(s) — at least 2 required. "
            f"Increase test_fraction or provide more data."
        )

    train_data = data.iloc[:n_train].reset_index(drop=True)
    test_data = data.iloc[n_train:].reset_index(drop=True)
    return train_data, test_data


def time_series_cv(
    data: pd.DataFrame,
    model_class,
    model_kwargs: Optional[dict] = None,
    train_kwargs: Optional[dict] = None,
    n_splits: int = 5,
    min_train_fraction: float = 0.5,
    metric: str = "rmse",
    time_column: str = "time",
) -> List[dict]:
    """Expanding-window time-series cross-validation.

    Each fold trains on progressively more data and validates on the
    immediately following window.

    Parameters
    ----------
    data : pd.DataFrame
        Full time series dataset.
    model_class : type
        Model class to instantiate for each fold (e.g. ``ude.NODE``).
    model_kwargs : dict, optional
        Keyword arguments passed to the model constructor.
    train_kwargs : dict, optional
        Keyword arguments passed to ``model.train()``.
    n_splits : int
        Number of folds. Each validation window is
        ``floor((1 - min_train_fraction) * len(data) / n_splits)`` rows.
    min_train_fraction : float
        Minimum fraction of data used for training in the first fold.
        Default ``0.5`` means the first training set uses at least half
        the data.
    metric : {"mse", "rmse", "mae", "r2_score"}
        Metric to evaluate forecasts against validation observations.
    time_column : str
        Name of the time column in ``data``.

    Returns
    -------
    list of dict, one per fold::

        {
            "fold": int,
            "n_train": int,
            "n_val": int,
            "train_end_time": float,
            "val_end_time": float,
            "val_score": float,          # metric averaged over states
            "state_scores": dict,        # per-state metric values
        }

    Raises
    ------
    ValueError
        If there is not enough data for the requested number of splits.
    """
    from pyUDE.analysis.metrics import score as _score

    if model_kwargs is None:
        model_kwargs = {}
    if train_kwargs is None:
        train_kwargs = {}

    n = len(data)
    min_train = int(n * min_train_fraction)
    remaining = n - min_train

    if remaining < n_splits:
        raise ValueError(
            f"Not enough data for {n_splits} splits with "
            f"min_train_fraction={min_train_fraction}. "
            f"Only {remaining} row(s) remain after the minimum training window "
            f"(need at least {n_splits})."
        )

    step = remaining // n_splits
    state_cols = [c for c in data.columns if c != time_column]
    results = []

    for fold_idx in range(n_splits):
        train_end = min_train + fold_idx * step
        val_end = train_end + step if fold_idx < n_splits - 1 else n

        train_df = data.iloc[:train_end].reset_index(drop=True)
        val_df = data.iloc[train_end:val_end].reset_index(drop=True)
        n_val = len(val_df)

        model = model_class(train_df, **model_kwargs)
        model.train(**train_kwargs)

        forecast_df = model.forecast(steps=n_val)

        # Align lengths (adaptive solver may produce one extra row in rare cases)
        min_len = min(len(forecast_df), n_val)
        forecast_aligned = forecast_df.iloc[:min_len].reset_index(drop=True)
        val_aligned = val_df.iloc[:min_len].reset_index(drop=True)

        scores = _score(val_aligned[state_cols], forecast_aligned[state_cols], metric=metric)

        results.append({
            "fold": fold_idx + 1,
            "n_train": train_end,
            "n_val": n_val,
            "train_end_time": float(train_df[time_column].iloc[-1]),
            "val_end_time": float(val_df[time_column].iloc[-1]),
            "val_score": scores["mean"],
            "state_scores": {k: v for k, v in scores.items() if k != "mean"},
        })

    return results
