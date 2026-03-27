"""Shared base class for Julia-backed UDE models."""

from typing import List

import pandas as pd


class JuliaModelBase:
    """Shared properties and helpers for all Julia-backed models."""

    _data: pd.DataFrame
    _time_column: str
    _state_columns: List[str]
    _n_states: int
    _is_trained: bool

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

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (
            f"{self.__class__.__name__}("
            f"states={self._n_states}, "
            f"columns={self._state_columns}, "
            f"{status})"
        )
