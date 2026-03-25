import pandas as pd


def validate_dataframe(data: pd.DataFrame, time_column: str = "time") -> None:
    """Validate that a DataFrame is suitable for use as UDE training data."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data)}")

    if time_column not in data.columns:
        raise ValueError(
            f"time column '{time_column}' not found in DataFrame. "
            f"Available columns: {list(data.columns)}"
        )

    state_cols = [c for c in data.columns if c != time_column]
    if len(state_cols) == 0:
        raise ValueError("DataFrame must have at least one state variable column.")

    if data[time_column].is_monotonic_increasing is False:
        raise ValueError(f"Time column '{time_column}' must be monotonically increasing.")

    if data.isnull().any().any():
        raise ValueError("DataFrame contains NaN values. Please clean data before use.")

    if len(data) < 2:
        raise ValueError("DataFrame must have at least 2 time steps.")
