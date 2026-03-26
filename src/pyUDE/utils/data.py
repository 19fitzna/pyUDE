from typing import List, Tuple

import numpy as np
import pandas as pd
import torch


def dataframe_to_tensors(
    data: pd.DataFrame,
    time_column: str = "time",
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Convert a DataFrame to (t, u, state_columns) tensors.

    Returns
    -------
    t : Tensor, shape (T,)
    u : Tensor, shape (T, n_states)
    state_columns : list of column names in order
    """
    state_cols = [c for c in data.columns if c != time_column]
    t = torch.tensor(data[time_column].to_numpy(dtype=np.float64), dtype=torch.float64)
    u = torch.tensor(data[state_cols].to_numpy(dtype=np.float64), dtype=torch.float64)
    return t, u, state_cols


def tensors_to_dataframe(
    t: torch.Tensor,
    u: torch.Tensor,
    state_columns: List[str],
    time_column: str = "time",
) -> pd.DataFrame:
    """Convert (t, u) tensors back to a DataFrame."""
    if u.ndim != 2 or u.shape[1] != len(state_columns):
        raise ValueError(
            f"u has shape {tuple(u.shape)} but {len(state_columns)} "
            f"state columns were provided."
        )
    t_np = t.detach().cpu().numpy()
    u_np = u.detach().cpu().numpy()
    df = pd.DataFrame(u_np, columns=state_columns)
    df.insert(0, time_column, t_np)
    return df
