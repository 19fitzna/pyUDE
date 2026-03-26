"""Forward-simulation / forecasting utilities."""

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from pyUDE.core.base import UDEModel
    from pyUDE.core.custom_differences import CustomDifferences


def forecast(
    model: "UDEModel",
    steps: int,
    dt: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    """
    Integrate the learned ODE forward from the last observed state.

    Parameters
    ----------
    model : UDEModel
        A trained continuous-time model.
    steps : int
        Number of additional time steps to predict.
    dt : float, optional
        Step size. Defaults to the median interval in the training data.
    initial_state : Tensor, optional
        1-D tensor of shape ``(n_states,)``. Defaults to the last training row.

    Returns
    -------
    pd.DataFrame
        Columns: [time_column, *state_columns]. Does NOT include the
        initial state row — only the forecasted steps.
    """
    try:
        from torchdiffeq import odeint
    except ImportError as e:
        raise ImportError(
            "torchdiffeq is required for forecasting. "
            "Install it with: pip install torchdiffeq"
        ) from e

    t_train, u_train, _ = model._get_training_tensors()

    if dt is None:
        dt = float(torch.median(t_train[1:] - t_train[:-1]).item())

    t_last = float(t_train[-1].item())
    t_forecast = torch.tensor(
        [t_last + dt * i for i in range(steps + 1)], dtype=torch.float64
    )

    u0 = u_train[-1] if initial_state is None else initial_state

    # Move to the device the model was trained on
    device = getattr(model, '_device', torch.device('cpu'))
    t_forecast = t_forecast.to(device)
    u0 = u0.to(device)

    # Use the solver that was used during training for consistency
    solver = getattr(model, '_solver', None) or 'dopri5'

    with torch.no_grad():
        u_pred = odeint(model._ode_func, u0, t_forecast, method=solver)

    # Drop the first point (= last training observation); move to CPU for numpy
    t_out = t_forecast[1:].cpu()
    u_out = u_pred[1:].cpu()

    df = pd.DataFrame(
        u_out.numpy(), columns=model.state_columns
    )
    df.insert(0, model.time_column, t_out.numpy())
    return df


def forecast_differences(
    model: "CustomDifferences",
    steps: int,
    initial_state: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    """
    Step the discrete map forward from the last observed state.

    Parameters
    ----------
    model : CustomDifferences
        A trained discrete-time model.
    steps : int
    initial_state : Tensor, optional

    Returns
    -------
    pd.DataFrame
    """
    t_train, u_train, _ = model._get_training_tensors()
    dt = float(torch.median(t_train[1:] - t_train[:-1]).item())
    t_last = float(t_train[-1].item())

    device = getattr(model, '_device', torch.device('cpu'))
    u = u_train[-1] if initial_state is None else initial_state
    u = u.to(device)
    p = {k: v for k, v in model._param_dict.items()}

    states = []
    times = []
    with torch.no_grad():
        for i in range(1, steps + 1):
            t_next = torch.tensor(t_last + dt * i, dtype=torch.float64, device=device)
            u = model._known_map(u, p, t_next) + model._network_module(u)
            states.append(u.cpu().numpy())
            times.append(float(t_next.item()))

    df = pd.DataFrame(np.stack(states), columns=model.state_columns)
    df.insert(0, model.time_column, times)
    return df
