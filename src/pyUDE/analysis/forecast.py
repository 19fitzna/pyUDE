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

    if steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}.")

    t_train, u_train = model._get_training_tensors()

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
    u0 = u0.to(device=device, dtype=torch.float64)

    # Use the solver that was used during training for consistency
    solver = getattr(model, '_solver', None) or 'dopri5'

    was_training = model._ode_func.training
    model._ode_func.eval()
    with torch.no_grad():
        u_pred = odeint(model._ode_func, u0, t_forecast, method=solver)
    if was_training:
        model._ode_func.train()

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
    if steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}.")

    t_train, u_train = model._get_training_tensors()
    dt = float(torch.median(t_train[1:] - t_train[:-1]).item())
    t_last = float(t_train[-1].item())

    device = getattr(model, '_device', torch.device('cpu'))
    u = u_train[-1] if initial_state is None else initial_state
    u = u.to(device=device, dtype=torch.float64)
    p = {k: v for k, v in model._param_dict.items()}

    # Pre-compute all forecast times in one tensor
    t_forecast = torch.arange(1, steps + 1, dtype=torch.float64, device=device) * dt + t_last

    was_training = model._network_module.training
    model._network_module.eval()
    states = []
    with torch.no_grad():
        for i in range(steps):
            u = model._known_map(u, p, t_forecast[i]) + model._network_module(u)
            states.append(u)
    if was_training:
        model._network_module.train()

    u_stack = torch.stack(states).cpu().numpy()
    t_out = t_forecast.cpu().numpy()

    df = pd.DataFrame(u_stack, columns=model.state_columns)
    df.insert(0, model.time_column, t_out)
    return df
