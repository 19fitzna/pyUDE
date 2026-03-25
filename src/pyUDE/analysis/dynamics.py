"""Utilities for inspecting learned dynamics."""

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from pyUDE.core.base import UDEModel


def get_right_hand_side(model: "UDEModel") -> Callable:
    """
    Return a callable ``f(u, t) -> du`` representing the learned dynamics.

    The returned function accepts numpy arrays (or lists) and returns a
    numpy array, making it compatible with ``scipy.integrate.odeint`` and
    similar tools.

    Parameters
    ----------
    model : UDEModel
        A trained continuous-time model.

    Returns
    -------
    callable
        ``f(u, t) -> du`` where ``u`` and ``du`` are numpy arrays of shape
        ``(n_states,)``.

    Examples
    --------
    >>> rhs = model.get_right_hand_side()
    >>> from scipy.integrate import odeint
    >>> sol = odeint(rhs, y0=[1.0, 0.5], t=t_span)
    """
    ode_func = model._ode_func

    def f(u, t):
        u_t = torch.tensor(np.asarray(u, dtype=np.float64), dtype=torch.float64)
        t_t = torch.tensor(float(t), dtype=torch.float64)
        with torch.no_grad():
            du = ode_func(t_t, u_t)
        return du.numpy()

    return f
