"""Integration tests for NODE."""

import pytest

pytest.importorskip("torchdiffeq", reason="torchdiffeq required for NODE tests")

import numpy as np
import pandas as pd

import pyUDE as ude


@pytest.fixture
def simple_decay():
    """Single-state exponential decay: dx/dt = -0.5 x."""
    t = np.linspace(0, 5, 40)
    x = np.exp(-0.5 * t)
    return pd.DataFrame({"time": t, "x": x})


def test_node_loss_decreases(simple_decay):
    """NODE training should reduce the simulation loss."""
    model = ude.NODE(simple_decay, hidden_units=16, hidden_layers=1)

    import torch
    from torchdiffeq import odeint

    t, u_obs = model._get_training_tensors()
    ode_func = model._build_ode_func()

    # Loss before training
    with torch.no_grad():
        u_pred_before = odeint(ode_func, u_obs[0], t, method="rk4")
    loss_before = float(((u_pred_before - u_obs) ** 2).mean())

    model.train(epochs=200, verbose=False, solver="rk4", loss="simulation")

    # Loss after training
    with torch.no_grad():
        u_pred_after = odeint(model._ode_func, u_obs[0], t, method="rk4")
    loss_after = float(((u_pred_after - u_obs) ** 2).mean())

    assert loss_after < loss_before, "Training should reduce the loss"


def test_node_forecast_shape(simple_decay):
    """forecast() should return a DataFrame with the right shape."""
    model = ude.NODE(simple_decay, hidden_units=16, hidden_layers=1)
    model.train(epochs=50, verbose=False, solver="rk4")

    result = model.forecast(steps=10)
    assert result.shape == (10, 2)  # time + x
    assert "time" in result.columns
    assert "x" in result.columns


def test_node_forecast_time_is_after_training(simple_decay):
    """Forecasted times should be strictly after the last training time."""
    model = ude.NODE(simple_decay, hidden_units=16)
    model.train(epochs=50, verbose=False, solver="rk4")

    result = model.forecast(steps=5)
    t_last_train = simple_decay["time"].iloc[-1]
    assert (result["time"] > t_last_train).all()


def test_node_requires_training_before_forecast(simple_decay):
    model = ude.NODE(simple_decay)
    with pytest.raises(RuntimeError, match="trained"):
        model.forecast(steps=5)


def test_node_get_rhs(simple_decay):
    """get_right_hand_side should return a callable that works with numpy."""
    import numpy as np

    model = ude.NODE(simple_decay, hidden_units=16)
    model.train(epochs=50, verbose=False, solver="rk4")

    rhs = model.get_right_hand_side()
    du = rhs(np.array([1.0]), 0.0)
    assert du.shape == (1,)
