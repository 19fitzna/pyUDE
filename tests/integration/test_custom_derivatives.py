"""Integration tests for CustomDerivatives."""

import pytest

pytest.importorskip("torchdiffeq", reason="torchdiffeq required")

import numpy as np
import pandas as pd
import torch

import pyUDE as ude
from pyUDE import CustomDerivatives


@pytest.fixture
def decay_data():
    """dx/dt = -alpha * x, true alpha = 0.5."""
    t = np.linspace(0, 5, 40)
    x = np.exp(-0.5 * t)
    return pd.DataFrame({"time": t, "x": x})


def linear_decay(u, p, t):
    """Known part: -alpha * u."""
    return -p["alpha"] * u


def test_custom_derivatives_trains(decay_data):
    model = CustomDerivatives(
        decay_data,
        known_dynamics=linear_decay,
        init_params={"alpha": 1.0},
        hidden_units=8,
        hidden_layers=1,
    )
    model.train(epochs=100, verbose=False, solver="rk4")
    assert model.is_trained


def test_custom_derivatives_forecast_shape(decay_data):
    model = CustomDerivatives(
        decay_data,
        known_dynamics=linear_decay,
        init_params={"alpha": 1.0},
        hidden_units=8,
    )
    model.train(epochs=50, verbose=False, solver="rk4")
    result = model.forecast(steps=10)
    assert result.shape == (10, 2)


def test_custom_derivatives_get_params(decay_data):
    model = CustomDerivatives(
        decay_data,
        known_dynamics=linear_decay,
        init_params={"alpha": 1.0},
        hidden_units=8,
    )
    model.train(epochs=50, verbose=False, solver="rk4")
    params = model.get_params()
    assert "alpha" in params
    assert isinstance(params["alpha"], float)


def test_custom_derivatives_derivative_matching_loss(decay_data):
    model = CustomDerivatives(
        decay_data,
        known_dynamics=linear_decay,
        init_params={"alpha": 1.0},
        hidden_units=8,
    )
    # derivative_matching loss should not raise
    model.train(epochs=50, verbose=False, loss="derivative_matching")
    assert model.is_trained
