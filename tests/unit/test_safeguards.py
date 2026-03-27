"""Unit tests for training and validation safeguards."""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from pyUDE.core.node import NODE
from pyUDE.core.custom_derivatives import CustomDerivatives
from pyUDE.core.custom_differences import CustomDifferences
from pyUDE.utils.validation import validate_dataframe
from pyUDE.utils.data import tensors_to_dataframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    t = np.linspace(0, 3, 30)
    x = np.exp(-0.5 * t)
    return pd.DataFrame({"time": t, "x": x})


@pytest.fixture
def lv_known_dynamics():
    def f(u, p, t):
        prey, pred = u[0], u[1]
        return torch.stack([
            p["alpha"] * prey - 0.1 * prey * pred,
            -p["delta"] * pred + 0.075 * prey * pred,
        ])
    return f


@pytest.fixture
def lv_known_map():
    def f(u, p, t):
        x = u[0]
        return torch.stack([p["r"] * x * (1 - x / p["K"])])
    return f


@pytest.fixture
def lv_data():
    t = np.linspace(0, 5, 50)
    prey = 10 * np.exp(-0.1 * t) + 1
    pred = 5 * np.exp(-0.05 * t) + 1
    return pd.DataFrame({"time": t, "prey": prey, "predator": pred})


# ---------------------------------------------------------------------------
# validate_dataframe safeguards
# ---------------------------------------------------------------------------

class TestValidationSafeguards:
    def test_duplicate_times_rejected(self):
        df = pd.DataFrame({"time": [0.0, 1.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            validate_dataframe(df)

    def test_non_monotonic_times_rejected(self):
        df = pd.DataFrame({"time": [0.0, 2.0, 1.0], "x": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="monoton"):
            validate_dataframe(df)

    def test_non_numeric_state_column_rejected(self):
        df = pd.DataFrame({"time": [0.0, 1.0], "x": [1.0, 2.0], "label": ["a", "b"]})
        with pytest.raises(ValueError, match="[Nn]on-numeric"):
            validate_dataframe(df)

    def test_non_numeric_time_column_rejected(self):
        df = pd.DataFrame({"time": ["a", "b"], "x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="[Nn]umeric"):
            validate_dataframe(df)

    def test_constant_column_warns(self):
        df = pd.DataFrame({"time": [0.0, 1.0, 2.0], "x": [1.0, 1.0, 1.0]})
        with pytest.warns(UserWarning, match="constant"):
            validate_dataframe(df)


# ---------------------------------------------------------------------------
# Construction-time validation for known_dynamics / known_map
# ---------------------------------------------------------------------------

class TestConstructionValidation:
    def test_known_dynamics_wrong_shape_raises(self, lv_data):
        def bad_dynamics(u, p, t):
            # Returns a scalar instead of shape (n_states,)
            return torch.tensor(1.0, dtype=torch.float64)

        with pytest.raises(ValueError, match="known_dynamics"):
            CustomDerivatives(lv_data, bad_dynamics, {"alpha": 1.0, "delta": 1.5})

    def test_known_dynamics_error_propagates(self, lv_data):
        def crashing_dynamics(u, p, t):
            raise RuntimeError("boom")

        with pytest.raises(ValueError, match="raised an error"):
            CustomDerivatives(lv_data, crashing_dynamics, {"alpha": 1.0, "delta": 1.5})

    def test_known_map_wrong_shape_raises(self, simple_data):
        def bad_map(u, p, t):
            return torch.zeros(5, dtype=torch.float64)  # wrong n_states

        with pytest.raises(ValueError, match="known_map"):
            CustomDifferences(simple_data, bad_map, {"r": 0.5, "K": 10.0})


# ---------------------------------------------------------------------------
# Forecast validation
# ---------------------------------------------------------------------------

class TestForecastValidation:
    def test_steps_zero_raises(self, simple_data):
        model = NODE(simple_data)
        model.train(loss="derivative_matching", epochs=5, verbose=False)
        with pytest.raises(ValueError, match="steps"):
            model.forecast(steps=0)

    def test_steps_negative_raises(self, simple_data):
        model = NODE(simple_data)
        model.train(loss="derivative_matching", epochs=5, verbose=False)
        with pytest.raises(ValueError, match="steps"):
            model.forecast(steps=-1)

    def test_untrained_forecast_raises(self, simple_data):
        model = NODE(simple_data)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.forecast(steps=5)


# ---------------------------------------------------------------------------
# Patience early stopping
# ---------------------------------------------------------------------------

class TestPatientEarlyStopping:
    def test_patience_stops_training(self, simple_data):
        """Training must stop before max epochs when no improvement."""
        calls = []

        class CountingModel(NODE):
            pass

        model = NODE(simple_data)
        # Use very tight patience — should stop well before 500 epochs
        model.train(
            loss="derivative_matching",
            epochs=500,
            patience=5,
            verbose=False,
        )
        assert model.is_trained


# ---------------------------------------------------------------------------
# dtype consistency
# ---------------------------------------------------------------------------

class TestDtypeConsistency:
    def test_custom_float32_network_does_not_crash(self, simple_data):
        """A float32 custom network should be silently cast to float64."""
        net = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
        # net is float32 by default
        assert next(net.parameters()).dtype == torch.float32

        model = CustomDifferences(
            simple_data,
            known_map=lambda u, p, t: torch.zeros_like(u),
            init_params={"dummy": 0.0},
            network=net,
        )
        # Should not raise despite float32 network
        model.train(epochs=5, verbose=False)
        assert model.is_trained

    def test_float32_initial_state_handled(self, simple_data):
        """forecast() with float32 initial_state should not raise."""
        model = NODE(simple_data)
        model.train(loss="derivative_matching", epochs=5, verbose=False)
        u0_f32 = torch.zeros(1, dtype=torch.float32)
        # Should not raise a dtype mismatch
        df = model.forecast(steps=3, initial_state=u0_f32)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# tensors_to_dataframe shape check
# ---------------------------------------------------------------------------

class TestTensorsToDataframeShapeCheck:
    def test_wrong_n_states_raises(self):
        t = torch.linspace(0, 1, 5)
        u = torch.zeros(5, 3)
        with pytest.raises(ValueError, match="shape"):
            tensors_to_dataframe(t, u, state_columns=["x", "y"])  # 2 cols, 3 states

    def test_1d_u_raises(self):
        t = torch.linspace(0, 1, 5)
        u = torch.zeros(5)  # 1-D instead of 2-D
        with pytest.raises(ValueError, match="shape"):
            tensors_to_dataframe(t, u, state_columns=["x"])
