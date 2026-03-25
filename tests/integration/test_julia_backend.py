"""
Integration tests for the Julia backend.

All tests are automatically skipped when juliacall is not installed or when
Julia / UniversalDiffEq.jl are not available.  Run them explicitly with:

    pytest tests/integration/test_julia_backend.py -v

Requirements:
  - pip install 'pyUDE[julia]'
  - Julia >= 1.10 with UniversalDiffEq.jl, Lux.jl, OrdinaryDiffEq.jl
"""

import numpy as np
import pandas as pd
import pytest

# Skip entire module if juliacall is not installed
juliacall = pytest.importorskip("juliacall", reason="juliacall not installed")

# Try to actually start Julia and load UniversalDiffEq; skip if it fails
try:
    from pyUDE.julia._env import get_julia
    _jl, _UDE = get_julia()
    _JULIA_AVAILABLE = True
except Exception as exc:
    _JULIA_AVAILABLE = False
    _JULIA_SKIP_REASON = str(exc)

pytestmark = pytest.mark.skipif(
    not _JULIA_AVAILABLE,
    reason=_JULIA_SKIP_REASON if not _JULIA_AVAILABLE else "",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def decay_data():
    """Single-state exponential decay: dx/dt = -0.5 x."""
    t = np.linspace(0, 5, 40)
    x = np.exp(-0.5 * t)
    return pd.DataFrame({"time": t, "x": x})


@pytest.fixture
def lv_data():
    """Two-state Lotka-Volterra time series."""
    np.random.seed(0)
    t = np.linspace(0, 8, 60)
    dt = t[1] - t[0]
    prey, pred = 10.0, 5.0
    rows = []
    for ti in t:
        rows.append((ti, prey, pred))
        prey = max(prey + dt * (1.0 * prey - 0.1 * prey * pred), 1e-3)
        pred = max(pred + dt * (0.075 * prey * pred - 1.5 * pred), 1e-3)
    return pd.DataFrame(rows, columns=["time", "prey", "predator"])


# ---------------------------------------------------------------------------
# JuliaNODE tests
# ---------------------------------------------------------------------------

class TestJuliaNODE:
    def test_import(self):
        from pyUDE import JuliaNODE
        assert JuliaNODE is not None

    def test_construct(self, decay_data):
        from pyUDE import JuliaNODE
        model = JuliaNODE(decay_data)
        assert model.n_states == 1
        assert model.state_columns == ["x"]
        assert not model.is_trained

    def test_requires_training_before_forecast(self, decay_data):
        from pyUDE import JuliaNODE
        model = JuliaNODE(decay_data)
        with pytest.raises(RuntimeError, match="trained"):
            model.forecast(steps=5)

    def test_train(self, decay_data):
        from pyUDE import JuliaNODE
        model = JuliaNODE(decay_data, hidden_units=16, hidden_layers=1)
        result = model.train(epochs=100, verbose=False)
        assert result is model  # fluent
        assert model.is_trained

    def test_forecast_shape(self, decay_data):
        from pyUDE import JuliaNODE
        model = JuliaNODE(decay_data, hidden_units=16)
        model.train(epochs=50, verbose=False)
        fc = model.forecast(steps=10)
        assert fc.shape == (10, 2)  # time + x
        assert "time" in fc.columns
        assert "x" in fc.columns

    def test_forecast_time_after_training(self, decay_data):
        from pyUDE import JuliaNODE
        model = JuliaNODE(decay_data, hidden_units=16)
        model.train(epochs=50, verbose=False)
        fc = model.forecast(steps=5)
        t_last = decay_data["time"].iloc[-1]
        assert (fc["time"] > t_last).all()

    def test_get_right_hand_side(self, decay_data):
        from pyUDE import JuliaNODE
        model = JuliaNODE(decay_data, hidden_units=16)
        model.train(epochs=50, verbose=False)
        rhs = model.get_right_hand_side()
        du = rhs(np.array([1.0]), 0.0)
        assert du.shape == (1,)

    def test_two_state(self, lv_data):
        from pyUDE import JuliaNODE
        model = JuliaNODE(lv_data, hidden_units=16)
        model.train(epochs=50, verbose=False)
        fc = model.forecast(steps=5)
        assert set(fc.columns) == {"time", "prey", "predator"}


# ---------------------------------------------------------------------------
# JuliaCustomDerivatives tests
# ---------------------------------------------------------------------------

class TestJuliaCustomDerivatives:
    def test_train(self, decay_data):
        from pyUDE import JuliaCustomDerivatives

        def linear_decay(u, p, t):
            return [-p["alpha"] * u[0]]

        model = JuliaCustomDerivatives(
            decay_data,
            known_dynamics=linear_decay,
            init_params={"alpha": 1.0},
            hidden_units=8, hidden_layers=1,
        )
        model.train(epochs=100, verbose=False)
        assert model.is_trained

    def test_forecast_shape(self, decay_data):
        from pyUDE import JuliaCustomDerivatives

        def linear_decay(u, p, t):
            return [-p["alpha"] * u[0]]

        model = JuliaCustomDerivatives(
            decay_data, linear_decay, {"alpha": 1.0}, hidden_units=8,
        )
        model.train(epochs=50, verbose=False)
        fc = model.forecast(steps=8)
        assert fc.shape == (8, 2)

    def test_get_params(self, decay_data):
        from pyUDE import JuliaCustomDerivatives

        def linear_decay(u, p, t):
            return [-p["alpha"] * u[0]]

        model = JuliaCustomDerivatives(
            decay_data, linear_decay, {"alpha": 1.0}, hidden_units=8,
        )
        model.train(epochs=50, verbose=False)
        params = model.get_params()
        assert "alpha" in params
        assert isinstance(params["alpha"], float)

    def test_python_callback_works(self, lv_data):
        """Verify Python known_dynamics is correctly called from Julia's ODE solver."""
        from pyUDE import JuliaCustomDerivatives

        call_count = []

        def lv_known(u, p, t):
            call_count.append(1)
            return [p["alpha"] * u[0], -p["delta"] * u[1]]

        model = JuliaCustomDerivatives(
            lv_data, lv_known, {"alpha": 0.8, "delta": 1.2},
            hidden_units=8, hidden_layers=1,
        )
        model.train(epochs=20, verbose=False)
        assert len(call_count) > 0, "Python known_dynamics was never called"


# ---------------------------------------------------------------------------
# JuliaCustomDifferences tests
# ---------------------------------------------------------------------------

class TestJuliaCustomDifferences:
    def test_train(self):
        from pyUDE import JuliaCustomDifferences
        t = np.arange(50, dtype=float)
        x = 3.6 * np.linspace(0.1, 0.9, 50)  # approximate logistic
        data = pd.DataFrame({"time": t, "x": x})

        def logistic(u, p, t):
            return [p["r"] * u[0] * (1 - u[0])]

        model = JuliaCustomDifferences(
            data, logistic, {"r": 3.0}, hidden_units=8, hidden_layers=1,
        )
        model.train(epochs=50, verbose=False)
        assert model.is_trained

    def test_forecast_shape(self):
        from pyUDE import JuliaCustomDifferences
        t = np.arange(50, dtype=float)
        x = [0.5]
        for _ in range(49):
            x.append(3.6 * x[-1] * (1 - x[-1]))
        data = pd.DataFrame({"time": t, "x": x})

        def logistic(u, p, t):
            return [p["r"] * u[0] * (1 - u[0])]

        model = JuliaCustomDifferences(data, logistic, {"r": 3.0}, hidden_units=8)
        model.train(epochs=30, verbose=False)
        fc = model.forecast(steps=10)
        assert fc.shape == (10, 2)
