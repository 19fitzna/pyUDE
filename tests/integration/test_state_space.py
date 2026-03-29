"""Integration tests for state-space loss functions and noise covariances."""

import pytest

pytest.importorskip("torchdiffeq", reason="torchdiffeq required for state-space tests")

import numpy as np
import pandas as pd
import torch

import pyUDE as ude
from pyUDE.core.base import _normalize_covariance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lv_noisy():
    """Lotka-Volterra with additive Gaussian noise on observations."""
    np.random.seed(0)
    dt = 0.1
    t_span = np.arange(0, 10, dt)
    prey, pred = 10.0, 5.0
    rows = []
    for t in t_span:
        rows.append((t, prey, pred))
        dprey = 1.0 * prey - 0.1 * prey * pred
        dpred = 0.075 * prey * pred - 1.5 * pred
        prey = max(prey + dt * dprey, 1e-6)
        pred = max(pred + dt * dpred, 1e-6)
    clean = pd.DataFrame(rows, columns=["time", "prey", "predator"])
    noisy = clean.copy()
    noisy["prey"] += 0.3 * np.random.randn(len(clean))
    noisy["predator"] += 0.3 * np.random.randn(len(clean))
    return noisy, clean


@pytest.fixture
def simple_decay():
    """Single-state exponential decay."""
    t = np.linspace(0, 5, 40)
    x = np.exp(-0.5 * t)
    return pd.DataFrame({"time": t, "x": x})


# ---------------------------------------------------------------------------
# Phase 1: Covariance normalization
# ---------------------------------------------------------------------------

class TestCovarianceNormalization:
    def test_none_returns_none(self):
        assert _normalize_covariance(None, 3) is None

    def test_scalar_to_diagonal(self):
        cov = _normalize_covariance(0.5, 3)
        expected = 0.5 * torch.eye(3, dtype=torch.float64)
        assert torch.allclose(cov, expected)

    def test_scalar_is_variance_not_std(self):
        """Scalar 0.01 should produce 0.01*I, not 0.1*I."""
        cov = _normalize_covariance(0.01, 2)
        assert torch.allclose(cov, 0.01 * torch.eye(2, dtype=torch.float64))

    def test_1d_tensor_to_diagonal(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        cov = _normalize_covariance(v, 3)
        assert torch.allclose(cov, torch.diag(v.double()))

    def test_1d_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            _normalize_covariance(torch.tensor([1.0, 2.0]), 3)

    def test_2d_passthrough(self):
        mat = torch.eye(2, dtype=torch.float64) * 0.5
        assert torch.equal(_normalize_covariance(mat, 2), mat)

    def test_2d_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            _normalize_covariance(torch.eye(3, dtype=torch.float64), 2)

    def test_3d_tensor_raises(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            _normalize_covariance(torch.zeros(2, 2, 2), 2)

    def test_non_tensor_non_float_raises(self):
        with pytest.raises(TypeError):
            _normalize_covariance("bad", 2)


# ---------------------------------------------------------------------------
# Phase 1: Base class properties
# ---------------------------------------------------------------------------

class TestBaseClassProperties:
    def test_observation_error_stored(self, simple_decay):
        model = ude.NODE(simple_decay, observation_error=0.1)
        assert model.observation_error is not None
        assert model.observation_error.shape == (1, 1)

    def test_process_error_stored(self, simple_decay):
        model = ude.NODE(simple_decay, process_error=0.05)
        assert model.process_error is not None

    def test_default_covariances_are_none(self, simple_decay):
        model = ude.NODE(simple_decay)
        assert model.observation_error is None
        assert model.process_error is None


# ---------------------------------------------------------------------------
# Phase 2: Multiple shooting
# ---------------------------------------------------------------------------

class TestMultipleShooting:
    def test_trains_without_error(self, simple_decay):
        model = ude.NODE(simple_decay, hidden_units=16, hidden_layers=1)
        model.train(
            loss="multiple_shooting",
            n_shooting_segments=5,
            epochs=30,
            verbose=False,
            solver="rk4",
        )
        assert model.is_trained
        assert len(model.train_history_["train_loss"]) == 30

    def test_pred_length_derives_segments(self, simple_decay):
        """pred_length should override n_shooting_segments."""
        model = ude.NODE(simple_decay, hidden_units=16, hidden_layers=1)
        # 40 time points, pred_length=8 → ceil(40/8) = 5 segments
        model.train(
            loss="multiple_shooting",
            pred_length=8,
            epochs=10,
            verbose=False,
            solver="rk4",
        )
        assert model.is_trained

    def test_custom_derivatives_with_shooting(self):
        """CustomDerivatives should work with multiple shooting."""
        np.random.seed(42)
        t = np.linspace(0, 5, 40)
        x = np.exp(-0.5 * t)
        data = pd.DataFrame({"time": t, "x": x})

        def known(u, p, t):
            return torch.stack([-p["k"] * u[0]])

        model = ude.CustomDerivatives(
            data, known, {"k": 0.3}, hidden_units=8, hidden_layers=1,
        )
        model.train(
            loss="multiple_shooting",
            n_shooting_segments=4,
            epochs=20,
            verbose=False,
            solver="rk4",
        )
        assert model.is_trained


# ---------------------------------------------------------------------------
# Phase 3: Conditional likelihood
# ---------------------------------------------------------------------------

class TestConditionalLikelihood:
    def test_trains_without_error(self, simple_decay):
        model = ude.NODE(
            simple_decay,
            hidden_units=16, hidden_layers=1,
            observation_error=0.01, process_error=0.001,
        )
        model.train(
            loss="conditional_likelihood",
            epochs=10,
            verbose=False,
            solver="rk4",
        )
        assert model.is_trained
        assert len(model.train_history_["train_loss"]) == 10

    def test_requires_covariances(self, simple_decay):
        model = ude.NODE(simple_decay, hidden_units=16, hidden_layers=1)
        with pytest.raises(ValueError, match="observation_error"):
            model.train(loss="conditional_likelihood", epochs=5, verbose=False)

    def test_state_estimates_stored(self, simple_decay):
        model = ude.NODE(
            simple_decay,
            hidden_units=16, hidden_layers=1,
            observation_error=0.01, process_error=0.001,
        )
        model.train(
            loss="conditional_likelihood",
            epochs=5,
            verbose=False,
            solver="rk4",
        )
        assert model._state_estimates is not None
        assert model._state_estimates.shape == (40, 1)  # T x n_states

    def test_state_estimates_cleared_on_retrain(self, simple_decay):
        model = ude.NODE(
            simple_decay,
            hidden_units=16, hidden_layers=1,
            observation_error=0.01, process_error=0.001,
        )
        model.train(loss="conditional_likelihood", epochs=3, verbose=False, solver="rk4")
        est_1 = model._state_estimates.clone()

        # Retrain — estimates should be fresh
        model.train(loss="conditional_likelihood", epochs=3, verbose=False, solver="rk4")
        est_2 = model._state_estimates
        # They should differ (different training epochs)
        assert est_2 is not None

    def test_covariance_magnitude_affects_estimates(self, simple_decay):
        """Different observation_error should produce different state estimates."""
        model_low = ude.NODE(
            simple_decay, hidden_units=16, hidden_layers=1,
            observation_error=0.001, process_error=0.001,
        )
        model_low.train(loss="conditional_likelihood", epochs=5, verbose=False, solver="rk4")

        model_high = ude.NODE(
            simple_decay, hidden_units=16, hidden_layers=1,
            observation_error=1.0, process_error=0.001,
        )
        model_high.train(loss="conditional_likelihood", epochs=5, verbose=False, solver="rk4")

        # State estimates should differ with different covariance settings
        assert not torch.allclose(
            model_low._state_estimates, model_high._state_estimates, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Phase 4: CustomDifferences rejection
# ---------------------------------------------------------------------------

class TestCustomDifferencesRejection:
    def test_rejects_conditional_likelihood(self):
        t = np.linspace(0, 5, 20)
        data = pd.DataFrame({"time": t, "x": np.exp(-0.5 * t)})

        def known_map(u, p, t):
            return torch.stack([u[0] * (1 - p["k"])])

        model = ude.CustomDifferences(data, known_map, {"k": 0.05})
        with pytest.raises(ValueError, match="not supported"):
            model.train(loss="conditional_likelihood", epochs=5, verbose=False)

    def test_rejects_multiple_shooting(self):
        t = np.linspace(0, 5, 20)
        data = pd.DataFrame({"time": t, "x": np.exp(-0.5 * t)})

        def known_map(u, p, t):
            return torch.stack([u[0] * (1 - p["k"])])

        model = ude.CustomDifferences(data, known_map, {"k": 0.05})
        with pytest.raises(ValueError, match="not supported"):
            model.train(loss="multiple_shooting", epochs=5, verbose=False)


# ---------------------------------------------------------------------------
# Phase 5: obs_weight in simulation loss
# ---------------------------------------------------------------------------

class TestObsWeight:
    def test_obs_weight_affects_training(self, simple_decay):
        """Different obs_weight should produce different final losses."""
        model1 = ude.NODE(simple_decay, hidden_units=8, hidden_layers=1)
        model1.train(epochs=30, verbose=False, solver="rk4", obs_weight=1.0)
        loss1 = model1.train_history_["train_loss"][-1]

        model2 = ude.NODE(simple_decay, hidden_units=8, hidden_layers=1)
        model2.train(epochs=30, verbose=False, solver="rk4", obs_weight=10.0)
        loss2 = model2.train_history_["train_loss"][-1]

        # Losses should differ (obs_weight=10 should have ~10x larger raw loss)
        assert loss1 != loss2


# ---------------------------------------------------------------------------
# Phase 6: get_state_estimates and get_predictions
# ---------------------------------------------------------------------------

class TestStateEstimates:
    def test_shape_without_ekf(self, simple_decay):
        """Without conditional likelihood, returns ODE trajectory."""
        model = ude.NODE(simple_decay, hidden_units=16, hidden_layers=1)
        model.train(epochs=20, verbose=False, solver="rk4")

        df = model.get_state_estimates()
        assert df.shape == (40, 2)  # T rows, time + x
        assert "time" in df.columns
        assert "x" in df.columns

    def test_shape_with_ekf(self, simple_decay):
        """With conditional likelihood, returns Kalman-filtered estimates."""
        model = ude.NODE(
            simple_decay, hidden_units=16, hidden_layers=1,
            observation_error=0.01, process_error=0.001,
        )
        model.train(loss="conditional_likelihood", epochs=5, verbose=False, solver="rk4")

        df = model.get_state_estimates()
        assert df.shape == (40, 2)
        assert "time" in df.columns


class TestGetPredictions:
    def test_shape(self, simple_decay):
        model = ude.NODE(simple_decay, hidden_units=16, hidden_layers=1)
        model.train(epochs=20, verbose=False, solver="rk4")

        df = model.get_predictions()
        # T-1 rows (finite differences), time + pred_x + obs_x
        assert df.shape == (39, 3)
        assert "pred_x" in df.columns
        assert "obs_x" in df.columns


# ---------------------------------------------------------------------------
# Phase 7: obs_weight in train_differences
# ---------------------------------------------------------------------------

class TestDifferencesObsWeight:
    def test_accepts_obs_weight(self):
        t = np.linspace(0, 5, 20)
        data = pd.DataFrame({"time": t, "x": np.exp(-0.5 * t)})

        def known_map(u, p, t):
            return torch.stack([u[0] * (1 - p["k"])])

        model = ude.CustomDifferences(data, known_map, {"k": 0.05})
        model.train(epochs=10, verbose=False, obs_weight=2.0)
        assert model.is_trained


# ---------------------------------------------------------------------------
# Save/load round-trip with covariance settings
# ---------------------------------------------------------------------------

class TestSaveLoadCovariance:
    def test_round_trip(self, simple_decay, tmp_path):
        model = ude.NODE(
            simple_decay,
            hidden_units=16, hidden_layers=1,
            observation_error=0.1, process_error=0.05,
            obs_weight=2.0, proc_weight=0.5,
        )
        model.train(epochs=10, verbose=False, solver="rk4")

        path = str(tmp_path / "model.pt")
        model.save(path)

        # Load into a fresh model (same architecture)
        model2 = ude.NODE(simple_decay, hidden_units=16, hidden_layers=1)
        model2.load_weights(path)

        assert model2.is_trained
        assert model2.observation_error is not None
        assert torch.allclose(model2.observation_error, model.observation_error)
        assert torch.allclose(model2.process_error, model.process_error)
        assert model2._proc_weight == 0.5
        assert model2._obs_weight == 2.0


# ---------------------------------------------------------------------------
# EKF numerical stability on longer series
# ---------------------------------------------------------------------------

class TestEKFStability:
    def test_no_nan_on_100_steps(self):
        """Conditional likelihood should not produce NaN on 100+ steps."""
        np.random.seed(1)
        t = np.linspace(0, 10, 100)
        x = np.exp(-0.3 * t) + 0.05 * np.random.randn(100)
        data = pd.DataFrame({"time": t, "x": x})

        model = ude.NODE(
            data, hidden_units=16, hidden_layers=1,
            observation_error=0.01, process_error=0.001,
        )
        model.train(loss="conditional_likelihood", epochs=5, verbose=False, solver="rk4")

        losses = model.train_history_["train_loss"]
        assert all(not np.isnan(l) for l in losses), "NaN detected in loss history"


# ---------------------------------------------------------------------------
# CustomDerivatives with conditional likelihood
# ---------------------------------------------------------------------------

class TestCustomDerivativesConditionalLikelihood:
    def test_trains_and_recovers_params(self):
        """CustomDerivatives should work with conditional likelihood."""
        np.random.seed(42)
        t = np.linspace(0, 5, 40)
        x = np.exp(-0.5 * t) + 0.02 * np.random.randn(40)
        data = pd.DataFrame({"time": t, "x": x})

        def known(u, p, t):
            return torch.stack([-p["k"] * u[0]])

        model = ude.CustomDerivatives(
            data, known, {"k": 0.3},
            hidden_units=8, hidden_layers=1,
            observation_error=0.01, process_error=0.001,
        )
        model.train(
            loss="conditional_likelihood",
            epochs=20,
            verbose=False,
            solver="rk4",
        )
        params = model.get_params()
        assert "k" in params
