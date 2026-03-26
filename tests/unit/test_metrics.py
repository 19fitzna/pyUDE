"""Unit tests for pyUDE.analysis.metrics."""

import numpy as np
import pandas as pd
import pytest

from pyUDE.analysis.metrics import mse, rmse, mae, r2_score, score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(arr, cols=None):
    if cols is None:
        cols = [f"state_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


# ---------------------------------------------------------------------------
# Perfect prediction (zero error)
# ---------------------------------------------------------------------------

class TestPerfectPrediction:
    def test_mse_zero(self):
        obs = _df(np.ones((10, 2)))
        result = mse(obs, obs)
        assert result["state_0"] == pytest.approx(0.0)
        assert result["state_1"] == pytest.approx(0.0)
        assert result["mean"] == pytest.approx(0.0)

    def test_rmse_zero(self):
        obs = _df(np.ones((10, 2)))
        result = rmse(obs, obs)
        assert result["mean"] == pytest.approx(0.0)

    def test_mae_zero(self):
        obs = _df(np.ones((10, 2)))
        result = mae(obs, obs)
        assert result["mean"] == pytest.approx(0.0)

    def test_r2_one(self):
        obs = _df(np.random.default_rng(0).random((10, 2)))
        result = r2_score(obs, obs)
        assert result["state_0"] == pytest.approx(1.0)
        assert result["state_1"] == pytest.approx(1.0)
        assert result["mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Known numeric values
# ---------------------------------------------------------------------------

class TestKnownValues:
    def test_mse_known(self):
        obs = np.array([[0.0], [2.0], [4.0]])
        pred = np.array([[1.0], [2.0], [3.0]])
        result = mse(obs, pred)
        # errors: 1, 0, 1 → mean = 2/3
        assert result["state_0"] == pytest.approx(2.0 / 3.0)

    def test_rmse_known(self):
        obs = np.array([[0.0], [2.0], [4.0]])
        pred = np.array([[1.0], [2.0], [3.0]])
        result = rmse(obs, pred)
        assert result["state_0"] == pytest.approx(np.sqrt(2.0 / 3.0))

    def test_mae_known(self):
        obs = np.array([[0.0], [2.0], [4.0]])
        pred = np.array([[1.0], [2.0], [3.0]])
        result = mae(obs, pred)
        assert result["state_0"] == pytest.approx(2.0 / 3.0)

    def test_r2_known(self):
        obs = np.array([[1.0], [2.0], [3.0]])
        pred = np.array([[1.0], [2.0], [3.0]])
        result = r2_score(obs, pred)
        assert result["state_0"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DataFrame vs ndarray consistency
# ---------------------------------------------------------------------------

class TestInputTypes:
    def test_dataframe_and_ndarray_equal(self):
        arr_obs = np.random.default_rng(42).random((20, 2))
        arr_pred = arr_obs + 0.1
        df_obs = _df(arr_obs, ["a", "b"])
        df_pred = _df(arr_pred, ["a", "b"])

        result_arr = rmse(arr_obs, arr_pred)
        result_df = rmse(df_obs, df_pred)

        assert result_arr["state_0"] == pytest.approx(result_df["a"])
        assert result_arr["state_1"] == pytest.approx(result_df["b"])

    def test_1d_array_input(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.1, 3.1])
        result = mse(obs, pred)
        assert "state_0" in result
        assert result["state_0"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# score() dispatcher
# ---------------------------------------------------------------------------

class TestScoreDispatcher:
    @pytest.mark.parametrize("metric", ["mse", "rmse", "mae", "r2_score", "r2"])
    def test_dispatcher_works(self, metric):
        obs = _df(np.random.default_rng(0).random((10, 2)))
        pred = obs.copy()
        result = score(obs, pred, metric=metric)
        assert "mean" in result

    def test_unknown_metric_raises(self):
        obs = _df(np.ones((5, 2)))
        with pytest.raises(ValueError, match="Unknown metric"):
            score(obs, obs, metric="foobar")


# ---------------------------------------------------------------------------
# "mean" key always present
# ---------------------------------------------------------------------------

class TestMeanKey:
    @pytest.mark.parametrize("fn", [mse, rmse, mae, r2_score])
    def test_mean_always_present(self, fn):
        obs = _df(np.ones((5, 3)))
        result = fn(obs, obs)
        assert "mean" in result

    def test_mean_is_average_of_states(self):
        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = np.array([[2.0, 2.0], [4.0, 4.0]])
        result = mse(obs, pred)
        expected_mean = (result["state_0"] + result["state_1"]) / 2
        assert result["mean"] == pytest.approx(expected_mean)


# ---------------------------------------------------------------------------
# Shape mismatch
# ---------------------------------------------------------------------------

class TestShapeMismatch:
    def test_raises_on_shape_mismatch(self):
        obs = np.ones((10, 2))
        pred = np.ones((8, 2))
        with pytest.raises(ValueError, match="shapes do not match"):
            mse(obs, pred)
