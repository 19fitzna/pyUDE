"""Unit tests for pyUDE.utils.splitting."""

import numpy as np
import pandas as pd
import pytest

import pyUDE as ude
from pyUDE.utils.splitting import train_test_split, time_series_cv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data(n=50):
    t = np.linspace(0, 5, n)
    return pd.DataFrame({
        "time": t,
        "x": np.sin(t),
        "y": np.cos(t),
    })


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------

class TestTrainTestSplit:
    def test_preserves_order(self):
        data = _make_data(50)
        train, test = train_test_split(data, test_fraction=0.2)
        assert list(train["time"]) == list(data["time"][:40])
        assert list(test["time"]) == list(data["time"][40:])

    def test_resets_index(self):
        data = _make_data(50)
        train, test = train_test_split(data, test_fraction=0.2)
        assert list(train.index) == list(range(len(train)))
        assert list(test.index) == list(range(len(test)))

    def test_correct_sizes(self):
        data = _make_data(100)
        train, test = train_test_split(data, test_fraction=0.3)
        assert len(train) + len(test) == 100
        assert len(test) == 30

    def test_invalid_fraction_zero(self):
        with pytest.raises(ValueError, match="test_fraction"):
            train_test_split(_make_data(), test_fraction=0.0)

    def test_invalid_fraction_one(self):
        with pytest.raises(ValueError, match="test_fraction"):
            train_test_split(_make_data(), test_fraction=1.0)

    def test_invalid_fraction_negative(self):
        with pytest.raises(ValueError, match="test_fraction"):
            train_test_split(_make_data(), test_fraction=-0.1)

    def test_raises_if_train_too_small(self):
        # 4 rows, test_fraction=0.6 → 2 test rows, 2 train rows (border OK)
        # test_fraction=0.75 → 3 test, 1 train → ValueError
        data = _make_data(4)
        with pytest.raises(ValueError, match="Training split"):
            train_test_split(data, test_fraction=0.75)

    def test_raises_if_test_too_small(self):
        # 4 rows, test_fraction=0.1 → int(0.4)=0, max(1, 0)=1 test row → ValueError
        data = _make_data(4)
        with pytest.raises(ValueError, match="Test split"):
            train_test_split(data, test_fraction=0.1)

    def test_returns_dataframes(self):
        data = _make_data(30)
        train, test = train_test_split(data)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_no_data_loss(self):
        data = _make_data(40)
        train, test = train_test_split(data, test_fraction=0.25)
        combined = pd.concat([train, test], ignore_index=True)
        assert len(combined) == len(data)


# ---------------------------------------------------------------------------
# time_series_cv
# ---------------------------------------------------------------------------

class TestTimeSeriesCV:
    def test_returns_n_splits_results(self):
        data = _make_data(60)
        results = time_series_cv(
            data,
            model_class=ude.NODE,
            model_kwargs={"hidden_units": 8, "hidden_layers": 1},
            train_kwargs={"loss": "derivative_matching", "epochs": 10, "verbose": False},
            n_splits=3,
        )
        assert len(results) == 3

    def test_fold_numbers_sequential(self):
        data = _make_data(60)
        results = time_series_cv(
            data,
            model_class=ude.NODE,
            model_kwargs={"hidden_units": 8, "hidden_layers": 1},
            train_kwargs={"loss": "derivative_matching", "epochs": 5, "verbose": False},
            n_splits=3,
        )
        assert [r["fold"] for r in results] == [1, 2, 3]

    def test_result_keys_present(self):
        data = _make_data(60)
        results = time_series_cv(
            data,
            model_class=ude.NODE,
            model_kwargs={"hidden_units": 8, "hidden_layers": 1},
            train_kwargs={"loss": "derivative_matching", "epochs": 5, "verbose": False},
            n_splits=2,
        )
        required_keys = {"fold", "n_train", "n_val", "train_end_time", "val_end_time",
                         "val_score", "state_scores"}
        for r in results:
            assert required_keys.issubset(r.keys())

    def test_train_sizes_expand(self):
        data = _make_data(60)
        results = time_series_cv(
            data,
            model_class=ude.NODE,
            model_kwargs={"hidden_units": 8, "hidden_layers": 1},
            train_kwargs={"loss": "derivative_matching", "epochs": 5, "verbose": False},
            n_splits=3,
        )
        train_sizes = [r["n_train"] for r in results]
        assert train_sizes == sorted(train_sizes), "Train sizes should be non-decreasing"

    def test_val_score_is_float(self):
        data = _make_data(60)
        results = time_series_cv(
            data,
            model_class=ude.NODE,
            model_kwargs={"hidden_units": 8, "hidden_layers": 1},
            train_kwargs={"loss": "derivative_matching", "epochs": 5, "verbose": False},
            n_splits=2,
        )
        for r in results:
            assert isinstance(r["val_score"], float)
            assert r["val_score"] >= 0

    def test_raises_if_not_enough_data(self):
        data = _make_data(10)
        with pytest.raises(ValueError, match="Not enough data"):
            time_series_cv(
                data,
                model_class=ude.NODE,
                train_kwargs={"loss": "derivative_matching", "epochs": 5, "verbose": False},
                n_splits=20,
                min_train_fraction=0.9,
            )
