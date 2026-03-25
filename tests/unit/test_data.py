"""Unit tests for data utilities."""

import numpy as np
import pandas as pd
import pytest
import torch

from pyUDE.utils.data import dataframe_to_tensors, tensors_to_dataframe
from pyUDE.utils.validation import validate_dataframe


class TestDataframeToTensors:
    def test_basic_conversion(self):
        df = pd.DataFrame({"time": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        t, u, cols = dataframe_to_tensors(df)
        assert t.shape == (3,)
        assert u.shape == (3, 2)
        assert cols == ["x", "y"]

    def test_custom_time_column(self):
        df = pd.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0]})
        t, u, cols = dataframe_to_tensors(df, time_column="t")
        assert cols == ["x"]

    def test_dtype_is_float64(self):
        df = pd.DataFrame({"time": [0.0, 1.0], "x": [1.0, 2.0]})
        t, u, _ = dataframe_to_tensors(df)
        assert t.dtype == torch.float64
        assert u.dtype == torch.float64


class TestTensorsToDataframe:
    def test_roundtrip(self):
        df_in = pd.DataFrame({"time": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0]})
        t, u, cols = dataframe_to_tensors(df_in)
        df_out = tensors_to_dataframe(t, u, cols)
        pd.testing.assert_frame_equal(df_in, df_out)


class TestValidateDataframe:
    def test_valid_df(self):
        df = pd.DataFrame({"time": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0]})
        validate_dataframe(df)  # should not raise

    def test_missing_time_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="time column"):
            validate_dataframe(df)

    def test_no_state_columns(self):
        df = pd.DataFrame({"time": [0.0, 1.0]})
        with pytest.raises(ValueError, match="at least one state"):
            validate_dataframe(df)

    def test_nan_values(self):
        df = pd.DataFrame({"time": [0.0, 1.0], "x": [1.0, float("nan")]})
        with pytest.raises(ValueError, match="NaN"):
            validate_dataframe(df)

    def test_too_few_rows(self):
        df = pd.DataFrame({"time": [0.0], "x": [1.0]})
        with pytest.raises(ValueError, match="at least 2"):
            validate_dataframe(df)

    def test_not_dataframe(self):
        with pytest.raises(TypeError):
            validate_dataframe([[0, 1], [1, 2]])
