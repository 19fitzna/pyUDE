"""Universal Differential Equations in Python."""

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.0.2"
__description__ = "Universal Differential Equations in Python"

from pyUDE.core.node import NODE
from pyUDE.core.custom_derivatives import CustomDerivatives
from pyUDE.core.custom_differences import CustomDifferences
from pyUDE.analysis.forecast import forecast
from pyUDE.analysis.dynamics import get_right_hand_side
from pyUDE.utils.splitting import train_test_split, time_series_cv
from pyUDE.analysis.metrics import score, mse, rmse, mae, r2_score

__all__ = [
    "NODE",
    "CustomDerivatives",
    "CustomDifferences",
    "forecast",
    "get_right_hand_side",
    "train_test_split",
    "time_series_cv",
    "score",
    "mse",
    "rmse",
    "mae",
    "r2_score",
]

# Julia backend — only available when juliacall is installed
try:
    from pyUDE.julia import JuliaNODE, JuliaCustomDerivatives, JuliaCustomDifferences
    __all__ += ["JuliaNODE", "JuliaCustomDerivatives", "JuliaCustomDifferences"]
except ImportError:
    pass  # juliacall not installed or Julia environment not set up
