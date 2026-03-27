"""Universal Differential Equations in Python."""

__version__ = "0.0.2"
__description__ = "Universal Differential Equations in Python"

from pyUDE.core.node import NODE
from pyUDE.core.custom_derivatives import CustomDerivatives
from pyUDE.core.custom_differences import CustomDifferences
from pyUDE.analysis.forecast import forecast
from pyUDE.analysis.dynamics import get_right_hand_side
from pyUDE.training.trainer import TrainResult

__all__ = [
    "NODE",
    "CustomDerivatives",
    "CustomDifferences",
    "forecast",
    "get_right_hand_side",
    "TrainResult",
]

# Julia backend — only available when juliacall is installed
try:
    from pyUDE.julia import JuliaNODE, JuliaCustomDerivatives, JuliaCustomDifferences
    __all__ += ["JuliaNODE", "JuliaCustomDerivatives", "JuliaCustomDifferences"]
except Exception:
    pass  # juliacall not installed or Julia environment not set up
