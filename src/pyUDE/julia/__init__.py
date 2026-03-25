"""
pyUDE Julia backend — requires ``pip install 'pyUDE[julia]'`` and a Julia
installation with UniversalDiffEq.jl available.

Julia starts up lazily on first model construction, so importing this module
is fast.
"""

from pyUDE.julia.node import JuliaNODE
from pyUDE.julia.custom_derivatives import JuliaCustomDerivatives
from pyUDE.julia.custom_differences import JuliaCustomDifferences

__all__ = ["JuliaNODE", "JuliaCustomDerivatives", "JuliaCustomDifferences"]
