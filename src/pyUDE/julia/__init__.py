"""
pyUDE Julia backend — requires ``pip install 'pyUDE[julia]'`` and a Julia
installation with UniversalDiffEq.jl available.

Julia starts up lazily on first model construction, so importing this module
is fast.

.. warning::
    **Thread safety**: Julia is single-threaded. Do not call Julia-backed
    model methods (``train``, ``forecast``, ``get_params``, etc.) concurrently
    from multiple Python threads. Doing so can corrupt state or cause a
    segfault. Use a single thread (or process) per Julia model.
"""

from pyUDE.julia.node import JuliaNODE
from pyUDE.julia.custom_derivatives import JuliaCustomDerivatives
from pyUDE.julia.custom_differences import JuliaCustomDifferences

__all__ = ["JuliaNODE", "JuliaCustomDerivatives", "JuliaCustomDifferences"]
