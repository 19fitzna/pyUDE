"""
Backwards-compatibility shim.

All public classes live in the top-level ``pyUDE`` namespace.
Import from there directly::

    from pyUDE import NODE, CustomDerivatives, CustomDifferences
"""

import warnings as _warnings


def _deprecated(name: str):
    _warnings.warn(
        f"Importing '{name}' from 'pyUDE.pyUDE' is deprecated. "
        f"Use 'from pyUDE import {name}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )


from pyUDE.core.node import NODE as _NODE
from pyUDE.core.custom_derivatives import CustomDerivatives as _CustomDerivatives
from pyUDE.core.custom_differences import CustomDifferences as _CustomDifferences


class NODE(_NODE):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __class_getitem__(cls, item):
        _deprecated("NODE")
        return _NODE


def __getattr__(name: str):
    _map = {
        "NODE": _NODE,
        "CustomDerivatives": _CustomDerivatives,
        "CustomDifferences": _CustomDifferences,
    }
    if name in _map:
        _deprecated(name)
        return _map[name]
    raise AttributeError(f"module 'pyUDE.pyUDE' has no attribute {name!r}")
