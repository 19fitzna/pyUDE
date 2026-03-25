"""
Julia runtime initialisation — lazy singleton.

Julia starts up only when a Julia model is first constructed or trained.
Subsequent calls return the already-running runtime instantly.
"""

import pathlib
import threading

_lock = threading.Lock()
_jl = None       # juliacall.Main namespace
_UDE = None      # UniversalDiffEq Julia module handle


def get_julia():
    """
    Return (jl, UDE) where:
    - ``jl``  is the juliacall.Main namespace (call Julia functions via jl.fname(...))
    - ``UDE`` is the UniversalDiffEq Julia module handle

    Raises
    ------
    ImportError
        If ``juliacall`` is not installed.
    RuntimeError
        If the Julia environment cannot be activated or packages loaded.
    """
    global _jl, _UDE

    if _jl is not None:
        return _jl, _UDE

    with _lock:
        # Double-checked locking: another thread may have initialised while we waited
        if _jl is not None:
            return _jl, _UDE

        try:
            import juliacall
        except ImportError as exc:
            raise ImportError(
                "juliacall is required for the Julia backend.\n"
                "Install it with:  pip install 'pyUDE[julia]'"
            ) from exc

        jl = juliacall.Main

        # Locate the bundled julia/ environment relative to this file:
        #   src/pyUDE/julia/_env.py  →  ../../..  →  repo root  →  julia/
        repo_root = pathlib.Path(__file__).parent.parent.parent.parent
        julia_env = repo_root / "julia"

        if not julia_env.exists():
            raise RuntimeError(
                f"Julia environment directory not found: {julia_env}\n"
                "Ensure the 'julia/' directory with Project.toml is present."
            )

        # Activate environment and resolve dependencies
        julia_env_str = str(julia_env).replace("\\", "/")
        jl.seval(f'import Pkg; Pkg.activate("{julia_env_str}"); Pkg.instantiate()')

        # Load required packages
        jl.seval("using UniversalDiffEq, Lux, OrdinaryDiffEq, Optimisers, PythonCall")

        # Load bridge helpers
        bridge = julia_env / "pyude_bridge.jl"
        bridge_str = str(bridge).replace("\\", "/")
        jl.seval(f'include("{bridge_str}")')

        _jl = jl
        _UDE = jl.UniversalDiffEq

    return _jl, _UDE
