"""
pyude_bridge.jl

Julia-side glue loaded once at startup via juliacall.
Provides helpers for building Lux networks, wrapping Python callables as
Julia ODE right-hand-side functions, and converting data across the bridge.
"""

using UniversalDiffEq
using Lux
using OrdinaryDiffEq
using Optimisers
using PythonCall
using Random


# ---------------------------------------------------------------------------
# Neural network construction
# ---------------------------------------------------------------------------

"""
    make_lux_mlp(in_dim, out_dim, hidden_units, hidden_layers)

Build a Lux.jl MLP with tanh activations, matching the default architecture
used by the Python (PyTorch) backend.
"""
function make_lux_mlp(in_dim::Int, out_dim::Int, hidden_units::Int, hidden_layers::Int)
    layers = Any[Dense(in_dim => hidden_units, tanh)]
    for _ in 1:(hidden_layers - 1)
        push!(layers, Dense(hidden_units => hidden_units, tanh))
    end
    push!(layers, Dense(hidden_units => out_dim))
    return Chain(layers...)
end


# ---------------------------------------------------------------------------
# Python callable → Julia ODE function bridge
# ---------------------------------------------------------------------------

"""
    wrap_python_dynamics(py_fn)

Return a Julia function `f(u, p, t) -> du` that delegates to a Python
callable via PythonCall.jl.

Called by Julia's ODE solver millions of times per training run.  The
returned function converts Julia arrays → Python lists, calls the Python
function, then converts the result back.

Note: `p` here is the neural-network parameter component passed by
UniversalDiffEq — the Python `known_dynamics` signature receives a Python
dict of mechanistic parameters (not NN weights).  UniversalDiffEq manages
the NN weights separately via Lux.
"""
function wrap_python_dynamics(py_fn)
    function julia_dynamics(u::AbstractVector, p, t::Real)
        # Convert Julia state vector → Python list
        py_u = pylist(Float64.(u))
        # p is the NamedTuple of mechanistic params from UniversalDiffEq;
        # expose as a Python dict so the user's Python function can read it
        py_p = pydict(Dict(string(k) => Float64(v) for (k, v) in pairs(p)))
        py_t = pyfloat(Float64(t))
        result = py_fn(py_u, py_p, py_t)
        return pyconvert(Vector{Float64}, result)
    end
    return julia_dynamics
end


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

"""
    make_optimizer(name, lr)

Map a string optimizer name to an Optimisers.jl rule.
"""
function make_optimizer(name::String, lr::Float64)
    name = lowercase(name)
    if name == "adam"
        return Adam(lr)
    elseif name == "sgd"
        return Descent(lr)
    else
        error("Unknown optimizer '$name'. Choose 'adam' or 'sgd'.")
    end
end


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------

"""
    py_matrix_to_julia(py_arr)

Convert a Python 2-D array (numpy or list-of-lists) to a Julia Matrix{Float64}.
Rows = time steps, columns = state variables.
"""
function py_matrix_to_julia(py_arr)
    return pyconvert(Matrix{Float64}, py_arr)
end

"""
    py_vector_to_julia(py_arr)

Convert a Python 1-D array (numpy or list) to a Julia Vector{Float64}.
"""
function py_vector_to_julia(py_arr)
    return pyconvert(Vector{Float64}, py_arr)
end
