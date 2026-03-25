# Installation

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- torchdiffeq >= 0.2

## Standard Install

```bash
pip install pyUDE
```

This installs the PyTorch backend (NODE, CustomDerivatives, CustomDifferences) and all core
dependencies.

## Optional Extras

### Notebook support

Adds Jupyter and scipy for running the example notebooks and using
`get_right_hand_side()` with `scipy.integrate.odeint`:

```bash
pip install "pyUDE[notebook]"
```

### Julia backend

Adds `juliacall` for the Julia backend (JuliaNODE, JuliaCustomDerivatives,
JuliaCustomDifferences). Julia itself must be installed separately — see
[Julia Backend Setup](guides/julia-setup.md).

```bash
pip install "pyUDE[julia]"
```

### Development

```bash
pip install "pyUDE[dev]"
```

Installs `pytest`, `pytest-cov`, and `ruff`.

### Everything

```bash
pip install "pyUDE[julia,notebook,dev]"
```

## Development Install (editable)

To work on pyUDE itself:

```bash
git clone https://github.com/19fitzna/pyUDE
cd pyUDE
pip install -e ".[dev,notebook]"
```

## Verifying the Install

```python
import pyUDE
print(pyUDE.__version__)   # 0.0.2

# Check Julia backend is available (requires juliacall)
print(hasattr(pyUDE, "JuliaNODE"))
```

## Running Tests

```bash
# Fast unit tests (no torchdiffeq required)
pytest tests/unit/ -v

# Integration tests (require torchdiffeq)
pytest tests/integration/test_node.py tests/integration/test_custom_derivatives.py -v

# Julia backend tests (require Julia + UniversalDiffEq.jl)
pytest tests/integration/test_julia_backend.py -v
```
