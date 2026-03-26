# Installation

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- torchdiffeq >= 0.2 *(required only for `loss="simulation"`)*

`loss="derivative_matching"` does **not** require torchdiffeq. scipy is optional but
recommended — without it, derivative estimation falls back to finite differences:

```bash
pip install scipy   # recommended for derivative_matching
```

## Standard Install

```bash
pip install pyUDE
```

This installs the PyTorch backend (NODE, CustomDerivatives, CustomDifferences) and all core
dependencies.

## Optional Extras

### Notebook support

Adds Jupyter and scipy for running the example notebooks, for improved derivative estimation
with `loss="derivative_matching"`, and for using `get_right_hand_side()` with
`scipy.integrate.odeint`:

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

## GPU Support

pyUDE delegates GPU acceleration to PyTorch. No additional pyUDE-specific install step is
required — just ensure your PyTorch installation has the right backend.

### NVIDIA (CUDA)

> **Important:** `pip install pyUDE` installs the CPU-only build of PyTorch by default.
> If you have an NVIDIA GPU you must reinstall PyTorch with CUDA support. pyUDE itself
> needs no change — only the torch wheel matters.

**Step 1 — Check your NVIDIA driver CUDA version:**

```bash
nvidia-smi   # top-right corner shows "CUDA Version: X.Y"
```

**Step 2 — Uninstall the CPU-only torch, then reinstall with CUDA:**

For CUDA 12.x drivers (most common — also works with CUDA 13.x due to forward compatibility):

```bash
pip uninstall torch torchdiffeq -y
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install torchdiffeq
```

For CUDA 11.8 drivers:

```bash
pip uninstall torch torchdiffeq -y
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torchdiffeq
```

> **Tip:** Check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
> for the exact command matching your driver version and Python environment.

**Step 3 — Verify:**

```python
import torch
print(torch.__version__)              # should show e.g. 2.x.x+cu126  (NOT +cpu)
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # e.g. "NVIDIA GeForce RTX 4080"

model = ude.NODE(data, device="cuda")
```

**CUDA version compatibility:** NVIDIA drivers are forward-compatible — a system with CUDA
13.x can run PyTorch built for CUDA 12.x. Install the latest `cu12x` wheel unless you have
a specific reason to match exactly.

### Apple Silicon (MPS)

PyTorch >= 2.0 includes MPS support out of the box on macOS with Apple Silicon:

```python
import torch
print(torch.backends.mps.is_available())   # True on M-series Macs

model = ude.NODE(data, device="mps")
```

### Auto-select device

```python
import torch

def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

model = ude.NODE(data, device=best_device())
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
# Fast unit tests (no torchdiffeq required — derivative_matching works here)
pytest tests/unit/ -v

# Integration tests (require torchdiffeq for simulation loss)
pytest tests/integration/test_node.py tests/integration/test_custom_derivatives.py -v

# Julia backend tests (require Julia + UniversalDiffEq.jl)
pytest tests/integration/test_julia_backend.py -v
```
