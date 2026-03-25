"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


def _lotka_volterra(t_span, u0, alpha=1.0, beta=0.1, delta=0.075, gamma=1.5, noise=0.0):
    """Generate synthetic Lotka-Volterra data using Euler integration."""
    dt = t_span[1] - t_span[0]
    prey, pred = u0
    rows = []
    for t in t_span:
        rows.append((t, prey, pred))
        dprey = alpha * prey - beta * prey * pred
        dpred = delta * prey * pred - gamma * pred
        prey = max(prey + dt * dprey + noise * np.random.randn(), 1e-6)
        pred = max(pred + dt * dpred + noise * np.random.randn(), 1e-6)
    return pd.DataFrame(rows, columns=["time", "prey", "predator"])


@pytest.fixture
def lv_data():
    """Clean Lotka-Volterra time series for integration tests."""
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    return _lotka_volterra(t, u0=(10.0, 5.0))


@pytest.fixture
def lv_data_noisy():
    """Noisy Lotka-Volterra time series."""
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    return _lotka_volterra(t, u0=(10.0, 5.0), noise=0.05)


@pytest.fixture
def simple_data():
    """Single-state exponential decay data."""
    t = np.linspace(0, 5, 50)
    u = np.exp(-0.5 * t)
    return pd.DataFrame({"time": t, "x": u})
