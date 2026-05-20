import numpy as np
import pytest
from src.monitoring.drift import compute_psi, compute_ks


def test_psi_stable():
    ref = np.random.normal(100, 20, 1000)
    cur = np.random.normal(100, 20, 1000)
    assert compute_psi(ref, cur) < 0.1


def test_psi_shifted():
    ref = np.random.normal(100, 20, 1000)
    cur = np.random.normal(300, 20, 1000)
    assert compute_psi(ref, cur) > 0.2


def test_ks_stable():
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0, 1, 1000)
    result = compute_ks(ref, cur)
    assert "p_value" in result
    assert "drift_detected" in result


def test_ks_shifted():
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(10, 1, 1000)
    result = compute_ks(ref, cur)
    assert result["drift_detected"] == True