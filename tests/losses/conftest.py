"""Shared fixtures and utilities for loss tests."""

import pytest
import torch


def get_available_devices():
    """
    Get list of available devices for testing.
    
    Returns:
        List of torch.device objects available on the system.
    """
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f"cuda:{i}"))
    return devices


@pytest.fixture(scope="session")
def available_devices():
    """Fixture providing list of available devices."""
    return get_available_devices()


@pytest.fixture
def sample_inputs():
    """Create sample inputs for loss computation."""
    return {
        "pred": torch.randn(4, 3, 32, 32),
        "target": torch.randn(4, 3, 32, 32),
        "pred2": torch.randn(4, 3, 32, 32),
    }


@pytest.fixture(scope="session")
def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def has_multiple_devices():
    """Check if multiple devices (CPU + GPU or multiple GPUs) are available."""
    return len(get_available_devices()) >= 2


@pytest.fixture
def sample_inputs():
    """Create sample inputs for loss computation."""
    return {
        "pred": torch.randn(4, 3, 32, 32),
        "target": torch.randn(4, 3, 32, 32),
        "pred2": torch.randn(4, 3, 32, 32),
    }
