"""Shared fixtures for engine tests."""

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def simple_conv_model():
    """
    Simple single-layer Conv2d model that preserves dimensions.
    Input: (B, 3, H, W) -> Output: (B, 3, H, W)
    """
    return nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=1,  # preserves spatial dimensions
        bias=True
    )


@pytest.fixture
def random_input():
    """Random input tensor (batch=2, channels=3, height=8, width=8)."""
    return torch.randn(2, 3, 8, 8)


@pytest.fixture
def random_target():
    """Random target tensor (batch=2, channels=3, height=8, width=8)."""
    return torch.randn(2, 3, 8, 8)


@pytest.fixture
def available_devices():
    """List of available devices for testing."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f"cuda:{i}"))
    return devices


@pytest.fixture
def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def multi_output_model():
    """
    Model that returns multiple outputs.
    Input: (B, 3, H, W) -> Output: ((B, 3, H, W), (B, 3, H, W))
    """
    class MultiOutputConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                padding=1,
                bias=True
            )
        
        def forward(self, x):
            out = self.conv(x)
            return (out, out)  # Return tuple of 2 outputs
    
    return MultiOutputConv()
