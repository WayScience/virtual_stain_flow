"""Shared fixtures for engine tests."""

import pytest
import torch
import torch.nn as nn


# ============================================================================
# Helper functions
# ============================================================================


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


# ============================================================================
# Session-scoped fixtures (shared across all tests)
# ============================================================================


@pytest.fixture(scope="session")
def available_devices():
    """Fixture providing list of available devices."""
    return get_available_devices()


@pytest.fixture(scope="session")
def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def has_multiple_devices(available_devices):
    """Check if multiple devices (CPU + GPU or multiple GPUs) are available."""
    return len(available_devices) >= 2


# ============================================================================
# Function-scoped fixtures (fresh for each test)
# ============================================================================


@pytest.fixture
def torch_device(available_devices):
    """
    Fixture that provides the first available device for testing.
    Use this for individual tests that need to specify a device.
    """
    return available_devices[0]


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


@pytest.fixture
def simple_discriminator():
    """
    Simple discriminator model for GAN testing.
    Takes concatenated input/target stack (B, 6, H, W) -> outputs score (B, 1)
    Uses conv + global average pooling + linear layer.
    """
    class SimpleDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=6,  # stacked input + target
                out_channels=16,
                kernel_size=3,
                padding=1,
                bias=True
            )
            self.pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
            self.fc = nn.Linear(16, 1)  # Output single score
        
        def forward(self, x):
            x = self.conv(x)
            x = torch.relu(x)
            x = self.pool(x)  # (B, 16, 1, 1)
            x = x.flatten(1)  # (B, 16)
            x = self.fc(x)    # (B, 1)
            return x
    
    return SimpleDiscriminator()


@pytest.fixture
def random_stack():
    """Random stack tensor (batch=2, channels=6, height=8, width=8) for discriminator."""
    return torch.randn(2, 6, 8, 8)


@pytest.fixture
def sample_inputs():
    """Create sample inputs for loss computation."""
    return {
        "pred": torch.randn(4, 3, 32, 32),
        "target": torch.randn(4, 3, 32, 32),
        "pred2": torch.randn(4, 3, 32, 32),
    }


# ============================================================================
# Integration test fixtures (forward group + loss group setup)
# ============================================================================


@pytest.fixture
def optimizer(simple_conv_model):
    """Create an Adam optimizer for the simple conv model."""
    import torch.optim as optim
    return optim.Adam(simple_conv_model.parameters(), lr=1e-3)


@pytest.fixture
def forward_group(simple_conv_model, optimizer, torch_device):
    """Create a GeneratorForwardGroup with the simple model and optimizer."""
    from virtual_stain_flow.engine.forward_groups import GeneratorForwardGroup
    return GeneratorForwardGroup(
        generator=simple_conv_model,
        optimizer=optimizer,
        device=torch_device,
    )


@pytest.fixture
def l1_loss_group(torch_device):
    """Create a LossGroup with L1Loss for testing."""
    from virtual_stain_flow.engine.loss_group import LossItem, LossGroup
    from virtual_stain_flow.engine.names import PREDS, TARGETS
    return LossGroup(
        items=[
            LossItem(
                module=nn.L1Loss(),
                args=(PREDS, TARGETS),
                device=torch_device,
            )
        ]
    )


@pytest.fixture
def multi_loss_group(torch_device):
    """Create a LossGroup with multiple loss items (L1Loss + MSELoss) for testing."""
    from virtual_stain_flow.engine.loss_group import LossItem, LossGroup
    from virtual_stain_flow.engine.names import PREDS, TARGETS
    return LossGroup(
        items=[
            LossItem(
                module=nn.L1Loss(),
                args=(PREDS, TARGETS),
                weight=1.0,
                device=torch_device,
            ),
            LossItem(
                module=nn.MSELoss(),
                args=(PREDS, TARGETS),
                weight=0.5,
                device=torch_device,
            ),
        ]
    )


@pytest.fixture
def forward_pass_context(forward_group, random_input, random_target, torch_device):
    """
    Create a context from a forward pass in training mode.
    This fixture runs a complete forward pass through the forward group.
    """
    return forward_group(
        train=True,
        inputs=random_input.to(torch_device),
        targets=random_target.to(torch_device),
    )


@pytest.fixture
def forward_pass_context_eval(forward_group, random_input, random_target, torch_device):
    """
    Create a context from a forward pass in eval mode.
    This fixture runs a complete forward pass through the forward group in eval mode.
    """
    return forward_group(
        train=False,
        inputs=random_input.to(torch_device),
        targets=random_target.to(torch_device),
    )


@pytest.fixture
def disc_optimizer(simple_discriminator):
    """Create an Adam optimizer for the discriminator model."""
    import torch.optim as optim
    return optim.Adam(simple_discriminator.parameters(), lr=1e-3)


@pytest.fixture
def discriminator_forward_group(simple_discriminator, disc_optimizer, torch_device):
    """Create a DiscriminatorForwardGroup with the simple discriminator and optimizer."""
    from virtual_stain_flow.engine.forward_groups import DiscriminatorForwardGroup
    return DiscriminatorForwardGroup(
        discriminator=simple_discriminator,
        optimizer=disc_optimizer,
        device=torch_device,
    )


@pytest.fixture
def gan_orchestrator(forward_group, discriminator_forward_group):
    """Create a GANOrchestrator with generator and discriminator forward groups."""
    from virtual_stain_flow.engine.orchestrators import GANOrchestrator
    return GANOrchestrator(
        generator_fg=forward_group,
        discriminator_fg=discriminator_forward_group,
    )
