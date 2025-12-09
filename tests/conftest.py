"""
Testing fixtures meant to be shared across the whole package
"""

import pathlib

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from virtual_stain_flow.vsf_logging import MlflowLogger


# ----- Mock virtual_stain_flow components ----- #

class DummyLogger(MlflowLogger):
    """
    Dummy logger fixture that tracks method calls for testing.
    Mimics the MlflowLogger interface without actually logging to MLflow.
    
    Inherits from MlflowLogger to pass type checks. 
    """
    
    def __init__(self):

        # bypassing the superclass init here as we don't need
        # actual logging behavior but just the interface and the ability
        # to record life cycle method calls for testing.

        self.trainer = None
        self.bind_trainer_called = False
        self.on_train_start_called = False
        self.on_epoch_start_calls = []
        self.on_epoch_end_calls = []
        self.on_train_end_called = False
        self.logged_metrics = []
    
    def bind_trainer(self, trainer):
        """Bind trainer to logger."""
        self.trainer = trainer
        self.bind_trainer_called = True
    
    def on_train_start(self):
        """Called at the start of training."""
        self.on_train_start_called = True
    
    def on_epoch_start(self):
        """Called at the start of each epoch."""
        self.on_epoch_start_calls.append(True)
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        self.on_epoch_end_calls.append(True)
    
    def on_train_end(self):
        """Called at the end of training."""
        self.on_train_end_called = True
    
    def log_metric(self, metric_name: str, metric_value, step: int):
        """Log a metric."""
        self.logged_metrics.append({
            'name': metric_name,
            'value': metric_value,
            'step': step
        })

    def end_run(self, *args, **kwargs):
        """No-op fixture for Logger cleanup."""
        pass


@pytest.fixture
def dummy_logger():
    """Create a dummy logger for testing."""
    return DummyLogger()


class MockModelWithSaveWeights(torch.nn.Module):
    """
    Mock model that implements save_weights method for testing.
    Mimics the BaseModel interface for save_weights.
    """
    
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)
    
    def save_weights(self, filename: str, dir) -> pathlib.Path:
        """Save model weights to file."""
        if isinstance(dir, str):
            dir = pathlib.Path(dir)
        
        if not dir.exists():
            raise FileNotFoundError(f"Path {dir} does not exist.")
        if not dir.is_dir():
            raise NotADirectoryError(f"Path {dir} is not a directory.")
        
        weight_file = dir / filename
        torch.save(self.state_dict(), weight_file)
        return weight_file


@pytest.fixture
def mock_model_with_save():
    """Create a mock model with save_weights method."""
    return MockModelWithSaveWeights()


@pytest.fixture
def mock_optimizer(mock_model_with_save):
    """Create an optimizer for the mock model."""
    return torch.optim.Adam(mock_model_with_save.parameters(), lr=0.001)


# ----- Fixtures for simulating minimal training ----- #

class MinimalDataset(Dataset):
    """Minimal torch.utils.data.Dataset to test training."""
    
    def __init__(self, num_samples: int = 10, input_size: int = 4, target_size: int = 2):

        self.num_samples = num_samples
        self.input_size = input_size
        self.target_size = target_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (
            torch.randn(self.input_size),
            torch.randn(self.target_size)
        )


@pytest.fixture
def small_minimal_dataset():
    """Create a minimal dataset with 5 samples."""
    return MinimalDataset(num_samples=2, input_size=4, target_size=2)


@pytest.fixture
def big_minimal_dataset():
    """Create a minimal dataset with 50 samples."""
    return MinimalDataset(num_samples=100, input_size=4, target_size=2)


@pytest.fixture
def image_dataset():
    """Create a minimal image dataset for testing."""
    class ImageDataset(Dataset):
        def __init__(self, num_samples=20):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Return 1-channel 16x16 images
            return (
                torch.randn(1, 16, 16),
                torch.randn(1, 16, 16)
            )
    
    return ImageDataset(num_samples=20)


@pytest.fixture
def minimal_model():
    """Create a minimal PyTorch model."""
    model = torch.nn.Linear(4, 2)
    return model


@pytest.fixture
def minimal_optimizer(minimal_model):
    """Create a minimal optimizer."""
    return torch.optim.SGD(minimal_model.parameters(), lr=0.01)


@pytest.fixture
def conv_model():
    """
    Create a simple convolutional network with same input/output size
        to simulate image-to-image translation tasks.
    """
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, kernel_size=3, padding=1)
    )
    return model


@pytest.fixture
def conv_optimizer(conv_model):
    """Create an optimizer for the conv model."""
    return torch.optim.Adam(conv_model.parameters(), lr=0.001)


@pytest.fixture
def train_dataloader():
    """Create a train dataloader with 5 batches of 2 samples each."""
    dataset = MinimalDataset(num_samples=10, input_size=4, target_size=2)
    return DataLoader(dataset, batch_size=2, shuffle=False)


@pytest.fixture
def val_dataloader():
    """Create a validation dataloader with 3 batches of 2 samples each."""
    dataset = MinimalDataset(num_samples=6, input_size=4, target_size=2)
    return DataLoader(dataset, batch_size=2, shuffle=False)


@pytest.fixture
def empty_dataloader():
    """Create an empty dataloader."""
    dataset = MinimalDataset(num_samples=0)
    return DataLoader(dataset, batch_size=2, shuffle=False)



@pytest.fixture
def image_train_loader(image_dataset):
    """Create a train dataloader with image data."""
    from torch.utils.data import random_split
    train_size = 12
    val_size = len(image_dataset) - train_size
    train_dataset, _ = random_split(image_dataset, [train_size, val_size])
    return DataLoader(train_dataset, batch_size=4, shuffle=False)


@pytest.fixture
def image_val_loader(image_dataset):
    """Create a validation dataloader with image data."""
    from torch.utils.data import random_split
    train_size = 12
    val_size = len(image_dataset) - train_size
    _, val_dataset = random_split(image_dataset, [train_size, val_size])
    return DataLoader(val_dataset, batch_size=4, shuffle=False)


@pytest.fixture
def simple_loss():
    """Create a simple MSE loss function."""
    return torch.nn.MSELoss()


@pytest.fixture
def multiple_losses():
    """Create multiple loss functions."""
    return [torch.nn.MSELoss(), torch.nn.L1Loss()]
