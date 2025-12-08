"""
conftest.py - Fixtures for trainer tests
"""
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from virtual_stain_flow.trainers.AbstractTrainer import AbstractTrainer
from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
from virtual_stain_flow.vsf_logging.MlflowLogger import MlflowLogger

class MinimalDataset(Dataset):
    """Minimal dataset for testing."""
    
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


class MinimalTrainerRealization(AbstractTrainer):
    """
    Minimal concrete realization of AbstractTrainer for testing.
    Tracks method calls and provides controllable step behavior.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track method calls for testing
        self.train_step_calls = []
        self.evaluate_step_calls = []
        self.on_epoch_start_called = False
        self.on_epoch_end_called = False

        # Create a dummy progress bar property that does nothing beyond
        # allowing set_postfix_str calls
        class DummyProgressBar:
            def set_postfix_str(self, *args, **kwargs):
                pass
                
        self._epoch_pbar = DummyProgressBar()
    
    def train_step(self, inputs: torch.tensor, targets: torch.tensor) -> dict:
        """
        Minimal train step that returns a dict of losses.
        Stores call information for verification.
        """
        
        self.train_step_calls.append({
            'inputs_shape': inputs.shape,
            'targets_shape': targets.shape,
        })
        
        # Return scalar tensor losses (simulating real losses)
        return {
            'loss_a': torch.tensor(0.5),
            'loss_b': torch.tensor(0.3),
        }
    
    def evaluate_step(self, inputs: torch.tensor, targets: torch.tensor) -> dict:
        """
        Minimal evaluate step that returns a dict of losses.
        Stores call information for verification.
        """
        
        self.evaluate_step_calls.append({
            'inputs_shape': inputs.shape,
            'targets_shape': targets.shape,
        })
        
        # Return scalar tensor losses (simulating real losses)
        return {
            'loss_a': torch.tensor(0.4),
            'loss_b': torch.tensor(0.2),
        }
    
    def save_model(self, save_path, file_name_prefix=None, file_name_suffix=None, 
                   file_ext='.pth', best_model=True):
        """Minimal save_model implementation."""
        return None


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
def trainer_with_loaders(minimal_model, minimal_optimizer, train_dataloader, val_dataloader):
    """
    Create a MinimalTrainerRealization with train and validation loaders.
    """
    trainer = MinimalTrainerRealization(
        model=minimal_model,
        optimizer=minimal_optimizer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        batch_size=2,
        device=torch.device('cpu')
    )
    return trainer


@pytest.fixture
def trainer_with_empty_val_loader(minimal_model, minimal_optimizer, train_dataloader, empty_dataloader):
    """
    Create a MinimalTrainerRealization with empty validation loader.
    """
    trainer = MinimalTrainerRealization(
        model=minimal_model,
        optimizer=minimal_optimizer,
        train_loader=train_dataloader,
        val_loader=empty_dataloader,
        batch_size=2,
        device=torch.device('cpu')
    )
    return trainer


@pytest.fixture
def simple_loss():
    """Create a simple MSE loss function."""
    return torch.nn.MSELoss()


@pytest.fixture
def multiple_losses():
    """Create multiple loss functions."""
    return [torch.nn.MSELoss(), torch.nn.L1Loss()]


@pytest.fixture
def single_generator_trainer(minimal_model, minimal_optimizer, simple_loss, train_dataloader, val_dataloader):
    """
    Create a SingleGeneratorTrainer with a single loss function.
    """
    trainer = SingleGeneratorTrainer(
        model=minimal_model,
        optimizer=minimal_optimizer,
        losses=simple_loss,
        device=torch.device('cpu'),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        batch_size=2
    )
    return trainer


@pytest.fixture
def multi_loss_trainer(minimal_model, minimal_optimizer, multiple_losses, train_dataloader, val_dataloader):
    """
    Create a SingleGeneratorTrainer with multiple loss functions.
    """
    trainer = SingleGeneratorTrainer(
        model=minimal_model,
        optimizer=minimal_optimizer,
        losses=multiple_losses,
        device=torch.device('cpu'),
        loss_weights=[0.5, 0.5],
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        batch_size=2
    )
    return trainer


class DummyLogger(MlflowLogger):
    """
    Dummy logger fixture that tracks method calls for testing.
    Mimics the MlflowLogger interface without actually logging to MLflow.
    
    Inherits from MlflowLogger to pass type checks. 
    """
    
    def __init__(self):
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


@pytest.fixture
def conv_model():
    """Create a simple convolutional model for testing."""
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
def conv_trainer(conv_model, conv_optimizer, simple_loss, image_train_loader, image_val_loader):
    """
    Create a SingleGeneratorTrainer with conv model for full training tests.
    """
    trainer = SingleGeneratorTrainer(
        model=conv_model,
        optimizer=conv_optimizer,
        losses=simple_loss,
        device=torch.device('cpu'),
        train_loader=image_train_loader,
        val_loader=image_val_loader,
        batch_size=4,
        early_termination_metric='MSELoss'
    )
    return trainer
