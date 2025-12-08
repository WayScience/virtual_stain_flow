"""
conftest.py - Fixtures for trainer tests
"""
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from virtual_stain_flow.trainers.AbstractTrainer import AbstractTrainer


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
