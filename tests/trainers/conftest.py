"""
Fixtures for trainer tests
"""

import pytest
import torch

from virtual_stain_flow.trainers.AbstractTrainer import AbstractTrainer
from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer


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
