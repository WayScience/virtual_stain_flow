"""
Tests for SingleGeneratorTrainer initialization and save_model functionality.
"""

import pathlib
import tempfile

import pytest
import torch


class TestSingleGeneratorTrainerInitialization:
    """Tests for SingleGeneratorTrainer initialization."""

    def test_init_with_single_loss_no_weights(
        self, mock_model_with_save, mock_optimizer, simple_loss, 
        train_dataloader, val_dataloader
    ):
        """Test initialization with single loss and no weights."""
        from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
        
        trainer = SingleGeneratorTrainer(
            model=mock_model_with_save,
            optimizer=mock_optimizer,
            losses=simple_loss,
            device=torch.device('cpu'),
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2
        )
        
        # Check that loss_group has 1 item with weight 1.0
        assert len(trainer._loss_group.items) == 1
        assert trainer._loss_group.items[0].weight == 1.0

    def test_init_with_single_loss_scalar_weight(
        self, mock_model_with_save, mock_optimizer, simple_loss, 
        train_dataloader, val_dataloader
    ):
        """Test initialization with single loss and scalar weight."""
        from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
        
        trainer = SingleGeneratorTrainer(
            model=mock_model_with_save,
            optimizer=mock_optimizer,
            losses=simple_loss,
            device=torch.device('cpu'),
            loss_weights=0.5,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2
        )
        
        # Check that loss_group has 1 item with weight 0.5
        assert len(trainer._loss_group.items) == 1
        assert trainer._loss_group.items[0].weight == 0.5

    def test_init_with_multiple_losses_no_weights(
        self, mock_model_with_save, mock_optimizer, multiple_losses, 
        train_dataloader, val_dataloader
    ):
        """Test initialization with multiple losses and no weights."""
        from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
        
        trainer = SingleGeneratorTrainer(
            model=mock_model_with_save,
            optimizer=mock_optimizer,
            losses=multiple_losses,
            device=torch.device('cpu'),
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2
        )
        
        # Check that loss_group has 2 items with default weight 1.0
        assert len(trainer._loss_group.items) == 2
        assert trainer._loss_group.items[0].weight == 1.0
        assert trainer._loss_group.items[1].weight == 1.0

    def test_init_with_multiple_losses_list_weights(
        self, mock_model_with_save, mock_optimizer, multiple_losses, 
        train_dataloader, val_dataloader
    ):
        """Test initialization with multiple losses and list of weights."""
        from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
        
        trainer = SingleGeneratorTrainer(
            model=mock_model_with_save,
            optimizer=mock_optimizer,
            losses=multiple_losses,
            device=torch.device('cpu'),
            loss_weights=[0.7, 0.3],
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2
        )
        
        # Check that loss_group has 2 items with specified weights
        assert len(trainer._loss_group.items) == 2
        assert trainer._loss_group.items[0].weight == 0.7
        assert trainer._loss_group.items[1].weight == 0.3

    def test_init_with_multiple_losses_scalar_weight_expansion(
        self, mock_model_with_save, mock_optimizer, multiple_losses, 
        train_dataloader, val_dataloader
    ):
        """Test that scalar weight is expanded to list for multiple losses."""
        from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
        
        trainer = SingleGeneratorTrainer(
            model=mock_model_with_save,
            optimizer=mock_optimizer,
            losses=multiple_losses,
            device=torch.device('cpu'),
            loss_weights=0.8,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2
        )
        
        # Check that scalar weight is expanded to all losses
        assert len(trainer._loss_group.items) == 2
        assert trainer._loss_group.items[0].weight == 0.8
        assert trainer._loss_group.items[1].weight == 0.8

    def test_init_with_mismatched_loss_weights_length_raises_error(
        self, mock_model_with_save, mock_optimizer, multiple_losses, 
        train_dataloader, val_dataloader
    ):
        """Test that mismatched loss_weights length raises ValueError."""
        from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
        
        with pytest.raises(ValueError, match="Length of loss_weights must match"):
            SingleGeneratorTrainer(
                model=mock_model_with_save,
                optimizer=mock_optimizer,
                losses=multiple_losses,  # 2 losses
                device=torch.device('cpu'),
                loss_weights=[0.5, 0.3, 0.2],  # 3 weights - mismatch!
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                batch_size=2
            )

    def test_init_loss_items_have_correct_args(
        self, mock_model_with_save, mock_optimizer, simple_loss, 
        train_dataloader, val_dataloader
    ):
        """Test that loss items have correct args for forward pass."""
        from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
        
        trainer = SingleGeneratorTrainer(
            model=mock_model_with_save,
            optimizer=mock_optimizer,
            losses=simple_loss,
            device=torch.device('cpu'),
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2
        )
        
        # Check that loss item has correct args
        assert trainer._loss_group.items[0].args == ('preds', 'targets')

    def test_init_loss_items_on_correct_device(
        self, mock_model_with_save, mock_optimizer, simple_loss, 
        train_dataloader, val_dataloader
    ):
        """Test that loss items are moved to correct device."""
        from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
        
        trainer = SingleGeneratorTrainer(
            model=mock_model_with_save,
            optimizer=mock_optimizer,
            losses=simple_loss,
            device=torch.device('cpu'),
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2
        )
        
        # Check that loss item has correct device
        assert trainer._loss_group.items[0].device == torch.device('cpu')


class TestSingleGeneratorTrainerSaveModel:
    """Tests for SingleGeneratorTrainer.save_model method."""

    def test_save_model_creates_file(
        self, mock_model_with_save, mock_optimizer, simple_loss, 
        train_dataloader, val_dataloader
    ):
        """Test that save_model creates a file."""
        from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
        
        trainer = SingleGeneratorTrainer(
            model=mock_model_with_save,
            optimizer=mock_optimizer,
            losses=simple_loss,
            device=torch.device('cpu'),
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            
            paths = trainer.save_model(save_path=tmpdir_path, best_model=True)
            
            assert paths is not None
            assert len(paths) == 1
            assert paths[0].exists()

    def test_save_model_returns_list_of_paths(
        self, mock_model_with_save, mock_optimizer, simple_loss, 
        train_dataloader, val_dataloader
    ):
        """Test that save_model returns a list of paths."""
        from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
        
        trainer = SingleGeneratorTrainer(
            model=mock_model_with_save,
            optimizer=mock_optimizer,
            losses=simple_loss,
            device=torch.device('cpu'),
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            
            paths = trainer.save_model(save_path=tmpdir_path)
            
            assert isinstance(paths, list)
            assert all(isinstance(p, pathlib.Path) for p in paths)
