"""
Tests for SingleGeneratorTrainer
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


class TestSingleGeneratorTrainerTrainStep:
    """Tests for SingleGeneratorTrainer.train_step method."""

    def test_train_step_returns_dict(self, single_generator_trainer):
        """Test that train_step returns a dictionary."""
        inputs = torch.randn(2, 4)
        targets = torch.randn(2, 2)
        
        result = single_generator_trainer.train_step(inputs, targets)
        
        assert isinstance(result, dict)

    def test_train_step_returns_loss_values(self, single_generator_trainer):
        """Test that train_step returns loss values in the dictionary."""
        inputs = torch.randn(2, 4)
        targets = torch.randn(2, 2)
        
        result = single_generator_trainer.train_step(inputs, targets)
        
        assert len(result) > 0
        for key, value in result.items():
            assert isinstance(value, (float, int, torch.Tensor))

    def test_train_step_updates_model_parameters(self, single_generator_trainer):
        """Test that train_step updates model parameters."""
        inputs = torch.randn(2, 4)
        targets = torch.randn(2, 2)
        
        # Store initial parameters
        initial_params = [p.clone() for p in single_generator_trainer.model.parameters()]
        
        # Perform train step
        single_generator_trainer.train_step(inputs, targets)
        
        # Check that at least one parameter changed
        params_changed = False
        for initial_param, current_param in zip(initial_params, single_generator_trainer.model.parameters()):
            if not torch.equal(initial_param, current_param):
                params_changed = True
                break
        
        assert params_changed, "Model parameters should be updated during train_step"

    def test_train_step_with_multiple_losses(self, multi_loss_trainer):
        """Test that train_step works with multiple loss functions."""
        inputs = torch.randn(2, 4)
        targets = torch.randn(2, 2)
        
        result = multi_loss_trainer.train_step(inputs, targets)
        
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_train_step_accepts_different_batch_sizes(self, single_generator_trainer):
        """Test that train_step handles different batch sizes."""
        for batch_size in [1, 2, 4]:
            inputs = torch.randn(batch_size, 4)
            targets = torch.randn(batch_size, 2)
            
            result = single_generator_trainer.train_step(inputs, targets)
            
            assert isinstance(result, dict)


class TestSingleGeneratorTrainerEvaluateStep:
    """Tests for SingleGeneratorTrainer.evaluate_step method."""

    def test_evaluate_step_returns_dict(self, single_generator_trainer):
        """Test that evaluate_step returns a dictionary."""
        inputs = torch.randn(2, 4)
        targets = torch.randn(2, 2)
        
        result = single_generator_trainer.evaluate_step(inputs, targets)
        
        assert isinstance(result, dict)

    def test_evaluate_step_returns_loss_values(self, single_generator_trainer):
        """Test that evaluate_step returns loss values in the dictionary."""
        inputs = torch.randn(2, 4)
        targets = torch.randn(2, 2)
        
        result = single_generator_trainer.evaluate_step(inputs, targets)
        
        assert len(result) > 0
        for key, value in result.items():
            assert isinstance(value, (float, int, torch.Tensor))

    def test_evaluate_step_does_not_update_model(self, single_generator_trainer):
        """Test that evaluate_step does not update model parameters."""
        inputs = torch.randn(2, 4)
        targets = torch.randn(2, 2)
        
        # Store initial parameters
        initial_params = [p.clone() for p in single_generator_trainer.model.parameters()]
        
        # Perform evaluate step
        single_generator_trainer.evaluate_step(inputs, targets)
        
        # Check that parameters did not change
        for initial_param, current_param in zip(initial_params, single_generator_trainer.model.parameters()):
            assert torch.equal(initial_param, current_param), \
                "Model parameters should not be updated during evaluate_step"

    def test_evaluate_step_with_multiple_losses(self, multi_loss_trainer):
        """Test that evaluate_step works with multiple loss functions."""
        inputs = torch.randn(2, 4)
        targets = torch.randn(2, 2)
        
        result = multi_loss_trainer.evaluate_step(inputs, targets)
        
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_evaluate_step_accepts_different_batch_sizes(self, single_generator_trainer):
        """Test that evaluate_step handles different batch sizes."""
        for batch_size in [1, 2, 4]:
            inputs = torch.randn(batch_size, 4)
            targets = torch.randn(batch_size, 2)
            
            result = single_generator_trainer.evaluate_step(inputs, targets)
            
            assert isinstance(result, dict)

    def test_evaluate_step_consistent_results(self, single_generator_trainer):
        """Test that evaluate_step returns consistent results for same inputs."""
        inputs = torch.randn(2, 4)
        targets = torch.randn(2, 2)
        
        # Run evaluate_step twice with same inputs
        result1 = single_generator_trainer.evaluate_step(inputs, targets)
        result2 = single_generator_trainer.evaluate_step(inputs, targets)
        
        # Results should be identical (no randomness in eval mode)
        assert result1.keys() == result2.keys()
        for key in result1.keys():
            val1 = result1[key]
            val2 = result2[key]
            if isinstance(val1, torch.Tensor):
                assert torch.equal(val1, val2)
            else:
                assert val1 == val2


class TestSingleGeneratorTrainerTrain:
    """Tests for SingleGeneratorTrainer.train() method."""

    def test_train_completes_successfully(self, conv_trainer, dummy_logger):
        """Test that train function completes without errors."""
        conv_trainer.train(logger=dummy_logger, epochs=2, verbose=False)
        
        assert conv_trainer.epoch == 2

    def test_train_calls_logger_methods(self, conv_trainer, dummy_logger):
        """Test that train function calls all logger lifecycle methods."""
        conv_trainer.train(logger=dummy_logger, epochs=3, verbose=False)
        
        assert dummy_logger.bind_trainer_called
        assert dummy_logger.on_train_start_called
        assert len(dummy_logger.on_epoch_start_calls) == 3
        assert len(dummy_logger.on_epoch_end_calls) == 3
        assert dummy_logger.on_train_end_called

    def test_train_logs_train_and_val_losses(self, conv_trainer, dummy_logger):
        """Test that train function logs both train and val losses."""
        conv_trainer.train(logger=dummy_logger, epochs=2, verbose=False)
        
        # Check that metrics were logged
        assert len(dummy_logger.logged_metrics) > 0
        
        # Check for train and val loss metrics
        metric_names = [m['name'] for m in dummy_logger.logged_metrics]
        assert any('train_' in name for name in metric_names)
        assert any('val_' in name for name in metric_names)

    def test_train_updates_epoch_counter(self, conv_trainer, dummy_logger):
        """Test that train function correctly updates epoch counter."""
        initial_epoch = conv_trainer.epoch
        conv_trainer.train(logger=dummy_logger, epochs=5, verbose=False)
        
        assert conv_trainer.epoch == initial_epoch + 5

    def test_train_accumulates_losses(self, conv_trainer, dummy_logger):
        """Test that train function accumulates losses over epochs."""
        epochs = 3
        conv_trainer.train(logger=dummy_logger, epochs=epochs, verbose=False)
        
        # Check that losses were accumulated
        assert len(conv_trainer.train_losses) > 0
        assert len(conv_trainer.val_losses) > 0
        
        # Each loss type should have one entry per epoch
        for loss_name, loss_values in conv_trainer.train_losses.items():
            assert len(loss_values) == epochs
        
        for loss_name, loss_values in conv_trainer.val_losses.items():
            assert len(loss_values) == epochs

    def test_train_with_patience_early_stopping(self, conv_trainer, dummy_logger):
        """Test that train function respects patience for early stopping."""
        # Train with very low patience - should stop before all epochs
        conv_trainer.train(logger=dummy_logger, epochs=100, patience=2, verbose=False)
        
        # Should stop early (before 100 epochs)
        assert conv_trainer.epoch < 100

    def test_train_without_early_stopping(self, conv_trainer, dummy_logger):
        """Test that train function runs all epochs without early stopping."""
        epochs = 5
        # No patience parameter means early stopping is disabled
        conv_trainer.train(logger=dummy_logger, epochs=epochs, verbose=False)
        
        assert conv_trainer.epoch == epochs

    def test_train_updates_best_model(self, conv_trainer, dummy_logger):
        """Test that train function updates best_model during training."""
        conv_trainer.train(logger=dummy_logger, epochs=3, verbose=False)
        
        # best_model should be set
        assert conv_trainer.best_model is not None

    def test_train_with_single_epoch(self, conv_trainer, dummy_logger):
        """Test that train function works with just one epoch."""
        conv_trainer.train(logger=dummy_logger, epochs=1, verbose=False)
        
        assert conv_trainer.epoch == 1
        assert len(dummy_logger.on_epoch_start_calls) == 1
        assert len(dummy_logger.on_epoch_end_calls) == 1

    def test_train_logs_correct_number_of_metrics(self, conv_trainer, dummy_logger):
        """Test that train function logs the expected number of metrics."""
        epochs = 3
        conv_trainer.train(logger=dummy_logger, epochs=epochs, verbose=False)
        
        # Should log metrics for each epoch (train + val for each loss)
        # At minimum: train_loss and val_loss per epoch
        assert len(dummy_logger.logged_metrics) >= epochs * 2

    def test_train_with_verbose_false(self, conv_trainer, dummy_logger):
        """Test that train function works with verbose=False."""
        # Should not raise any errors
        conv_trainer.train(logger=dummy_logger, epochs=2, verbose=False)
        
        assert conv_trainer.epoch == 2

    def test_train_requires_mlflow_logger(self, conv_trainer):
        """Test that train function requires an MlflowLogger instance."""
        # DummyLogger is not an MlflowLogger, so this should raise TypeError
        class NotALogger:
            pass
        
        with pytest.raises(TypeError):
            conv_trainer.train(logger=NotALogger(), epochs=1, verbose=False)

    def test_train_epoch_counter_increments_correctly(self, conv_trainer, dummy_logger):
        """Test that epoch counter increments by 1 for each epoch."""
        epochs_to_train = 4
        
        for expected_epoch in range(1, epochs_to_train + 1):
            conv_trainer.train(logger=dummy_logger, epochs=1, verbose=False)
            assert conv_trainer.epoch == expected_epoch

    def test_train_logged_metrics_have_correct_steps(self, conv_trainer, dummy_logger):
        """Test that logged metrics have correct step numbers."""
        epochs = 3
        conv_trainer.train(logger=dummy_logger, epochs=epochs, verbose=False)
        
        # Check that step numbers are within expected range (0-indexed)
        for metric in dummy_logger.logged_metrics:
            assert 0 <= metric['step'] < epochs
