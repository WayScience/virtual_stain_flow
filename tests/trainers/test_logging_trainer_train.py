"""
Tests for SingleGeneratorTrainer.train() method inherited from AbstractTrainer.
"""

import pytest


class TestSingleGeneratorTrainerTrainFunction:
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
