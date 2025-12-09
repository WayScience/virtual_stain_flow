"""
Contract tests for AbstractTrainer.train_epoch and evaluate_epoch methods
"""

import pytest
import torch

from conftest import MinimalTrainerRealization


class TestTrainEpochBatchIteration:
    """Test that train_epoch iterates over all batches."""
    
    def test_train_epoch_iterates_all_batches(self, trainer_with_loaders):
        """Verify train_epoch processes all batches from the dataloader."""
        trainer = trainer_with_loaders
        
        expected_batches = trainer._train_loader.__len__()
        
        _ = trainer.train_epoch()
        
        # Verify train_step was called exactly once per batch
        assert len(trainer.train_step_calls) == expected_batches, \
            f"Expected {expected_batches} train_step calls, got {len(trainer.train_step_calls)}"
    
    def test_evaluate_epoch_iterates_all_batches(self, trainer_with_loaders):
        """Verify evaluate_epoch processes all batches from the dataloader."""
        trainer = trainer_with_loaders
        
        expected_batches = trainer._val_loader.__len__()
        
        _ = trainer.evaluate_epoch()
        
        # Verify evaluate_step was called exactly once per batch
        assert len(trainer.evaluate_step_calls) == expected_batches, \
            f"Expected {expected_batches} evaluate_step calls, got {len(trainer.evaluate_step_calls)}"


class TestTrainEpochStepCalls:
    """Test that train_epoch calls train_step once per batch."""
    
    def test_train_step_called_once_per_batch(self, trainer_with_loaders):
        """Verify train_step is invoked exactly once for each batch."""
        trainer = trainer_with_loaders
        num_batches = trainer._train_loader.__len__()
        
        trainer.train_epoch()
        
        assert len(trainer.train_step_calls) == num_batches
    
    def test_train_step_receives_correct_batch_data(self, trainer_with_loaders):
        """Verify train_step receives inputs and targets from the dataloader."""
        trainer = trainer_with_loaders
        
        trainer.train_epoch()
        
        # Verify each call has recorded the data shapes
        for call in trainer.train_step_calls:
            assert 'inputs_shape' in call
            assert 'targets_shape' in call
            # inputs_shape should be (batch_size, 4) and targets_shape should be (batch_size, 2)
            assert call['inputs_shape'][1] == 4
            assert call['targets_shape'][1] == 2
    
    def test_evaluate_step_called_once_per_batch(self, trainer_with_loaders):
        """Verify evaluate_step is invoked exactly once for each batch."""
        trainer = trainer_with_loaders
        num_batches = trainer._val_loader.__len__()
            
        trainer.evaluate_epoch()
        
        assert len(trainer.evaluate_step_calls) == num_batches
    
    def test_evaluate_step_receives_correct_batch_data(self, trainer_with_loaders):
        """Verify evaluate_step receives inputs and targets from the dataloader."""
        trainer = trainer_with_loaders
        
        trainer.evaluate_epoch()
        
        # Verify each call has recorded the data shapes
        for call in trainer.evaluate_step_calls:
            assert 'inputs_shape' in call
            assert 'targets_shape' in call
            # inputs_shape should be (batch_size, 4) and targets_shape should be (batch_size, 2)
            assert call['inputs_shape'][1] == 4
            assert call['targets_shape'][1] == 2


class TestTrainEpochLossAggregation:
    """Test that train_epoch aggregates losses correctly."""
    
    def test_train_epoch_returns_dict_of_losses(self, trainer_with_loaders):
        """Verify train_epoch returns a dictionary with loss names as keys."""
        trainer = trainer_with_loaders
        
        result = trainer.train_epoch()
        
        assert isinstance(result, dict)
        assert 'loss_a' in result
        assert 'loss_b' in result
    
    def test_train_epoch_computes_mean_loss(self, trainer_with_loaders):
        """
        Verify train_epoch computes the mean of per-batch losses.
        
        The trainer returns {loss_a: 0.5, loss_b: 0.3} per batch.
        With 5 batches, the mean should be 0.5 and 0.3 respectively.
        """
        trainer = trainer_with_loaders
        
        result = trainer.train_epoch()
        
        # Each batch returns loss_a=0.5 and loss_b=0.3
        # Mean of 5 batches should still be 0.5 and 0.3
        assert torch.isclose(result['loss_a'], torch.tensor(0.5))
        assert torch.isclose(result['loss_b'], torch.tensor(0.3))
    
    def test_evaluate_epoch_returns_dict_of_losses(self, trainer_with_loaders):
        """Verify evaluate_epoch returns a dictionary with loss names as keys."""
        trainer = trainer_with_loaders
        
        result = trainer.evaluate_epoch()
        
        assert isinstance(result, dict)
        assert 'loss_a' in result
        assert 'loss_b' in result
    
    def test_evaluate_epoch_computes_mean_loss(self, trainer_with_loaders):
        """
        Verify evaluate_epoch computes the mean of per-batch losses.
        
        The trainer returns {loss_a: 0.4, loss_b: 0.2} per batch.
        With 3 batches, the mean should be 0.4 and 0.2 respectively.
        """
        trainer = trainer_with_loaders
        
        result = trainer.evaluate_epoch()
        
        # Each batch returns loss_a=0.4 and loss_b=0.2
        # Mean of 3 batches should still be 0.4 and 0.2
        assert torch.isclose(result['loss_a'], torch.tensor(0.4))
        assert torch.isclose(result['loss_b'], torch.tensor(0.2))
    
    def test_train_epoch_loss_aggregation_correctness(self, trainer_with_loaders):
        """Verify the mathematical correctness of loss aggregation."""
        trainer = trainer_with_loaders
        
        # Mock train_step to return varying losses
        call_count = [0]
        
        def varying_train_step(inputs, targets):
            call_count[0] += 1
            # Return batch_idx * 0.1 as loss_a
            return {
                'loss': torch.tensor(float(call_count[0] * 0.1)),
            }
        
        trainer.train_step = varying_train_step
        result = trainer.train_epoch()
        
        # With 5 batches: losses are 0.1, 0.2, 0.3, 0.4, 0.5
        # Mean = (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5 = 0.3
        expected_mean = (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5
        assert torch.isclose(result['loss'], torch.tensor(expected_mean))


class TestTrainEpochEpochCounter:
    """Test that train_epoch and evaluate_epoch work with epoch counter."""
    
    def test_epoch_counter_not_incremented_by_train_epoch(self, trainer_with_loaders):
        """
        Verify that train_epoch does NOT increment the epoch counter.
        The epoch counter is incremented by the train() method, not train_epoch().
        """
        trainer = trainer_with_loaders
        initial_epoch = trainer.epoch
        
        trainer.train_epoch()
        
        # train_epoch should not increment epoch
        assert trainer.epoch == initial_epoch
    
    def test_epoch_counter_not_incremented_by_evaluate_epoch(self, trainer_with_loaders):
        """
        Verify that evaluate_epoch does NOT increment the epoch counter.
        The epoch counter is incremented by the train() method, not evaluate_epoch().
        """
        trainer = trainer_with_loaders
        initial_epoch = trainer.epoch
        
        trainer.evaluate_epoch()
        
        # evaluate_epoch should not increment epoch
        assert trainer.epoch == initial_epoch



class TestTrainEpochEdgeCases:
    """Test edge cases for train_epoch and evaluate_epoch."""
    
    def test_train_epoch_with_empty_dataloader(self, minimal_model, minimal_optimizer, empty_dataloader):
        """
        Verify train_epoch handles empty dataloader gracefully.
        Should return a dict with empty lists aggregated (likely NaN or empty).
        """
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=empty_dataloader,
            val_loader=empty_dataloader,
            batch_size=2,
            device=torch.device('cpu')
        )
        
        # Should not raise an error
        result = trainer.train_epoch()
        
        # With empty dataloader, no batches are processed
        assert len(trainer.train_step_calls) == 0
        # Result should be an empty dict (no losses collected)
        assert result == {}
    
    def test_evaluate_epoch_with_empty_dataloader(self, minimal_model, minimal_optimizer, empty_dataloader):
        """
        Verify evaluate_epoch handles empty dataloader gracefully.
        """
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=empty_dataloader,
            val_loader=empty_dataloader,
            batch_size=2,
            device=torch.device('cpu')
        )
        
        # Should not raise an error
        result = trainer.evaluate_epoch()
        
        # With empty dataloader, no batches are processed
        assert len(trainer.evaluate_step_calls) == 0
        # Result should be an empty dict (no losses collected)
        assert result == {}
    
    def test_train_epoch_with_single_batch(self, minimal_model, minimal_optimizer, small_minimal_dataset):
        """
        Verify train_epoch works correctly with a single batch.
        """
        from torch.utils.data import DataLoader
        
        dataset = small_minimal_dataset
        train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=train_loader,
            val_loader=train_loader,
            batch_size=2,
            device=torch.device('cpu')
        )
        
        result = trainer.train_epoch()
        
        # Should process exactly 1 batch
        assert len(trainer.train_step_calls) == 1
        # Result should contain loss_a and loss_b
        assert 'loss_a' in result
        assert 'loss_b' in result
    
    def test_evaluate_epoch_with_single_batch(self, minimal_model, minimal_optimizer, small_minimal_dataset):
        """
        Verify evaluate_epoch works correctly with a single batch.
        """
        from torch.utils.data import DataLoader
        
        dataset = small_minimal_dataset
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=val_loader,
            val_loader=val_loader,
            batch_size=2,
            device=torch.device('cpu')
        )
        
        result = trainer.evaluate_epoch()
        
        # Should process exactly 1 batch
        assert len(trainer.evaluate_step_calls) == 1
        # Result should contain loss_a and loss_b
        assert 'loss_a' in result
        assert 'loss_b' in result
    
    def test_train_epoch_with_large_batch_count(self, minimal_model, minimal_optimizer, big_minimal_dataset):
        """
        Verify train_epoch works correctly with many batches.
        """
        from torch.utils.data import DataLoader
        
        dataset = big_minimal_dataset
        train_loader = DataLoader(dataset, batch_size=10, shuffle=False)
        
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=train_loader,
            val_loader=train_loader,
            batch_size=10,
            device=torch.device('cpu')
        )
        
        result = trainer.train_epoch()
        
        # Should process all 10 batches
        assert len(trainer.train_step_calls) == 10
        # Result should contain loss_a and loss_b
        assert 'loss_a' in result
        assert 'loss_b' in result


class TestDataSplitting:
    """Test that AbstractTrainer correctly handles dataset splitting."""
    
    def test_init_with_dataset_creates_loaders(self, minimal_model, minimal_optimizer, dataset_for_splitting):
        """Verify that providing a dataset creates train/val/test loaders."""
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            dataset=dataset_for_splitting,
            batch_size=4,
            device=torch.device('cpu')
        )
        
        # Verify loaders exist
        assert trainer._train_loader is not None
        assert trainer._val_loader is not None
        assert trainer._test_loader is not None
        
        # Verify loaders are DataLoaders
        from torch.utils.data import DataLoader
        assert isinstance(trainer._train_loader, DataLoader)
        assert isinstance(trainer._val_loader, DataLoader)
        assert isinstance(trainer._test_loader, DataLoader)
    
    def test_init_with_dataset_respects_default_split_ratios(self, minimal_model, minimal_optimizer, dataset_for_splitting):
        """Verify that default split ratios (0.7/0.15/0.15) are applied."""
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            dataset=dataset_for_splitting,
            batch_size=4,
            device=torch.device('cpu')
        )
        
        # With 100 samples and default split: 70 train, 15 val, 15 test
        train_samples = len(trainer._train_loader.dataset)
        val_samples = len(trainer._val_loader.dataset)
        test_samples = len(trainer._test_loader.dataset)
        
        assert train_samples == 70
        assert val_samples == 15
        assert test_samples == 15
    
    def test_init_with_dataset_respects_custom_split_ratios(self, minimal_model, minimal_optimizer, dataset_for_splitting):
        """Verify that custom split ratios are applied correctly."""
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            dataset=dataset_for_splitting,
            batch_size=4,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            device=torch.device('cpu')
        )
        
        # With 100 samples and custom split: 60 train, 20 val, 20 test
        train_samples = len(trainer._train_loader.dataset)
        val_samples = len(trainer._val_loader.dataset)
        test_samples = len(trainer._test_loader.dataset)
        
        assert train_samples == 60
        assert val_samples == 20
        assert test_samples == 20
    
    def test_init_with_loaders_does_not_split(self, minimal_model, minimal_optimizer, train_dataloader, val_dataloader):
        """Verify that providing loaders directly skips dataset splitting."""
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2,
            device=torch.device('cpu')
        )
        
        # Verify the exact loaders are used
        assert trainer._train_loader is train_dataloader
        assert trainer._val_loader is val_dataloader
    
    def test_init_without_dataset_or_loaders_raises_error(self, minimal_model, minimal_optimizer):
        """Verify that initialization fails without dataset or loaders."""
        with pytest.raises(ValueError, match="Either provide dataset"):
            MinimalTrainerRealization(
                model=minimal_model,
                optimizer=minimal_optimizer,
                batch_size=2,
                device=torch.device('cpu')
            )
    
    def test_init_with_only_train_loader(self, minimal_model, minimal_optimizer, train_dataloader):
        """Verify that providing only train_loader works (val/test are empty)."""
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=train_dataloader,
            batch_size=2,
            device=torch.device('cpu')
        )
        
        assert trainer._train_loader is train_dataloader
        assert trainer._val_loader == []
        assert trainer._test_loader == []



class TestMetricComputation:
    """Test that AbstractTrainer correctly handles metrics."""
    
    def test_update_loss_training(self, trainer_with_loaders):
        """Verify that update_loss correctly appends training losses."""
        trainer = trainer_with_loaders
        
        loss_value = torch.tensor(0.5)
        trainer.update_loss(loss_value, "mse_loss", validation=False)
        
        assert "mse_loss" in trainer.train_losses
        assert len(trainer.train_losses["mse_loss"]) == 1
        assert trainer.train_losses["mse_loss"][0] == loss_value
    
    def test_update_loss_validation(self, trainer_with_loaders):
        """Verify that update_loss correctly appends validation losses."""
        trainer = trainer_with_loaders
        
        loss_value = torch.tensor(0.3)
        trainer.update_loss(loss_value, "mse_loss", validation=True)
        
        assert "mse_loss" in trainer.val_losses
        assert len(trainer.val_losses["mse_loss"]) == 1
        assert trainer.val_losses["mse_loss"][0] == loss_value
    
    def test_update_loss_multiple_calls(self, trainer_with_loaders):
        """Verify that update_loss accumulates multiple loss values."""
        trainer = trainer_with_loaders
        
        for i in range(5):
            trainer.update_loss(torch.tensor(float(i)), "loss", validation=False)
        
        assert len(trainer.train_losses["loss"]) == 5
        assert trainer.train_losses["loss"][-1] == torch.tensor(4.0)
    
    def test_update_metrics_training(self, trainer_with_loaders):
        """Verify that update_metrics correctly appends training metrics."""
        trainer = trainer_with_loaders
        
        metric_value = torch.tensor(0.85)
        trainer.update_metrics(metric_value, "accuracy", validation=False)
        
        assert "accuracy" in trainer.train_metrics
        assert len(trainer.train_metrics["accuracy"]) == 1
        assert trainer.train_metrics["accuracy"][0] == metric_value
    
    def test_update_metrics_validation(self, trainer_with_loaders):
        """Verify that update_metrics correctly appends validation metrics."""
        trainer = trainer_with_loaders
        
        metric_value = torch.tensor(0.75)
        trainer.update_metrics(metric_value, "accuracy", validation=True)
        
        assert "accuracy" in trainer.val_metrics
        assert len(trainer.val_metrics["accuracy"]) == 1
        assert trainer.val_metrics["accuracy"][0] == metric_value
    
    def test_update_metrics_multiple_calls(self, trainer_with_loaders):
        """Verify that update_metrics accumulates multiple metric values."""
        trainer = trainer_with_loaders
        
        for i in range(3):
            trainer.update_metrics(torch.tensor(0.5 + i * 0.1), "metric", validation=True)
        
        assert len(trainer.val_metrics["metric"]) == 3
        assert torch.isclose(trainer.val_metrics["metric"][-1], torch.tensor(0.7))
    
    def test_log_property_combines_losses_and_metrics(self, trainer_with_loaders):
        """Verify that log property correctly combines losses and metrics."""
        trainer = trainer_with_loaders
        
        # Add some losses and metrics
        trainer.update_loss(torch.tensor(0.5), "mse", validation=False)
        trainer.update_loss(torch.tensor(0.3), "mse", validation=True)
        trainer.update_metrics(torch.tensor(0.8), "acc", validation=False)
        trainer.update_metrics(torch.tensor(0.75), "acc", validation=True)
        
        trainer.epoch = 1
        
        log = trainer.log
        
        assert "epoch" in log
        assert "mse" in log
        assert "val_mse" in log
        assert "acc" in log
        assert "val_acc" in log
    
    def test_log_property_with_no_data(self, trainer_with_loaders):
        """Verify that log property works with no losses/metrics."""
        trainer = trainer_with_loaders
        trainer.epoch = 0
        
        log = trainer.log
        
        assert "epoch" in log
        assert log["epoch"] == []


class TestEarlyTermination:
    """Test early termination logic."""
    
    def test_early_termination_disabled_by_default(self, trainer_with_loaders):
        """Verify that early termination is disabled when no metric is specified."""
        trainer = trainer_with_loaders
        
        assert trainer._early_termination is False
        assert trainer._early_termination_metric is None
    
    def test_early_termination_enabled_with_metric(self, minimal_model, minimal_optimizer, train_dataloader, val_dataloader):
        """Verify that early termination is enabled when metric is specified."""
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2,
            device=torch.device('cpu'),
            early_termination_metric='loss_a'
        )
        
        assert trainer._early_termination is True
        assert trainer._early_termination_metric == 'loss_a'
    
    def test_update_early_stop_counter_improves(self, minimal_model, minimal_optimizer, train_dataloader, val_dataloader):
        """Verify that early stop counter resets when validation loss improves."""
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2,
            device=torch.device('cpu'),
            early_termination_metric='val_loss',
            early_termination_mode='min'
        )
        
        trainer._patience = 3
        trainer._early_stop_counter = 2
        trainer._best_loss = 0.5
        
        # Add a better validation loss
        trainer.update_loss(torch.tensor(0.3), "val_loss", validation=True)
        
        should_stop = trainer.update_early_stop_counter()
        
        assert trainer.early_stop_counter == 0  # Reset
        assert trainer.best_loss == 0.3  # Updated
        assert should_stop is False
    
    def test_update_early_stop_counter_worsens(self, minimal_model, minimal_optimizer, train_dataloader, val_dataloader):
        """Verify that early stop counter increments when validation loss worsens."""
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2,
            device=torch.device('cpu'),
            early_termination_metric='val_loss',
            early_termination_mode='min'
        )
        
        trainer._patience = 3
        trainer._early_stop_counter = 1
        trainer._best_loss = 0.3
        
        # Add a worse validation loss
        trainer.update_loss(torch.tensor(0.5), "val_loss", validation=True)
        
        should_stop = trainer.update_early_stop_counter()
        
        assert trainer.early_stop_counter == 2  # Incremented
        assert trainer.best_loss == 0.3  # Not updated
        assert should_stop is False
    
    def test_update_early_stop_counter_triggers_stop(self, minimal_model, minimal_optimizer, train_dataloader, val_dataloader):
        """Verify that early stopping triggers when patience is exceeded."""
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2,
            device=torch.device('cpu'),
            early_termination_metric='val_loss',
            early_termination_mode='min'
        )
        
        trainer._patience = 3
        trainer._early_stop_counter = 2
        trainer._best_loss = 0.3
        
        # Add a worse validation loss
        trainer.update_loss(torch.tensor(0.5), "val_loss", validation=True)
        
        should_stop = trainer.update_early_stop_counter()
        
        assert trainer.early_stop_counter == 3
        assert should_stop is True
    
    def test_early_termination_mode_max(self, minimal_model, minimal_optimizer, train_dataloader, val_dataloader):
        """Verify that early termination works in 'max' mode (e.g., for accuracy)."""
        trainer = MinimalTrainerRealization(
            model=minimal_model,
            optimizer=minimal_optimizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            batch_size=2,
            device=torch.device('cpu'),
            early_termination_metric='accuracy',
            early_termination_mode='max'
        )
        
        trainer._patience = 3
        trainer._early_stop_counter = 1
        trainer._best_loss = 0.8
        
        # Add a better accuracy (higher is better)
        trainer.update_metrics(torch.tensor(0.9), "accuracy", validation=True)
        
        should_stop = trainer.update_early_stop_counter()
        
        assert trainer.early_stop_counter == 0  # Reset
        assert trainer.best_loss == 0.9  # Updated
        assert should_stop is False
    
    def test_collect_early_stop_metric_from_val_losses(self, trainer_with_loaders):
        """Verify that early stop metric is collected from val_losses."""
        trainer = trainer_with_loaders
        trainer._early_termination_metric = "mse_loss"
        
        trainer.update_loss(torch.tensor(0.5), "mse_loss", validation=True)
        
        metric = trainer._collect_early_stop_metric()
        
        assert metric == torch.tensor(0.5)
    
    def test_collect_early_stop_metric_from_val_metrics(self, trainer_with_loaders):
        """Verify that early stop metric is collected from val_metrics."""
        trainer = trainer_with_loaders
        trainer._early_termination_metric = "accuracy"
        
        trainer.update_metrics(torch.tensor(0.85), "accuracy", validation=True)
        
        metric = trainer._collect_early_stop_metric()
        
        assert metric == torch.tensor(0.85)
    
    def test_collect_early_stop_metric_invalid_metric_raises_error(self, trainer_with_loaders):
        """Verify that invalid early termination metric raises ValueError."""
        trainer = trainer_with_loaders
        trainer._early_termination_metric = "nonexistent_metric"
        
        with pytest.raises(ValueError, match="Invalid early termination metric"):
            trainer._collect_early_stop_metric()
    
    def test_early_termination_disabled_updates_best_model(self, trainer_with_loaders):
        """Verify that when early termination is disabled, best model is still updated."""
        trainer = trainer_with_loaders
        trainer._early_termination = False
        trainer._early_termination_metric = None
        
        should_stop = trainer.update_early_stop_counter()
        
        assert trainer.best_model is not None
        assert should_stop is False
