"""
test_trainer_epoch.py

Contract tests for AbstractTrainer.train_epoch and evaluate_epoch methods
"""

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
