"""
test_trainer_epoch.py

Contract tests for AbstractTrainer.train_epoch and evaluate_epoch methods
"""


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
