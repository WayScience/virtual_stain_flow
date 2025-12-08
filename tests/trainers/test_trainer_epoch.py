"""
test_trainer_epoch.py

Contract tests for AbstractTrainer.train_epoch and evaluate_epoch methods
"""


class TestTrainEpochBatchIteration:
    """Test that train_epoch iterates over all batches."""
    
    def test_train_epoch_iterates_all_batches(self, trainer_with_loaders):
        """Verify train_epoch processes all batches from the dataloader."""
        trainer = trainer_with_loaders
        
        # train_loader has 10 samples / 2 batch_size = 5 batches
        expected_batches = trainer._train_loader.__len__()
        
        _ = trainer.train_epoch()
        
        # Verify train_step was called exactly once per batch
        assert len(trainer.train_step_calls) == expected_batches, \
            f"Expected {expected_batches} train_step calls, got {len(trainer.train_step_calls)}"
    
    def test_evaluate_epoch_iterates_all_batches(self, trainer_with_loaders):
        """Verify evaluate_epoch processes all batches from the dataloader."""
        trainer = trainer_with_loaders
        
        # val_loader has 6 samples / 2 batch_size = 3 batches
        expected_batches = trainer._val_loader.__len__()
        
        _ = trainer.evaluate_epoch()
        
        # Verify evaluate_step was called exactly once per batch
        assert len(trainer.evaluate_step_calls) == expected_batches, \
            f"Expected {expected_batches} evaluate_step calls, got {len(trainer.evaluate_step_calls)}"
