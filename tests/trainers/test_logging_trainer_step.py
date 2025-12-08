"""
Tests for SingleGeneratorTrainer train_step and evaluate_step methods.
"""

import torch


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
