"""
Tests for LoggingWGANTrainer train_step and evaluate_step methods
"""

import torch


class TestLoggingWGANTrainerTrainStep:
    """Tests for LoggingWGANTrainer.train_step method."""

    def test_train_step_returns_dict(self, wgan_trainer):
        """Test that train_step returns a dictionary."""
        inputs = torch.randn(2, 1, 16, 16)
        targets = torch.randn(2, 1, 16, 16)
        
        losses = wgan_trainer.train_step(inputs, targets)
        
        assert isinstance(losses, dict)

    def test_train_step_returns_generator_and_discriminator_losses(self, wgan_trainer):
        """Test that train_step returns both generator and discriminator losses."""
        inputs = torch.randn(2, 1, 16, 16)
        targets = torch.randn(2, 1, 16, 16)
        
        losses = wgan_trainer.train_step(inputs, targets)
        
        # Should have generator loss (MSE + Adversarial)
        assert 'MSELoss' in losses
        assert 'AdversarialLoss' in losses
        # Should have discriminator losses (Wasserstein + GP)
        assert 'WassersteinLoss' in losses
        assert 'GradientPenaltyLoss' in losses

    def test_train_step_updates_discriminator_every_step(self, wgan_trainer):
        """Test that discriminator is updated every training step."""
        inputs = torch.randn(2, 1, 16, 16)
        targets = torch.randn(2, 1, 16, 16)
        
        # Get initial discriminator parameters
        disc_params_before = [
            p.clone() for p in wgan_trainer._orchestrator.discriminator_forward_group.model.parameters()
        ]
        
        # Run train step
        wgan_trainer.train_step(inputs, targets)
        
        # Check that discriminator parameters changed
        disc_params_after = list(wgan_trainer._orchestrator.discriminator_forward_group.model.parameters())
        
        params_changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(disc_params_before, disc_params_after)
        )
        assert params_changed

    def test_train_step_generator_update_frequency(self, wgan_trainer):
        """Test that generator is updated according to n_discriminator_steps."""
        inputs = torch.randn(2, 1, 16, 16)
        targets = torch.randn(2, 1, 16, 16)
        
        # Reset global step to ensure consistent starting point
        wgan_trainer._global_step = 0
        
        # Get initial generator parameters
        gen_params_before = [
            p.clone() for p in wgan_trainer.model.parameters()
        ]
        
        # Run first train step (_global_step=0, should update)
        wgan_trainer.train_step(inputs, targets)
        
        gen_params_after_step0 = [
            p.clone() for p in wgan_trainer.model.parameters()
        ]
        
        # Generator should have updated on step 0
        params_changed_step0 = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(gen_params_before, gen_params_after_step0)
        )
        assert params_changed_step0, "Generator should update at step 0"
        
        # Run step 1 (_global_step=1, should NOT update)
        wgan_trainer.train_step(inputs, targets)
        
        gen_params_after_step1 = [
            p.clone() for p in wgan_trainer.model.parameters()
        ]
        
        # Generator should NOT have changed from step 0 to step 1
        params_unchanged_step1 = all(
            torch.equal(p_step0, p_step1)
            for p_step0, p_step1 in zip(gen_params_after_step0, gen_params_after_step1)
        )
        assert params_unchanged_step1, "Generator should not update at step 1"
        
        # Run step 2 (_global_step=2, should NOT update)
        wgan_trainer.train_step(inputs, targets)
        
        gen_params_after_step2 = [
            p.clone() for p in wgan_trainer.model.parameters()
        ]
        
        # Generator should NOT have changed from step 1 to step 2
        params_unchanged_step2 = all(
            torch.equal(p_step1, p_step2)
            for p_step1, p_step2 in zip(gen_params_after_step1, gen_params_after_step2)
        )
        assert params_unchanged_step2, "Generator should not update at step 2"
        
        # Run step 3 (_global_step=3, should update)
        wgan_trainer.train_step(inputs, targets)
        
        gen_params_after_step3 = [
            p.clone() for p in wgan_trainer.model.parameters()
        ]
        
        params_changed_step3 = any(
            not torch.equal(p_step2, p_step3)
            for p_step2, p_step3 in zip(gen_params_after_step2, gen_params_after_step3)
        )
        assert params_changed_step3, "Generator should update at step 3"


class TestLoggingWGANTrainerEvaluateStep:
    """Tests for LoggingWGANTrainer.evaluate_step method."""

    def test_evaluate_step_returns_dict(self, wgan_trainer):
        """Test that evaluate_step returns a dictionary."""
        inputs = torch.randn(2, 1, 16, 16)
        targets = torch.randn(2, 1, 16, 16)
        
        losses = wgan_trainer.evaluate_step(inputs, targets)
        
        assert isinstance(losses, dict)

    def test_evaluate_step_returns_generator_and_discriminator_losses(self, wgan_trainer):
        """Test that evaluate_step returns both generator and discriminator losses."""
        inputs = torch.randn(2, 1, 16, 16)
        targets = torch.randn(2, 1, 16, 16)
        
        losses = wgan_trainer.evaluate_step(inputs, targets)
        
        # Should have generator losses
        assert 'MSELoss' in losses
        assert 'AdversarialLoss' in losses
        # Should have discriminator losses
        assert 'WassersteinLoss' in losses
        assert 'GradientPenaltyLoss' in losses

    def test_evaluate_step_does_not_update_models(self, wgan_trainer):
        """Test that evaluate_step does not update generator or discriminator."""
        inputs = torch.randn(2, 1, 16, 16)
        targets = torch.randn(2, 1, 16, 16)
        
        # Get initial parameters
        gen_params_before = [p.clone() for p in wgan_trainer.model.parameters()]
        disc_params_before = [
            p.clone() for p in wgan_trainer._orchestrator.discriminator_forward_group.model.parameters()
        ]
        
        # Run evaluate step
        wgan_trainer.evaluate_step(inputs, targets)
        
        # Check that parameters did not change
        gen_params_after = list(wgan_trainer.model.parameters())
        disc_params_after = list(wgan_trainer._orchestrator.discriminator_forward_group.model.parameters())
        
        gen_changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(gen_params_before, gen_params_after)
        )
        disc_changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(disc_params_before, disc_params_after)
        )
        
        assert not gen_changed
        assert not disc_changed
