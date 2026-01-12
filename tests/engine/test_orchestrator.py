"""Tests for GANOrchestrator."""

import torch

from virtual_stain_flow.engine.names import INPUTS, TARGETS, PREDS


class TestGANOrchestrator:
    """Test GANOrchestrator functionality."""

    def test_discriminator_forward(self, gan_orchestrator, random_input, random_target):
        """Test that _discriminator_forward produces correct context with real and fake stacks."""
        ctx = gan_orchestrator._discriminator_forward(
            train=False,
            inputs=random_input,
            targets=random_target
        )
        
        # Check that generator outputs are present
        assert INPUTS in ctx
        assert TARGETS in ctx
        assert PREDS in ctx
        
        # Check that discriminator outputs for real and fake are present
        assert "real_stack" in ctx
        assert "fake_stack" in ctx
        assert "p_real_as_real" in ctx
        assert "p_fake_as_real" in ctx
        
        # Verify shapes
        batch_size = random_input.shape[0]
        assert ctx["p_real_as_real"].shape[0] == batch_size
        assert ctx["p_fake_as_real"].shape[0] == batch_size
        
        # Verify real_stack is concatenation of inputs and targets
        expected_real_stack = torch.cat([ctx[INPUTS], ctx[TARGETS]], dim=1)
        assert torch.allclose(ctx["real_stack"], expected_real_stack)
        
        # Verify fake_stack is concatenation of inputs and preds
        expected_fake_stack = torch.cat([ctx[INPUTS], ctx[PREDS]], dim=1)
        assert torch.allclose(ctx["fake_stack"], expected_fake_stack)

    def test_generator_forward(self, gan_orchestrator, random_input, random_target):
        """Test that _generator_forward produces correct context with generator outputs and discriminator score."""
        ctx = gan_orchestrator._generator_forward(
            train=False,
            inputs=random_input,
            targets=random_target
        )
        
        # Check that generator outputs are present
        assert INPUTS in ctx
        assert TARGETS in ctx
        assert PREDS in ctx
        
        # Check that discriminator score for fake is present
        assert "p_fake_as_real" in ctx
        
        # Verify shapes
        batch_size = random_input.shape[0]
        assert ctx[PREDS].shape[0] == batch_size
        assert ctx["p_fake_as_real"].shape[0] == batch_size
        assert ctx["p_fake_as_real"].shape[1] == 1  # Single score output
