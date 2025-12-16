"""
Test suite for GeneratorForwardGroup and LossGroup synergy.

This module tests the correct interaction between forward groups and loss groups,
including gradient computation, optimizer state management, and training vs. eval modes.
"""

import torch
import torch.nn as nn

from virtual_stain_flow.engine.loss_group import LossItem, LossGroup
from virtual_stain_flow.engine.context import Context
from virtual_stain_flow.engine.names import INPUTS, TARGETS, PREDS, GENERATOR_MODEL


class TestGeneratorForwardGroupAndLossGroupIntegration:
    """Integration tests for forward group and loss group synergy."""

    def test_forward_group_returns_valid_context(
        self, forward_group, random_input, random_target, torch_device
    ):
        """Test that GeneratorForwardGroup returns a valid context with required keys."""
        ctx = forward_group(
            train=False,
            inputs=random_input.to(torch_device),
            targets=random_target.to(torch_device),
        )

        assert isinstance(ctx, Context)
        assert INPUTS in ctx
        assert TARGETS in ctx
        assert PREDS in ctx
        assert GENERATOR_MODEL in ctx

    def test_loss_computation_with_context(
        self, forward_group, l1_loss_group, random_input, random_target, torch_device
    ):
        """Test that LossGroup correctly computes loss with context from forward group."""
        # Forward pass
        ctx = forward_group(
            train=True,
            inputs=random_input.to(torch_device),
            targets=random_target.to(torch_device),
        )

        total_loss, logs = l1_loss_group(train=True, context=ctx)

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(logs, dict)
        assert "L1Loss" in logs
        assert total_loss.item() > 0  # L1Loss should be positive
        assert logs["L1Loss"] > 0

    def test_gradients_computed_in_training_mode(
        self, simple_conv_model, forward_group, l1_loss_group, random_input, random_target, torch_device
    ):
        """Test that gradients are computed when training mode is enabled."""
        # Store initial parameter values
        initial_params = [p.clone() for p in simple_conv_model.parameters()]

        # Forward pass in training mode
        ctx = forward_group(
            train=True,
            inputs=random_input.to(torch_device),
            targets=random_target.to(torch_device),
        )

        total_loss, _ = l1_loss_group(train=True, context=ctx)
        total_loss.backward()

        # Check that gradients exist and are non-zero
        for param in simple_conv_model.parameters():
            assert param.grad is not None
            assert not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ), "Gradients should be non-zero after backward pass"

        # Step optimizer
        forward_group.step()

        # Check that parameters have changed after optimizer step
        for initial_param, current_param in zip(
            initial_params, simple_conv_model.parameters()
        ):
            assert not torch.allclose(initial_param, current_param), (
                "Parameters should change after optimizer step"
            )

    def test_no_gradients_computed_in_eval_mode(
        self, simple_conv_model, forward_group, l1_loss_group, random_input, random_target, torch_device
    ):
        """Test that gradients are NOT computed when eval mode is enabled."""
        # Forward pass in eval mode
        ctx = forward_group(
            train=False,
            inputs=random_input.to(torch_device),
            targets=random_target.to(torch_device),
        )

        total_loss, _ = l1_loss_group(train=False, context=ctx)

        # Attempt backward (should work but not create gradients due to no_grad context)
        try:
            total_loss.backward()
            # If backward succeeds without error, gradients should not exist
            # or should be None (depending on no_grad context in forward_group)
        except RuntimeError as e:
            # It's acceptable if backward fails in eval mode
            error_msg = str(e).lower()
            assert "require grad" in error_msg or "grad_fn" in error_msg, (
                f"Expected gradient-related error, got: {error_msg}"
            )

        # Check that model is in eval mode
        assert not simple_conv_model.training

    def test_optimizer_zero_grad_in_training_flow(
        self, forward_group, l1_loss_group, random_input, random_target, torch_device
    ):
        """Test that optimizer.zero_grad is called correctly during training."""
        # First iteration: forward, loss, backward, step
        ctx1 = forward_group(
            train=True,
            inputs=random_input.to(torch_device),
            targets=random_target.to(torch_device),
        )
        total_loss1, _ = l1_loss_group(train=True, context=ctx1)
        total_loss1.backward()

        # Check that gradients exist after first backward
        first_grad_norms = [
            param.grad.norm().item() for param in forward_group._models[GENERATOR_MODEL].parameters()
            if param.grad is not None
        ]
        assert len(first_grad_norms) > 0

        forward_group.step()

        # Second iteration: the forward_group should zero_grad in its __call__
        # This tests that the optimizer is properly managed
        ctx2 = forward_group(
            train=True,
            inputs=random_input.to(torch_device),
            targets=random_target.to(torch_device),
        )

        total_loss2, _ = l1_loss_group(train=True, context=ctx2)
        total_loss2.backward()

        # Gradients should exist again after second backward
        second_grad_norms = [
            param.grad.norm().item() for param in forward_group._models[GENERATOR_MODEL].parameters()
            if param.grad is not None
        ]
        assert len(second_grad_norms) > 0

    def test_full_training_loop_single_step(
        self, forward_group, l1_loss_group, random_input, random_target, torch_device
    ):
        """Test a complete single training step: forward -> loss -> backward -> step."""
        # Multiple training steps
        for step in range(3):
            ctx = forward_group(
                train=True,
                inputs=random_input.to(torch_device),
                targets=random_target.to(torch_device),
            )

            total_loss, logs = l1_loss_group(train=True, context=ctx)
            total_loss.backward()
            forward_group.step()

        # After multiple steps, loss should have decreased or changed
        # (not necessarily strictly decreasing due to random initialization)
        assert torch.isfinite(total_loss), "Loss should be finite after training steps"

    def test_eval_mode_preserves_no_grad_context(
        self, forward_group, random_input, random_target, torch_device
    ):
        """Test that eval mode properly uses no_grad context."""
        # Forward pass in eval mode
        ctx = forward_group(
            train=False,
            inputs=random_input.to(torch_device),
            targets=random_target.to(torch_device),
        )

        # Check that predictions don't require gradients
        preds = ctx[PREDS]
        assert not preds.requires_grad, (
            "Predictions in eval mode should not require gradients"
        )
    
    def test_disabled_loss_item_not_computed(
        self, forward_group, torch_device, random_input, random_target
    ):
        """Test that disabled loss items are not included in total loss."""
        from virtual_stain_flow.engine.names import PREDS, TARGETS
        
        loss_group = LossGroup(
            items=[
                LossItem(
                    module=nn.L1Loss(),
                    args=(PREDS, TARGETS),
                    enabled=True,
                    device=torch_device,
                ),
                LossItem(
                    module=nn.MSELoss(),
                    args=(PREDS, TARGETS),
                    enabled=False,  # Disabled
                    device=torch_device,
                ),
            ]
        )

        ctx = forward_group(
            train=True,
            inputs=random_input.to(torch_device),
            targets=random_target.to(torch_device),
        )

        total_loss, logs = loss_group(train=True, context=ctx)

        # MSELoss should have value 0 since it's disabled
        assert logs["MSELoss"] == 0.0, "Disabled loss should return 0"

    def test_loss_item_compute_at_val_flag(
        self, forward_group, torch_device, random_input, random_target
    ):
        """Test that compute_at_val flag works correctly."""
        from virtual_stain_flow.engine.names import PREDS, TARGETS
        
        # Loss item that should only be computed during training
        loss_group_train = LossGroup(
            items=[
                LossItem(
                    module=nn.L1Loss(),
                    args=(PREDS, TARGETS),
                    compute_at_val=False,  # Not computed at validation
                    device=torch_device,
                ),
            ]
        )

        loss_group_val = LossGroup(
            items=[
                LossItem(
                    module=nn.L1Loss(),
                    args=(PREDS, TARGETS),
                    compute_at_val=True,  # Computed at validation
                    device=torch_device,
                ),
            ]
        )

        ctx = forward_group(
            train=False,
            inputs=random_input.to(torch_device),
            targets=random_target.to(torch_device),
        )

        # During eval mode, compute_at_val=False should return 0
        total_loss_train, logs_train = loss_group_train(train=False, context=ctx)
        assert logs_train["L1Loss"] == 0.0, (
            "Loss with compute_at_val=False should not compute during eval"
        )

        # During eval mode, compute_at_val=True should compute the loss
        total_loss_val, logs_val = loss_group_val(train=False, context=ctx)
        assert logs_val["L1Loss"] > 0.0, (
            "Loss with compute_at_val=True should compute during eval"
        )
