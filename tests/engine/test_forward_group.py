"""Tests for ForwardGroup classes."""

import pytest
import torch

from virtual_stain_flow.engine.forward_groups import (
    AbstractForwardGroup,
    GeneratorForwardGroup,
    DiscriminatorForwardGroup
)
from virtual_stain_flow.engine.names import INPUTS, TARGETS, PREDS, GENERATOR_MODEL


class TestNormalizeOutputs:
    """Test _normalize_outputs static method."""

    def test_normalize_single_tensor(self):
        """Test normalization of single tensor output."""
        tensor = torch.randn(2, 3, 8, 8)
        result = AbstractForwardGroup._normalize_outputs(tensor)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert torch.equal(result[0], tensor)

    def test_normalize_tuple_of_tensors(self):
        """Test normalization of tuple output."""
        tensor1 = torch.randn(2, 3, 8, 8)
        tensor2 = torch.randn(2, 3, 8, 8)
        inputs = (tensor1, tensor2)
        result = AbstractForwardGroup._normalize_outputs(inputs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], tensor1)
        assert torch.equal(result[1], tensor2)

    def test_normalize_list_of_tensors(self):
        """Test normalization of list output."""
        tensor1 = torch.randn(2, 3, 8, 8)
        tensor2 = torch.randn(2, 3, 8, 8)
        inputs = [tensor1, tensor2]
        result = AbstractForwardGroup._normalize_outputs(inputs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], tensor1)
        assert torch.equal(result[1], tensor2)

    def test_normalize_dict_of_tensors(self):
        """Test normalization of dict output (preserves insertion order)."""
        tensor1 = torch.randn(2, 3, 8, 8)
        tensor2 = torch.randn(2, 3, 8, 8)
        inputs = {"first": tensor1, "second": tensor2}
        result = AbstractForwardGroup._normalize_outputs(inputs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], tensor1)
        assert torch.equal(result[1], tensor2)

    def test_normalize_unsupported_type(self):
        """Test that unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported model output type"):
            AbstractForwardGroup._normalize_outputs("string")


class TestGeneratorForwardGroup:
    """Test device management in GeneratorForwardGroup."""

    def test_cpu_forward(self, simple_conv_model, random_input, random_target):
        """Test that model is moved to specified device and forward works (CPU)."""

        device = torch.device("cpu")
        
        forward_group = GeneratorForwardGroup(
            device=device,
            **{GENERATOR_MODEL: simple_conv_model}
        )

        ctx = forward_group(train=False, inputs=random_input, targets=random_target)

        assert ctx[INPUTS].device == device
        assert ctx[TARGETS].device == device
        assert ctx[PREDS].device == device
        assert next(forward_group._models[GENERATOR_MODEL].parameters()).device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, simple_conv_model, random_input, random_target):
        """Test that model is moved to specified device and forward works (CUDA)."""
        device = torch.device("cuda:0")
        
        forward_group = GeneratorForwardGroup(
            device=device,
            **{GENERATOR_MODEL: simple_conv_model}
        )
        inputs_cpu = random_input.to("cpu")
        targets_cpu = random_target.to("cpu")

        ctx = forward_group(train=False, inputs=inputs_cpu, targets=targets_cpu)

        assert ctx[INPUTS].device == device
        assert ctx[TARGETS].device == device
        assert ctx[PREDS].device == device
        assert next(forward_group._models[GENERATOR_MODEL].parameters()).device == device

    def test_forward_missing_required_input(self, simple_conv_model, random_target):
        """Test that forward raises error when required input is missing."""
        forward_group = GeneratorForwardGroup(
            device=torch.device("cpu"),
            **{GENERATOR_MODEL: simple_conv_model}
        )
        
        with pytest.raises(ValueError, match="Missing required inputs.*inputs"):
            forward_group(train=False, targets=random_target)

    def test_forward_missing_required_target(self, simple_conv_model, random_input):
        """Test that forward raises error when required target is missing."""
        forward_group = GeneratorForwardGroup(
            device=torch.device("cpu"),
            **{GENERATOR_MODEL: simple_conv_model}
        )
        
        with pytest.raises(ValueError, match="Missing required inputs.*targets"):
            forward_group(train=False, inputs=random_input)

    def test_forward_output_arity_mismatch(self, multi_output_model, random_input, random_target):
        """Test that forward raises error when model output count doesn't match expected."""
        forward_group = GeneratorForwardGroup(
            device=torch.device("cpu"),
            generator=multi_output_model
        )
        
        with pytest.raises(ValueError, match="Model returned 2 outputs.*output_keys expects 1"):
            forward_group(train=False, inputs=random_input, targets=random_target)


class TestDiscriminatorForwardGroup:
    """Test DiscriminatorForwardGroup functionality."""

    def test_forward_train_mode(self, simple_discriminator, random_stack):
        """Test that discriminator is set to train mode when train=True."""
        forward_group = DiscriminatorForwardGroup(
            device=torch.device("cpu"),
            discriminator=simple_discriminator
        )
        
        ctx = forward_group(train=True, stack=random_stack)
        
        assert forward_group.model.training is True
        assert ctx["p"].requires_grad is True

    def test_forward_eval_mode(self, simple_discriminator, random_stack):
        """Test that discriminator is set to eval mode when train=False."""
        forward_group = DiscriminatorForwardGroup(
            device=torch.device("cpu"),
            discriminator=simple_discriminator
        )
        
        ctx = forward_group(train=False, stack=random_stack)
        
        assert forward_group.model.training is False
        assert ctx["p"].requires_grad is False

    def test_optimizer_zero_grad(self, simple_discriminator, disc_optimizer, random_stack):
        """Test that optimizer.zero_grad() is called when train=True."""
        forward_group = DiscriminatorForwardGroup(
            device=torch.device("cpu"),
            discriminator=simple_discriminator,
            optimizer=disc_optimizer
        )
        
        # Manually create some gradients
        dummy_loss = sum(p.sum() for p in forward_group.model.parameters())
        dummy_loss.backward()
        
        # Check that gradients exist
        assert any(p.grad is not None for p in forward_group.model.parameters())
        
        # Forward should zero gradients
        _ = forward_group(train=True, stack=random_stack)
        
        # Gradients should be None (set_to_none=True)
        assert all(p.grad is None for p in forward_group.model.parameters())
