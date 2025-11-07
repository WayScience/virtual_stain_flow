"""Tests for ForwardGroup classes."""

import pytest
import torch

from virtual_stain_flow.engine.forward_groups import (
    GeneratorForwardGroup
)
from virtual_stain_flow.engine.names import INPUTS, TARGETS, PREDS, GENERATOR_MODEL


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
        assert next(forward_group.models[GENERATOR_MODEL].parameters()).device == device

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
        assert next(forward_group.models[GENERATOR_MODEL].parameters()).device == device

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
