"""Tests for Context class."""

import pytest
import torch
import torch.nn as nn

from virtual_stain_flow.engine.context import Context, ReservedKeyTypeError
from virtual_stain_flow.engine.names import INPUTS, TARGETS, PREDS


class TestContextBasics:
    """Test basic Context functionality."""

    def test_context_initialization_empty(self):
        """Test creating an empty context."""
        ctx = Context()
        assert repr(ctx) == "Context()"

    @pytest.mark.parametrize("method", ["init", "add"])
    def test_context_with_tensors(self, random_input, random_target, method):
        """Test creating/adding context with tensors via initialization or add()."""
        if method == "init":
            ctx = Context(inputs=random_input, targets=random_target)
        else:
            ctx = Context()
            ctx.add(inputs=random_input, targets=random_target)
        
        assert ctx[INPUTS].shape == random_input.shape
        assert ctx[TARGETS].shape == random_target.shape

    @pytest.mark.parametrize("method", ["init", "add"])
    def test_context_with_module(self, simple_conv_model, method):
        """Test creating/adding context with module via initialization or add()."""
        if method == "init":
            ctx = Context(model=simple_conv_model)
        else:
            ctx = Context()
            ctx.add(model=simple_conv_model)
        
        assert isinstance(ctx["model"], nn.Module)

    def test_context_getitem(self, random_input):
        """Test retrieving items from context."""
        ctx = Context(inputs=random_input)
        retrieved = ctx[INPUTS]
        assert torch.equal(retrieved, random_input)

    @pytest.mark.parametrize("key,value,expected_msg", [
        (PREDS, "not a tensor", "Reserved key 'preds' must be a torch.Tensor"),
        (TARGETS, 42, "Reserved key 'targets' must be a torch.Tensor"),
        (INPUTS, [1, 2, 3], "Reserved key 'inputs' must be a torch.Tensor"),
    ])
    def test_context_reserved_key_type_error(self, key, value, expected_msg):
        """Test that reserved keys must be tensors."""
        with pytest.raises(ReservedKeyTypeError, match=expected_msg):
            ctx = Context()
            ctx.add(**{key: value})


class TestContextRequire:
    """Test Context.require() functionality."""

    def test_require_success(self, random_input, random_target):
        """Test require passes when all keys are present."""
        ctx = Context(inputs=random_input, targets=random_target)
        ctx.require([INPUTS, TARGETS])  # Should not raise

    def test_require_empty_list(self):
        """Test require with empty list of keys."""
        ctx = Context()
        ctx.require([])  # Should not raise

    @pytest.mark.parametrize("present_keys,required_keys,missing_pattern", [
        ({INPUTS: "dummy"}, [INPUTS, TARGETS], "targets"),
        ({}, [INPUTS, TARGETS], "inputs.*targets"),
        ({TARGETS: "dummy"}, [INPUTS, TARGETS, PREDS], "inputs.*preds"),
    ])
    def test_require_missing_keys(self, random_input, present_keys, required_keys, missing_pattern):
        """Test require raises when keys are missing."""
        # Build context with present keys using actual tensor
        ctx_kwargs = {k: random_input for k in present_keys.keys()}
        ctx = Context(**ctx_kwargs)
        
        with pytest.raises(ValueError, match=f"Missing required inputs.*{missing_pattern}"):
            ctx.require(required_keys)


class TestContextAsKwargs:
    """Test Context.as_kwargs() functionality."""

    def test_as_kwargs(self):
        """Test as_kwargs on empty context."""
        ctx = Context()
        kwargs = ctx.as_kwargs()
        assert isinstance(kwargs, dict)


class TestContextAsMetricArgs:
    """Test Context.as_metric_args() functionality."""

    def test_as_metric_args_success(self, random_input, random_target):
        """Test as_metric_args returns (preds, targets) tuple."""
        ctx = Context(preds=random_input, targets=random_target)
        preds, targets = ctx.as_metric_args()
        assert torch.equal(preds, random_input)
        assert torch.equal(targets, random_target)
