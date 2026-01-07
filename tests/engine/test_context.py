"""Tests for Context class."""

import pytest
import torch
import torch.nn as nn

from virtual_stain_flow.engine.context import Context, ReservedKeyTypeError
from virtual_stain_flow.engine.names import INPUTS, TARGETS, PREDS, GENERATOR_MODEL


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
        assert torch.equal(retrieved, random_input)#type: ignore

    def test_invalid_context_value_type(self):
        """Test that adding invalid context value types raises TypeError."""
        with pytest.raises(
            TypeError, 
            match="Context values must be torch.Tensor or torch.nn.Module"
        ):
            ctx = Context()
            ctx.add(invalid_value=42)  # type: ignore

    @pytest.mark.parametrize("key,expected_msg", [
        (PREDS, "Reserved key 'preds' must be a torch.Tensor"),
        (TARGETS, "Reserved key 'targets' must be a torch.Tensor"),
        (INPUTS, "Reserved key 'inputs' must be a torch.Tensor"),
    ])
    def test_context_reserved_key_type_error(self, key, expected_msg, simple_conv_model):
        """Test that reserved keys must be tensors."""
        with pytest.raises(
            ReservedKeyTypeError, 
            match=expected_msg
        ):
            ctx = Context()
            # try adding a module under a reserved tensor key
            ctx.add(**{key: simple_conv_model})

    def test_context_generator_model_addition(self, simple_conv_model):
        """Test adding generator model with reserved key."""
        ctx = Context()
        ctx.add(generator=simple_conv_model)
        assert isinstance(ctx[GENERATOR_MODEL], nn.Module)
        assert ctx[GENERATOR_MODEL] is simple_conv_model

    def test_context_reserved_model_key_type_error(self, random_input):
        """Test that reserved model keys must be torch.nn.Module."""
        with pytest.raises(
            ReservedKeyTypeError, 
            match="Reserved key 'generator' must be a torch.nn.Module"
        ):
            ctx = Context()
            # try adding a tensor under a reserved model key
            ctx.add(generator=random_input) #type: ignore


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


class TestContextDictBehavior:
    """Test dict-like behavior methods of Context."""

    def test_repr_empty(self):
        """Test __repr__ for empty context."""
        ctx = Context()
        assert repr(ctx) == "Context()"

    def test_repr_with_tensor(self, random_input):
        """Test __repr__ with tensor."""
        ctx = Context(inputs=random_input)
        repr_str = repr(ctx)
        assert "Context(" in repr_str
        assert "inputs" in repr_str
        assert "torch.float" in repr_str or "torch.float32" in repr_str

    def test_repr_with_module(self, simple_conv_model):
        """Test __repr__ with module."""
        ctx = Context(generator=simple_conv_model)
        repr_str = repr(ctx)
        assert "Context(" in repr_str
        assert "generator" in repr_str
        assert "Conv2d" in repr_str

    def test_getitem(self, random_input):
        """Test __getitem__ retrieves stored values."""
        ctx = Context(inputs=random_input)
        assert torch.equal(ctx[INPUTS], random_input) #type: ignore

    def test_getitem_missing_key(self):
        """Test __getitem__ raises KeyError for missing key."""
        ctx = Context()
        with pytest.raises(KeyError):
            ctx["nonexistent"]

    def test_iter(self, random_input, random_target):
        """Test __iter__ returns keys."""
        ctx = Context(inputs=random_input, targets=random_target)
        keys = list(iter(ctx))
        assert INPUTS in keys
        assert TARGETS in keys
        assert len(keys) == 2

    def test_len_empty(self):
        """Test __len__ on empty context."""
        ctx = Context()
        assert len(ctx) == 0

    def test_len_with_items(self, random_input, random_target):
        """Test __len__ with items."""
        ctx = Context(inputs=random_input, targets=random_target)
        assert len(ctx) == 2

    def test_values(self, random_input, random_target):
        """Test values() method."""
        ctx = Context(inputs=random_input, targets=random_target)
        values_list = list(ctx.values())
        assert len(values_list) == 2
        assert any(torch.equal(v, random_input) for v in values_list if isinstance(v, torch.Tensor))
        assert any(torch.equal(v, random_target) for v in values_list if isinstance(v, torch.Tensor))

    def test_items(self, random_input, random_target):
        """Test items() method."""
        ctx = Context(inputs=random_input, targets=random_target)
        items_list = list(ctx.items())
        assert len(items_list) == 2
        items_dict = dict(items_list)
        assert INPUTS in items_dict
        assert TARGETS in items_dict
        assert torch.equal(items_dict[INPUTS], random_input)
        assert torch.equal(items_dict[TARGETS], random_target)

    def test_keys(self, random_input, random_target):
        """Test keys() method."""
        ctx = Context(inputs=random_input, targets=random_target)
        keys_list = list(ctx.keys())
        assert len(keys_list) == 2
        assert INPUTS in keys_list
        assert TARGETS in keys_list

    def test_setitem(self, random_input):
        """Test __setitem__ to add/update values."""
        ctx = Context()
        ctx[INPUTS] = random_input
        assert torch.equal(ctx[INPUTS], random_input) #type: ignore
        assert len(ctx) == 1

    def test_setitem_override(self, random_input, random_target):
        """Test __setitem__ overrides existing value."""
        ctx = Context(inputs=random_input)
        ctx[INPUTS] = random_target
        assert torch.equal(ctx[INPUTS], random_target) #type: ignore
        assert not torch.equal(ctx[INPUTS], random_input) #type: ignore

    def test_contains_present_key(self, random_input):
        """Test __contains__ for present key."""
        ctx = Context(inputs=random_input)
        assert INPUTS in ctx
        assert "inputs" in ctx

    def test_contains_missing_key(self):
        """Test __contains__ for missing key."""
        ctx = Context()
        assert INPUTS not in ctx
        assert "nonexistent" not in ctx

    def test_get_existing_key(self, random_input):
        """Test get() with existing key."""
        ctx = Context(inputs=random_input)
        retrieved = ctx.get(INPUTS)
        assert torch.equal(retrieved, random_input) #type: ignore

    def test_get_missing_key_default_none(self):
        """Test get() with missing key returns None by default."""
        ctx = Context()
        result = ctx.get("nonexistent")
        assert result is None

    def test_get_missing_key_custom_default(self, random_input):
        """Test get() with missing key returns custom default."""
        ctx = Context()
        default_value = "default"
        result = ctx.get("nonexistent", default_value) #type: ignore
        assert result == default_value


class TestContextMerge:
    """Test Context merge operations using | operator."""

    def test_or_merge_basic(self, random_input, random_target):
        """Test basic merge of two contexts with different keys."""
        ctx1 = Context(inputs=random_input)
        ctx2 = Context(targets=random_target)
        
        merged = ctx1 | ctx2
        
        assert INPUTS in merged
        assert TARGETS in merged
        assert torch.equal(merged[INPUTS], random_input) #type: ignore
        assert torch.equal(merged[TARGETS], random_target) #type: ignore

    def test_or_merge_precedence(self, random_input, random_target):
        """Test that right operand takes precedence in key conflicts."""
        ctx1 = Context(inputs=random_input)
        ctx2 = Context(inputs=random_target)
        
        merged = ctx1 | ctx2
        
        # ctx2's value should win
        assert torch.equal(merged[INPUTS], random_target) #type: ignore
        assert not torch.equal(merged[INPUTS], random_input) #type: ignore

    def test_or_original_unchanged(self, random_input, random_target):
        """Test that original contexts are unchanged after merge."""
        ctx1 = Context(inputs=random_input)
        ctx2 = Context(targets=random_target)
        
        _ = ctx1 | ctx2
        
        # Original contexts should remain unchanged
        assert len(ctx1) == 1
        assert len(ctx2) == 1
        assert TARGETS not in ctx1
        assert INPUTS not in ctx2

    def test_ror_merge_basic(self, random_input, random_target):
        """Test reverse merge with same result as forward merge."""
        ctx1 = Context(inputs=random_input)
        ctx2 = Context(targets=random_target)
        
        # Both should produce same result for non-overlapping keys
        merged_or = ctx1 | ctx2
        merged_ror = ctx2.__ror__(ctx1)
        
        assert set(merged_or.keys()) == set(merged_ror.keys())
        assert torch.equal(merged_or[INPUTS], merged_ror[INPUTS]) #type: ignore
        assert torch.equal(merged_or[TARGETS], merged_ror[TARGETS]) #type: ignore

    def test_or_not_implemented(self, random_input):
        """Test __or__ returns NotImplemented for non-Context operand."""
        ctx = Context(inputs=random_input)
        x = {}
        
        with pytest.raises(
            NotImplementedError, 
            match="__or__ operation only supported between Context objects."
        ):
            _ = ctx | x  # type: ignore
    
    def test_ror_not_implemented(self, random_input):
        """Test __ror__ returns NotImplemented for non-Context operand."""
        ctx = Context(inputs=random_input)
        x = {}
        
        with pytest.raises(
            NotImplementedError, 
            match="__or__ operation only supported between Context objects."
        ):
            _ = x | ctx  # type: ignore
