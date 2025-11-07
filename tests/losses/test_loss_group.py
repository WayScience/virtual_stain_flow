import pytest
import torch
import torch.nn as nn

from virtual_stain_flow.engine.loss_group import LossItem, LossGroup


class SimpleLoss(nn.Module):
    """Simple loss for testing: MSE loss."""
    def forward(self, pred, target):
        return torch.nn.functional.mse_loss(pred, target)


class WeightedLoss(nn.Module):
    """Loss with internal weights for device testing."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, pred, target):
        return self.weight * torch.nn.functional.l1_loss(pred, target)


class TestLossItem:
    """Test single LossItem functionality."""
    
    def test_loss_item_initialization(self):
        """Test basic LossItem initialization."""
        loss_module = SimpleLoss()
        item = LossItem(
            module=loss_module,
            key="mse",
            weight=2.0,
            args=("pred", "target"),
            device=torch.device("cpu")
        )
        
        assert item.key == "mse"
        assert item.weight == 2.0
        assert item.args == ("pred", "target")
        assert item.enabled is True
        assert item.compute_at_val is True
    
    def test_loss_item_auto_key(self):
        """Test automatic key generation."""
        loss_module = SimpleLoss()
        item = LossItem(
            module=loss_module,
            args=("pred", "target"),
        )
        
        assert item.key is not None
    
    def test_loss_item_single_arg_conversion(self):
        """Test that single string arg is converted to tuple."""
        loss_module = SimpleLoss()
        item = LossItem(
            module=loss_module,
            args="pred",
        )
        
        assert isinstance(item.args, tuple)
        assert item.args == ("pred",)
    
    def test_loss_item_computation(self, sample_inputs):
        """Test loss computation returns correct values."""
        loss_module = SimpleLoss()
        item = LossItem(
            module=loss_module,
            weight=2.0,
            args=("pred", "target"),
        )
        
        raw, weighted = item(train=True, **sample_inputs)
        
        assert isinstance(raw, torch.Tensor)
        assert isinstance(weighted, torch.Tensor)
        assert weighted.item() == pytest.approx(raw.item() * 2.0)
    
    def test_loss_item_disabled(self, sample_inputs):
        """Test that disabled loss returns zeros."""
        loss_module = SimpleLoss()
        item = LossItem(
            module=loss_module,
            weight=2.0,
            args=("pred", "target"),
            enabled=False,
        )
        
        raw, weighted = item(train=True, **sample_inputs)
        
        assert raw.item() == 0.0
        assert weighted.item() == 0.0
    
    def test_loss_item_compute_at_val_false(self, sample_inputs):
        """
        Test that loss is skipped during validation with compute_at_val=False.
        """
        loss_module = SimpleLoss()
        item = LossItem(
            module=loss_module,
            weight=2.0,
            args=("pred", "target"),
            compute_at_val=False,
        )
        
        # Should compute during training
        raw_train, weighted_train = item(train=True, **sample_inputs)
        assert raw_train.item() != 0.0
        assert weighted_train.item() != 0.0
        
        # Should return zeros during validation
        raw_val, weighted_val = item(train=False, **sample_inputs)
        assert raw_val.item() == 0.0
        assert weighted_val.item() == 0.0
    
    def test_loss_item_missing_args(self, sample_inputs):
        """Test that missing arguments raise ValueError."""
        loss_module = SimpleLoss()
        item = LossItem(
            module=loss_module,
            args=("pred", "missing_arg"),
        )
        
        with pytest.raises(ValueError, match="Missing required arguments"):
            item(train=True, **sample_inputs)


class TestLossGroup:
    """Test LossGroup functionality."""
    
    def test_loss_group_initialization(self):
        """Test basic LossGroup initialization."""
        items = [
            LossItem(
                module=SimpleLoss(), 
                key="loss1", 
                args=("pred", "target")),
            LossItem(
                module=SimpleLoss(), 
                key="loss2", 
                args=("pred", "target")),
        ]
        group = LossGroup(items=items)
        
        assert len(group.items) == 2
        assert group.item_names == ["loss1", "loss2"]
    
    def test_loss_group_computation(self, sample_inputs):
        """Test loss group computes total and individual losses."""
        items = [
            LossItem(
                module=SimpleLoss(), 
                key="loss1", 
                weight=1.0, 
                args=("pred", "target")),
            LossItem(
                module=SimpleLoss(), 
                key="loss2", 
                weight=0.5, 
                args=("pred2", "target")),
        ]
        group = LossGroup(items=items)
        
        total, logs = group(train=True, **sample_inputs)
        
        assert isinstance(total, torch.Tensor)
        assert "loss1" in logs
        assert "loss2" in logs
        assert isinstance(logs["loss1"], float)
        assert isinstance(logs["loss2"], float)
    
    def test_loss_group_with_disabled_item(self, sample_inputs):
        """Test that disabled items don't contribute to total loss."""
        items = [
            LossItem(
                module=SimpleLoss(), 
                key="loss1", 
                weight=1.0, 
                args=("pred", "target"), 
                enabled=True),
            LossItem(
                module=SimpleLoss(), 
                key="loss2", 
                weight=1.0, 
                args=("pred", "target"), 
                enabled=False),
        ]
        group = LossGroup(items=items)
        
        _, logs = group(train=True, **sample_inputs)
        
        assert logs["loss1"] != 0.0
        assert logs["loss2"] == 0.0
    
    def test_loss_group_train_vs_val(self, sample_inputs):
        """Test loss group behavior differs between train and val modes."""
        items = [
            LossItem(module=SimpleLoss(), key="loss1", weight=1.0, args=("pred", "target"), compute_at_val=True),
            LossItem(module=SimpleLoss(), key="loss2", weight=1.0, args=("pred", "target"), compute_at_val=False),
        ]
        group = LossGroup(items=items)
        
        _, logs_train = group(train=True, **sample_inputs)
        _, logs_val = group(train=False, **sample_inputs)
        
        # Both should compute during training
        assert logs_train["loss1"] != 0.0
        assert logs_train["loss2"] != 0.0
        
        # Only loss1 should compute during validation
        assert logs_val["loss1"] != 0.0
        assert logs_val["loss2"] == 0.0


class TestDeviceManagement:
    """Test device management across different hardware configurations."""
    
    def test_multiple_loss_groups_different_devices(self, available_devices):
        """
        Test that multiple loss groups can operate independently 
            on different devices. This simulates having multiple separate 
            instances of trainer running on different devices. 
            All placeholder tensors generated by the items and groups should
                be on the consistent devices and not error out.
        """
        if len(available_devices) < 2:
            pytest.skip(
                "Requires at least 2 devices (CPU + GPU or multiple GPUs)")
        
        device1, device2 = available_devices[0], available_devices[1]
        
        # Create first loss group for device1
        group1_items = [
            LossItem(
                module=WeightedLoss(),
                key="loss1_dev1",
                weight=1.0,
                args=("pred", "target"),
                device=device1
            ),
            LossItem(
                module=SimpleLoss(),
                key="loss2_dev1",
                weight=0.5,
                args=("pred", "target"),
                device=device1
            ),
        ]
        group1 = LossGroup(items=group1_items)
        
        # Create second loss group for device2
        group2_items = [
            LossItem(
                module=WeightedLoss(),
                key="loss1_dev2",
                weight=1.0,
                args=("pred", "target"),
                device=device2
            ),
            LossItem(
                module=SimpleLoss(),
                key="loss2_dev2",
                weight=0.5,
                args=("pred", "target"),
                device=device2
            ),
        ]
        group2 = LossGroup(items=group2_items)
        
        # Verify all modules in group1 are on device1
        for item in group1.items:
            if hasattr(item.module, 'parameters') and \
                list(item.module.parameters()):
                assert next(item.module.parameters()).device == device1
        
        # Verify all modules in group2 are on device2
        for item in group2.items:
            if hasattr(item.module, 'parameters') and \
                list(item.module.parameters()):
                assert next(item.module.parameters()).device == device2
        
        # Create inputs for device1
        inputs_dev1 = {
            "pred": torch.randn(4, 3, 32, 32, device=device1),
            "target": torch.randn(4, 3, 32, 32, device=device1),
        }
        
        # Create inputs for device2
        inputs_dev2 = {
            "pred": torch.randn(4, 3, 32, 32, device=device2),
            "target": torch.randn(4, 3, 32, 32, device=device2),
        }
        
        # Both groups should compute independently without issues
        total1, logs1 = group1(train=True, **inputs_dev1)
        total2, logs2 = group2(train=True, **inputs_dev2)
        
        # Verify outputs are on correct devices
        assert total1.device == device1
        assert total2.device == device2
        
        # Verify logs contain expected keys
        assert "loss1_dev1" in logs1
        assert "loss2_dev1" in logs1
        assert "loss1_dev2" in logs2
        assert "loss2_dev2" in logs2
        
        # Verify all losses were computed (non-zero)
        assert logs1["loss1_dev1"] != 0.0
        assert logs1["loss2_dev1"] != 0.0
        assert logs2["loss1_dev2"] != 0.0
        assert logs2["loss2_dev2"] != 0.0
    
    def test_loss_group_all_items_same_device(self, available_devices):
        """
        Test that all items within a single loss group work correctly 
        when they're all on the same device.
        """
        if len(available_devices) < 2:
            pytest.skip("Requires at least 2 devices")
        
        # Use GPU if available
        device = available_devices[1]  
        
        items = [
            LossItem(
                module=WeightedLoss(),
                key="loss1",
                weight=1.0,
                args=("pred", "target"),
                device=device
            ),
            LossItem(
                module=SimpleLoss(),
                key="loss2",
                weight=0.5,
                args=("pred", "target"),
                device=device
            ),
        ]
        
        # Verify all modules are on the same device
        assert next(items[0].module.parameters()).device == device
        
        inputs = {
            "pred": torch.randn(4, 3, 32, 32, device=device),
            "target": torch.randn(4, 3, 32, 32, device=device),
        }
        
        group = LossGroup(items=items)
        total, logs = group(train=True, **inputs)
        
        assert total.device == device
        assert "loss1" in logs
        assert "loss2" in logs
        assert logs["loss1"] != 0.0
        assert logs["loss2"] != 0.0
    
    def test_loss_item_device_move_failure(self):
        """Test that device move failures are handled appropriately."""
        
        class FailingLoss(nn.Module):
            def to(self, device):
                raise RuntimeError("Cannot move to device")
            
            def forward(self, pred, target):
                return torch.tensor(0.0)
        
        with pytest.raises(RuntimeError, match="Failed to move loss module"):
            LossItem(
                module=FailingLoss(),
                args=("pred", "target"),
                device=torch.device("cpu")
            )
