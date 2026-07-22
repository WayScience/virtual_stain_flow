"""
test_data_split.py

Unit tests for the data_split module.
"""

import pytest
from torch.utils.data import Dataset, DataLoader

from virtual_stain_flow.datasets.data_split import default_random_split


class DummyDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, size: int = 100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return idx


class TestDefaultRandomSplit:
    """Tests for default_random_split function."""
    
    def test_default_split_ratios(self):
        """Test split with default ratios (0.7, 0.15, 0.15)."""
        dataset = DummyDataset(100)
        train_loader, val_loader, test_loader = default_random_split(dataset)
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        assert len(train_loader.dataset) == 70
        assert len(val_loader.dataset) == 15
        assert len(test_loader.dataset) == 15
    
    def test_custom_split_ratios(self):
        """Test split with custom ratios."""
        dataset = DummyDataset(100)
        train_loader, val_loader, test_loader = default_random_split(
            dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        
        assert len(train_loader.dataset) == 60
        assert len(val_loader.dataset) == 20
        assert len(test_loader.dataset) == 20
    
    def test_test_ratio_computed_from_remaining(self):
        """Test that test_ratio is computed from remaining when not provided."""
        dataset = DummyDataset(100)
        train_loader, val_loader, test_loader = default_random_split(
            dataset, train_ratio=0.5, val_ratio=0.3
        )
        
        assert len(train_loader.dataset) == 50
        assert len(val_loader.dataset) == 30
        assert len(test_loader.dataset) == 20  # remaining 0.2
    
    def test_invalid_train_ratio_zero(self):
        """Test that train_ratio=0.0 raises ValueError."""
        dataset = DummyDataset(100)
        with pytest.raises(ValueError, match="must be in \\(0.0, 1.0\\)"):
            default_random_split(dataset, train_ratio=0.0)
    
    def test_invalid_train_ratio_one(self):
        """Test that train_ratio=1.0 raises ValueError."""
        dataset = DummyDataset(100)
        with pytest.raises(ValueError, match="must be in \\(0.0, 1.0\\)"):
            default_random_split(dataset, train_ratio=1.0)
    
    def test_invalid_val_ratio(self):
        """Test that invalid val_ratio raises ValueError."""
        dataset = DummyDataset(100)
        with pytest.raises(ValueError, match="must be in \\(0.0, 1.0\\)"):
            default_random_split(dataset, val_ratio=1.5)
    
    def test_invalid_test_ratio(self):
        """Test that invalid test_ratio raises ValueError."""
        dataset = DummyDataset(100)
        with pytest.raises(ValueError, match="must be in \\(0.0, 1.0\\)"):
            default_random_split(dataset, test_ratio=-0.1)
    
    def test_ratios_exceeding_one(self):
        """Test that ratios summing to more than 1.0 raises ValueError."""
        dataset = DummyDataset(100)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            default_random_split(
                dataset, train_ratio=0.5, val_ratio=0.4, test_ratio=0.3
            )
    
    def test_custom_batch_size(self):
        """Test that custom batch_size is applied."""
        dataset = DummyDataset(100)
        train_loader, val_loader, test_loader = default_random_split(
            dataset, batch_size=8
        )
        
        assert train_loader.batch_size == 8
        assert val_loader.batch_size == 8
        assert test_loader.batch_size == 8
    
    def test_shuffle_parameter(self):
        """Test that shuffle parameter is passed to DataLoaders."""
        dataset = DummyDataset(100)
        
        # With shuffle=False
        train_loader, _, _ = default_random_split(dataset, shuffle=False)
        # DataLoader stores shuffle info indirectly; we check sampler type
        from torch.utils.data import SequentialSampler
        assert isinstance(train_loader.sampler, SequentialSampler)
