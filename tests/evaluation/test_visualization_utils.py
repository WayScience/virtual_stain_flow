import numpy as np
import pytest

from virtual_stain_flow.datasets.base_wrapper_dataset import BaseWrapperDataset
from virtual_stain_flow.evaluation.visualization_utils import extract_samples_from_dataset


def test_extract_samples_returns_channel_first_batches(basic_dataset):
    (
        inputs, targets, raw_images, patch_coords, input_channel_names, target_channel_names
    ) = extract_samples_from_dataset(basic_dataset, [2, 0])

    assert inputs.shape == (2, 2, 10, 10)
    assert targets.shape == (2, 1, 10, 10)
    assert raw_images is None
    assert patch_coords is None
    assert input_channel_names == ["input_ch1", "input_ch2"]
    assert target_channel_names == ["target_ch1"]
    assert np.all(inputs[0, 0] == 5)
    assert np.all(inputs[1, 1] == 2)


def test_extract_crop_samples_preserves_raw_channel_stack(crop_dataset):
    (
        inputs, targets, raw_images, patch_coords, input_channel_names, target_channel_names
    ) = extract_samples_from_dataset(crop_dataset, [1, 2])

    assert inputs.shape == (2, 2, 4, 4)
    assert targets.shape == (2, 1, 4, 4)
    assert raw_images.shape == (2, 2, 10, 10)
    assert patch_coords == [(5, 5, 4, 4), (0, 0, 4, 4)]
    assert input_channel_names == ["input_ch1", "input_ch2"]
    assert target_channel_names == ["target_ch1"]
    assert np.all(raw_images[0, 0] == 1)
    assert np.all(raw_images[0, 1] == 2)


def test_extract_samples_rejects_multiple_input_images(basic_dataset):
    class MultiInputDataset(BaseWrapperDataset):
        def __getitem__(self, index):
            input_image, target_image = self._dataset[index]
            return [input_image, input_image], target_image

    with pytest.raises(ValueError, match="single channel-first image"):
        extract_samples_from_dataset(MultiInputDataset(basic_dataset), [0])


def test_snapshot_copies_reused_buffers(basic_dataset):
    import torch

    class ReusingWrapper(BaseWrapperDataset):
        def __init__(self, dataset):
            super().__init__(dataset)
            self.buffer = torch.empty((1, 2, 2))

        def __getitem__(self, index):
            self.buffer.fill_(index)
            return self.buffer, self.buffer

    dataset = ReusingWrapper(basic_dataset)
    inputs, targets, *_ = extract_samples_from_dataset(dataset, [2, 0, 2])
    dataset.buffer.fill_(-1)
    for images in (inputs, targets):
        np.testing.assert_array_equal(images[:, 0, 0, 0], [2, 0, 2])


def test_snapshot_copies_reused_raw_buffers(crop_dataset, monkeypatch):
    dataset_type = type(crop_dataset)
    original_getitem = dataset_type.__getitem__
    raw_buffer = np.empty((2, 10, 10))

    def getitem(self, index):
        result = original_getitem(self, index)
        raw_buffer.fill(index)
        return result

    monkeypatch.setattr(dataset_type, "__getitem__", getitem)
    monkeypatch.setattr(dataset_type, "original_input_image", property(lambda self: raw_buffer))
    _, _, raw, coords, *_ = extract_samples_from_dataset(crop_dataset, [1, 2])
    raw_buffer.fill(-1)
    np.testing.assert_array_equal(raw[:, 0, 0, 0], [1, 2])
    assert coords == [(5, 5, 4, 4), (0, 0, 4, 4)]


@pytest.mark.parametrize("fixture_name", ["basic_dataset", "crop_dataset"])
def test_snapshot_matches_dataloader_with_separate_transforms(request, fixture_name):
    from torch.utils.data import DataLoader, Subset
    from virtual_stain_flow.transforms.gamma import ContinuousGammaTransform
    from virtual_stain_flow.transforms.normalizations import MaxScaleNormalize

    dataset = request.getfixturevalue(fixture_name)
    dataset.transforms = [MaxScaleNormalize(10)]
    dataset.input_transforms = [MaxScaleNormalize(2)]
    dataset.target_transforms = [ContinuousGammaTransform(0.3)]
    inputs, targets, *_ = extract_samples_from_dataset(dataset, [2, 0])
    loader_inputs, loader_targets = next(iter(DataLoader(Subset(dataset, [2, 0]), batch_size=2)))
    np.testing.assert_array_equal(inputs, loader_inputs.numpy())
    np.testing.assert_array_equal(targets, loader_targets.numpy())
    np.testing.assert_allclose(inputs[1, 0], 1 / 10 / 2)
    np.testing.assert_allclose(targets[1, 0], (1 / 10) ** 0.3)
