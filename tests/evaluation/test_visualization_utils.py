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
