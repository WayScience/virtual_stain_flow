import pytest
import torch

from virtual_stain_flow.models.up_down_blocks import (
	Bilinear2DUpsampleBlock,
	Conv2DDownBlock,
	ConvTrans2DUpBlock,
	IdentityBlock,
	MaxPool2DDownBlock,
	PixelShuffle2DUpBlock,
)


class TestUpDownBlocks:
	@pytest.mark.parametrize(
		("block_type", "kwargs", "in_channels", "expected_out_channels", "scale"),
		[
			(IdentityBlock, {}, 3, 3, 1),
			(IdentityBlock, {"out_channels": 8}, 3, 3, 1),
			(Conv2DDownBlock, {}, 3, 6, 0.5),
			(Conv2DDownBlock, {"out_channels": 8}, 3, 8, 0.5),
			(MaxPool2DDownBlock, {}, 3, 3, 0.5),
			(MaxPool2DDownBlock, {"out_channels": 8}, 3, 3, 0.5),
			(ConvTrans2DUpBlock, {}, 4, 2, 2),
			(ConvTrans2DUpBlock, {"out_channels": 3}, 4, 3, 2),
			(PixelShuffle2DUpBlock, {}, 4, 4, 2),
			(PixelShuffle2DUpBlock, {"out_channels": 8}, 4, 4, 2),
			(Bilinear2DUpsampleBlock, {}, 3, 3, 2),
			(Bilinear2DUpsampleBlock, {"out_channels": 8}, 3, 3, 2),
		],
	)
	def test_output_channels_and_spatial_dimensions(
		self, block_type, kwargs, in_channels, expected_out_channels, scale
	):
		input_tensor = torch.randn(2, in_channels, 12, 18)
		block = block_type(in_channels=in_channels, **kwargs)

		output = block(input_tensor)

		expected_height = int(input_tensor.shape[2] * scale)
		expected_width = int(input_tensor.shape[3] * scale)
		assert block.out_channels == expected_out_channels
		assert output.shape == (
			input_tensor.shape[0],
			expected_out_channels,
			expected_height,
			expected_width,
		)
		assert block.out_h(input_tensor.shape[2]) == output.shape[2]
		assert block.out_w(input_tensor.shape[3]) == output.shape[3]
