import timm
import torch
import pytest
import torch.nn as nn

from virtual_stain_flow.models.blocks import (
	AbstractBlock,
	Conv2DConvNeXtBlock,
	Conv2DNormActBlock,
)


class DummyBlock(AbstractBlock):
	def forward(self, x):
		return x


class TestAbstractBlock:
	@pytest.mark.parametrize(
		("parameter", "value", "exception", "message"),
		[
			("in_channels", "3", TypeError, "Expected in_channels to be int"),
			("in_channels", 0, ValueError, "Expected in_channels to be positive"),
			("in_channels", -1, ValueError, "Expected in_channels to be positive"),
			("out_channels", "8", TypeError, "Expected out_channels to be int"),
			("out_channels", 0, ValueError, "Expected out_channels to be positive"),
			("out_channels", -1, ValueError, "Expected out_channels to be positive"),
			("num_units", "2", TypeError, "Expected num_units to be int"),
			("num_units", 0, ValueError, "Expected num_units to be positive"),
			("num_units", -1, ValueError, "Expected num_units to be positive"),
		],
	)
	def test_rejects_invalid_channel_and_unit_values(
		self, parameter, value, exception, message
	):
		kwargs = {"in_channels": 3, "out_channels": 8, "num_units": 2}
		kwargs[parameter] = value

		with pytest.raises(exception, match=message):
			DummyBlock(**kwargs)


class TestConv2DConvNeXtBlock:
	@pytest.mark.parametrize(
		("in_channels", "out_channels"),
		[(1, 2), (3, 8), (8, 3)],
	)
	def test_convnext_block_produces_configured_output_channels(
		self, in_channels, out_channels
	):
		block = Conv2DConvNeXtBlock(
			in_channels=in_channels, out_channels=out_channels
		)

		output = block(torch.randn(2, in_channels, 16, 16))

		assert output.shape == (2, out_channels, 16, 16)

	def test_convnext_block_defaults_output_channels_to_input_channels(self):
		block = Conv2DConvNeXtBlock(in_channels=3)

		output = block(torch.randn(2, 3, 16, 16))

		assert output.shape == (2, 3, 16, 16)

	@pytest.mark.parametrize("num_units", [1, 2, 3])
	def test_convnext_block_appends_configured_number_of_units(self, num_units):
		block = Conv2DConvNeXtBlock(
			in_channels=3, out_channels=8, num_units=num_units
		)

		convnext_units = [
			layer
			for layer in block.network
			if isinstance(layer, timm.models.convnext.ConvNeXtBlock)
		]

		assert len(convnext_units) == num_units

	def test_convnext_block_exposes_abstract_block_metadata_and_dimensions(self):
		input_tensor = torch.randn(2, 3, 12, 18)
		block = Conv2DConvNeXtBlock(in_channels=3, out_channels=8, num_units=2)

		output = block(input_tensor)

		assert block.in_channels == input_tensor.shape[1]
		assert block.out_channels == output.shape[1]
		assert block.num_units == 2
		assert block.out_h(input_tensor.shape[2]) == output.shape[2]
		assert block.out_w(input_tensor.shape[3]) == output.shape[3]


class TestConv2DNormActBlock:
	@pytest.mark.parametrize(
		("in_channels", "out_channels"),
		[(1, 2), (3, 8), (8, 3)],
	)
	def test_norm_act_block_produces_configured_output_channels(
		self, in_channels, out_channels
	):
		block = Conv2DNormActBlock(
			in_channels=in_channels, out_channels=out_channels
		)

		output = block(torch.randn(2, in_channels, 16, 16))

		assert output.shape == (2, out_channels, 16, 16)

	def test_norm_act_block_defaults_output_channels_to_input_channels(self):
		block = Conv2DNormActBlock(in_channels=3)

		output = block(torch.randn(2, 3, 16, 16))

		assert output.shape == (2, 3, 16, 16)

	@pytest.mark.parametrize("num_units", [1, 2, 3])
	def test_norm_act_block_appends_configured_number_of_convolutions(
		self, num_units
	):
		block = Conv2DNormActBlock(
			in_channels=3, out_channels=8, num_units=num_units
		)

		convolutions = [
			layer for layer in block.network if isinstance(layer, nn.Conv2d)
		]

		assert len(convolutions) == num_units

	@pytest.mark.parametrize(
		("norm_type", "expected_type"),
		[
			("batch", nn.BatchNorm2d),
			("layer", nn.GroupNorm),
			("none", nn.Identity),
		],
	)
	def test_norm_act_block_uses_configured_normalization(
		self, norm_type, expected_type
	):
		block = Conv2DNormActBlock(in_channels=3, norm_type=norm_type)

		normalization_layers = block.network[1::3]

		assert all(isinstance(layer, expected_type) for layer in normalization_layers)

	@pytest.mark.parametrize(
		("act_type", "expected_type"),
		[
			("relu", nn.ReLU),
			("gelu", nn.GELU),
			("sigmoid", nn.Sigmoid),
			("softmax", nn.Softmax),
			("none", nn.Identity),
		],
	)
	def test_norm_act_block_uses_configured_activation(
		self, act_type, expected_type
	):
		block = Conv2DNormActBlock(in_channels=3, act_type=act_type)

		activation_layers = block.network[2::3]

		assert all(isinstance(layer, expected_type) for layer in activation_layers)

	def test_norm_act_block_exposes_abstract_block_metadata_and_dimensions(self):
		input_tensor = torch.randn(2, 3, 12, 18)
		block = Conv2DNormActBlock(in_channels=3, out_channels=8, num_units=2)

		output = block(input_tensor)

		assert block.in_channels == input_tensor.shape[1]
		assert block.out_channels == output.shape[1]
		assert block.num_units == 2
		assert block.out_h(input_tensor.shape[2]) == output.shape[2]
		assert block.out_w(input_tensor.shape[3]) == output.shape[3]
