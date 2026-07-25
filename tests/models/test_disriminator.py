import pytest
import torch
from torch import nn

from virtual_stain_flow.models.discriminator import PatchBasedDiscriminator


class TestPatchBasedDiscriminator:
	def test_forward_outputs_single_channel_feature_map(self):
		model = PatchBasedDiscriminator(in_channels=3)

		output = model(torch.randn(2, 3, 128, 128))

		assert output.ndim == 4
		assert output.shape[0] == 2
		assert output.shape[1] == 1

	@pytest.mark.parametrize("use_batch_norm", [True, False])
	def test_batch_norm_toggle_is_respected(self, use_batch_norm):
		model = PatchBasedDiscriminator(
			in_channels=3,
			_batch_norm=use_batch_norm,
		)

		norm_layers = [layer[1] for layer in model.conv_layers]

		expected_type = nn.BatchNorm2d if use_batch_norm else nn.Identity
		assert all(isinstance(layer, expected_type) for layer in norm_layers)

	@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.4])
	def test_leaky_relu_alpha_is_respected(self, alpha):
		model = PatchBasedDiscriminator(
			in_channels=3,
			_leaky_relu_alpha=alpha,
		)

		activations = [layer[2] for layer in model.conv_layers]

		assert all(isinstance(act, nn.LeakyReLU) for act in activations)
		assert all(act.negative_slope == pytest.approx(alpha) for act in activations)

	@pytest.mark.parametrize(
		("n_down_sample_layer", "expected_h_w"),
		[(1, 64), (2, 32), (3, 16)],
	)
	def test_downsample_depth_controls_spatial_reduction(
		self, n_down_sample_layer, expected_h_w
	):
		model = PatchBasedDiscriminator(
			in_channels=3,
			n_down_sample_layer=n_down_sample_layer,
			n_additional_layer=0,
		)

		conv_features = model.conv_layers(torch.randn(1, 3, 128, 128))

		assert conv_features.shape[2:] == (expected_h_w, expected_h_w)

	@pytest.mark.parametrize("n_additional_layer", [0, 1, 2])
	def test_additional_layers_use_stride1_and_double_channels(self, n_additional_layer):
		base_filters = 8
		n_down_sample_layer = 2
		channel_multiplier = 2
		model = PatchBasedDiscriminator(
			in_channels=3,
			base_filters=base_filters,
			n_down_sample_layer=n_down_sample_layer,
			n_additional_layer=n_additional_layer,
			channel_multiplier=channel_multiplier,
		)

		conv_layers = [block[0] for block in model.conv_layers]

		additional_layers = conv_layers[n_down_sample_layer:]
		assert len(additional_layers) == n_additional_layer
		assert all(layer.stride == (1, 1) for layer in additional_layers)

		for previous, current in zip(additional_layers[:-1], additional_layers[1:]):
			assert current.out_channels == previous.out_channels * channel_multiplier

		expected_last_conv_out_channels = base_filters * (
			channel_multiplier ** (n_down_sample_layer + n_additional_layer - 1)
		)
		assert conv_layers[-1].out_channels == expected_last_conv_out_channels
		assert model.out_conv.in_channels == expected_last_conv_out_channels
