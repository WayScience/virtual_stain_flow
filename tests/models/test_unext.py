from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from virtual_stain_flow.models.unext import ConvNeXtUNet


def _padded_convolutions(module):
    return [
        layer
        for layer in module.modules()
        if isinstance(layer, nn.Conv2d)
        and (
            layer.padding == 'same'
            if isinstance(layer.padding, str)
            else any(layer.padding)
        )
    ]


def _assert_padding_modes(model, encoder_mode, decoder_mode):
    for branch, expected in (
        (model.encoder, encoder_mode),
        *((stage.comp_block, decoder_mode) for stage in model.decoder.stages),
    ):
        convolutions = _padded_convolutions(branch)
        assert convolutions
        assert all(layer.padding_mode == expected for layer in convolutions)


@pytest.mark.parametrize(
    ('options', 'encoder_mode', 'decoder_mode'),
    [
        ({}, 'zeros', 'zeros'),
        ({'_encoder_padding_mode': 'zeros', '_decoder_padding_mode': 'reflect'}, 'zeros', 'reflect'),
        ({'padding_mode': 'reflect'}, 'reflect', 'reflect'),
        ({'padding_mode': 'reflect', '_encoder_padding_mode': 'zeros'}, 'zeros', 'reflect'),
        ({'padding_mode': 'reflect', '_decoder_padding_mode': 'zeros'}, 'reflect', 'zeros'),
        ({'_encoder_padding_mode': 'reflect', '_decoder_padding_mode': 'zeros'}, 'reflect', 'zeros'),
        ({'padding_mode': 'reflect', '_encoder_padding_mode': None, '_decoder_padding_mode': None}, 'reflect', 'reflect'),
    ],
)
@pytest.mark.parametrize('compute_block', ['convnext', 'conv2d'])
def test_unext_padding_configuration_and_roundtrip(
    options, encoder_mode, decoder_mode, compute_block,
):
    model = ConvNeXtUNet(
        in_channels=1,
        out_channels=1,
        decoder_compute_block=compute_block,
        _num_units=1,
        **options,
    )
    _assert_padding_modes(model, encoder_mode, decoder_mode)
    assert all(
        layer.padding_mode == 'zeros'
        for layer in model.encoder.modules()
        if isinstance(layer, nn.Conv2d) and layer.padding == (0, 0)
    )

    config = model.to_config()
    assert config['init']['padding_mode'] == options.get('padding_mode', 'zeros')
    assert config['init']['_encoder_padding_mode'] == encoder_mode
    assert config['init']['_decoder_padding_mode'] == decoder_mode
    restored = ConvNeXtUNet.from_config(config)
    _assert_padding_modes(restored, encoder_mode, decoder_mode)
    assert restored.to_config() == config


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('shared_mode', ['zeros', 'reflect'])
def test_from_config_inherits_shared_padding_without_mutating_input(wrapped, shared_mode):
    init = {'in_channels': 1, 'out_channels': 1, '_num_units': 1}
    if shared_mode == 'reflect':
        init['padding_mode'] = shared_mode
    config = {'init': init} if wrapped else init
    original = deepcopy(config)

    model = ConvNeXtUNet.from_config(config)

    _assert_padding_modes(model, shared_mode, shared_mode)
    assert config == original


@pytest.mark.parametrize('name', ['padding_mode', '_encoder_padding_mode', '_decoder_padding_mode'])
@pytest.mark.parametrize('invalid_mode', ['', 'replicate'])
def test_unext_rejects_invalid_padding_modes(name, invalid_mode):
    with pytest.raises(ValueError, match=f'Unsupported {name}:'):
        ConvNeXtUNet(in_channels=1, out_channels=1, **{name: invalid_mode})


def test_unext_rejects_none_shared_padding_even_with_overrides():
    with pytest.raises(ValueError, match='Unsupported padding_mode:'):
        ConvNeXtUNet(
            in_channels=1,
            out_channels=1,
            padding_mode=None,
            _encoder_padding_mode='zeros',
            _decoder_padding_mode='reflect',
        )


def test_unext_shared_reflect_padding_forward():
    model = ConvNeXtUNet(
        in_channels=1, out_channels=1, padding_mode='reflect', _num_units=1,
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 32, 32))
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()
