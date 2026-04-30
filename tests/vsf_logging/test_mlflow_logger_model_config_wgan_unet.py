"""
Tests for WGAN model config logging in MlflowLogger with UNet generator.
"""

import itertools

import pytest
import torch

from virtual_stain_flow.models.discriminator import GlobalDiscriminator
from virtual_stain_flow.models.unet import UNet
from virtual_stain_flow.trainers.logging_gan_trainer import LoggingWGANTrainer
from virtual_stain_flow.vsf_logging.MlflowLogger import MlflowLogger

UNET_VARIANTS = list(
    itertools.product(
        ["conv", "maxpool"],
        ["convt", "bilinear"],
        [4, 5],
        [32, 64],
    )
)


@pytest.mark.parametrize(
    "encoder_down_block,decoder_up_block,depth,base_channels",
    UNET_VARIANTS,
)
def test_on_train_start_logs_wgan_unet_model_configs_and_loss_items(
    patched_mlflow,
    simple_loss,
    train_dataloader,
    val_dataloader,
    encoder_down_block,
    decoder_up_block,
    depth,
    base_channels,
):
    generator = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        depth=depth,
        encoder_down_block=encoder_down_block,
        decoder_up_block=decoder_up_block,
        act_type="sigmoid",
        _num_units=2,
    )
    discriminator = GlobalDiscriminator(
        n_in_channels=1,
        n_in_filters=1,
        out_activation=None,
        _conv_depth=4,
        _leaky_relu_alpha=0.2,
        _batch_norm=False,
        _pool_before_fc=False,
    )
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    trainer = LoggingWGANTrainer(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_losses=simple_loss,
        device=torch.device("cpu"),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        batch_size=2,
        n_discriminator_steps=3,
    )

    logger = MlflowLogger(
        name="logger",
        experiment_name="exp",
    )
    logger.bind_trainer(trainer)

    logger.on_train_start()

    captured = patched_mlflow

    model_artifacts = [
        artifact
        for artifact in captured["artifacts"]
        if artifact["content"] is not None
        and "class_path" in artifact["content"]
    ]
    assert model_artifacts

    generator_config = next(
        artifact["content"]
        for artifact in model_artifacts
        if artifact["content"]["class_path"].endswith("UNet")
    )
    assert "init" in generator_config

    discriminator_config = next(
        artifact["content"]
        for artifact in model_artifacts
        if artifact["content"]["class_path"].endswith("GlobalDiscriminator")
    )
    assert "init" in discriminator_config

    assert captured["tags"]["model.0.class_path"].endswith("UNet")
    assert captured["tags"]["model.1.class_path"].endswith("GlobalDiscriminator")

    assert captured["tags"]["loss.generator.0.name"] == "MSELoss"
    assert captured["tags"]["loss.generator.0.weight"] == "1.0"
    assert captured["tags"]["loss.generator.1.name"] == "AdversarialLoss"
    assert captured["tags"]["loss.generator.1.weight"] == "1.0"

    assert captured["tags"]["loss.discriminator.0.name"] == "WassersteinLoss"
    assert captured["tags"]["loss.discriminator.0.weight"] == "1.0"
    assert captured["tags"]["loss.discriminator.1.name"] == "GradientPenaltyLoss"
    assert captured["tags"]["loss.discriminator.1.weight"] == "10.0"

    logger.end_run()
