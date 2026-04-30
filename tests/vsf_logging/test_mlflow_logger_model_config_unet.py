"""
Tests for UNet model config logging in MlflowLogger.
"""

import itertools

import pytest
import torch

from virtual_stain_flow.models.unet import UNet
from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
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
def test_on_train_start_logs_unet_model_config_and_loss_items(
    patched_mlflow,
    simple_loss,
    train_dataloader,
    val_dataloader,
    encoder_down_block,
    decoder_up_block,
    depth,
    base_channels,
):
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        depth=depth,
        encoder_down_block=encoder_down_block,
        decoder_up_block=decoder_up_block,
        act_type="sigmoid",
        _num_units=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = SingleGeneratorTrainer(
        model=model,
        optimizer=optimizer,
        losses=simple_loss,
        device=torch.device("cpu"),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        batch_size=2,
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

    model_config = next(
        artifact["content"]
        for artifact in model_artifacts
        if artifact["content"]["class_path"].endswith("UNet")
    )
    assert "init" in model_config

    assert captured["tags"]["loss.main.0.name"] == "MSELoss"
    assert captured["tags"]["loss.main.0.weight"] == "1.0"

    loss_group_artifacts = [
        artifact
        for artifact in captured["artifacts"]
        if artifact["content"] is not None
        and artifact["content"].get("group_name") == "main"
    ]
    assert len(loss_group_artifacts) == 1

    logger.end_run()
