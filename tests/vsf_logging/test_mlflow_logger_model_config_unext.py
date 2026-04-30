"""
Tests for ConvNeXtUNet model config logging in MlflowLogger.
"""

import itertools

import pytest
import torch

from virtual_stain_flow.models.unext import ConvNeXtUNet
from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
from virtual_stain_flow.vsf_logging.MlflowLogger import MlflowLogger

UNEXT_VARIANTS = list(
    itertools.product(
        ["pixelshuffle", "convt"],
        ["convnext", "conv2d"],
    )
)


@pytest.mark.parametrize(
    "decoder_up_block,decoder_compute_block",
    UNEXT_VARIANTS,
)
def test_on_train_start_logs_unext_model_config_and_loss_items(
    patched_mlflow,
    simple_loss,
    train_dataloader,
    val_dataloader,
    decoder_up_block,
    decoder_compute_block,
):
    model = ConvNeXtUNet(
        in_channels=1,
        out_channels=1,
        decoder_up_block=decoder_up_block,
        decoder_compute_block=decoder_compute_block,
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
        if artifact["content"]["class_path"].endswith("ConvNeXtUNet")
    )
    assert "init" in model_config

    assert captured["tags"]["model.0.class_path"].endswith("ConvNeXtUNet")

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
