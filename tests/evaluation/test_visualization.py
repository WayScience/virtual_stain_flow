import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from virtual_stain_flow.evaluation.visualization import (
    plot_dataset_grid,
    plot_predictions_grid,
    plot_predictions_grid_from_model,
)


def test_plot_predictions_grid_uses_batched_channel_arrays():
    inputs = np.zeros((2, 2, 4, 4))
    targets = np.ones((2, 1, 4, 4))
    predictions = np.full((2, 1, 4, 4), 2.0)

    figure = plot_predictions_grid(
        inputs,
        targets,
        predictions,
        input_channel_indices=[1],
        input_channel_names=["first_input", "second_input"],
        target_channel_names=["target"],
        show_plot=False,
    )

    assert len(figure.axes) == 6
    assert [axis.get_title() for axis in figure.axes[:3]] == [
        "second_input (Input 2)",
        "target (Target 1)",
        "target (Prediction 1)",
    ]


def test_plot_predictions_grid_falls_back_to_numbered_titles():
    figure = plot_predictions_grid(
        np.zeros((1, 1, 4, 4)),
        np.zeros((1, 1, 4, 4)),
        show_plot=False,
    )

    assert [axis.get_title() for axis in figure.axes] == ["Input 1", "Target 1"]


def test_plot_predictions_grid_rejects_mismatched_batches():
    with pytest.raises(ValueError, match="matching batch sizes"):
        plot_predictions_grid(
            np.zeros((2, 1, 4, 4)),
            np.zeros((1, 1, 4, 4)),
            show_plot=False,
        )


def test_plot_dataset_grid_draws_recorded_crop_rectangle(crop_dataset):
    figure = plot_dataset_grid(crop_dataset, [1], show_plot=False)

    raw_axes = figure.axes[:2]
    assert len(figure.axes) == 5
    for axis in raw_axes:
        rectangle = axis.patches[0]
        assert rectangle.get_xy() == (5, 5)
        assert rectangle.get_width() == 4
        assert rectangle.get_height() == 4


def test_plot_predictions_grid_from_model_uses_tensor_batches(basic_dataset):
    model = torch.nn.Conv2d(2, 1, kernel_size=1)

    figure = plot_predictions_grid_from_model(
        model,
        basic_dataset,
        indices=[0, 1],
        metrics=[],
        device="cpu",
        show_plot=False,
    )

    assert len(figure.axes) == 8
