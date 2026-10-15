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


@pytest.mark.parametrize("scaling,limits", [("target", None), ("fixed", (0, 0.01)), ("independent", None)])
def test_model_plot_uses_one_transformed_snapshot_for_metrics_and_metadata(crop_dataset, scaling, limits):
    from virtual_stain_flow.datasets.base_wrapper_dataset import BaseWrapperDataset
    from virtual_stain_flow.transforms.gamma import ContinuousGammaTransform
    from virtual_stain_flow.transforms.normalizations import MaxScaleNormalize

    crop_dataset.input_transforms = [MaxScaleNormalize(10)]
    crop_dataset.target_transforms = [MaxScaleNormalize(10), ContinuousGammaTransform(0.3)]

    class StatefulWrapper(BaseWrapperDataset):
        def __init__(self, dataset):
            super().__init__(dataset)
            self.calls = []

        def __getitem__(self, index):
            self.calls.append(index)
            inputs, targets = self._dataset[index]
            # Simulate augmentation that changes with every read.
            return inputs + len(self.calls), targets

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, inputs):
            self.inputs.append(inputs.clone())
            return inputs[:, :1] + 0.25

    class RecordingL1(torch.nn.L1Loss):
        def __init__(self):
            super().__init__()
            self.values = []

        def forward(self, first, second):
            result = super().forward(first, second)
            self.values.append(result.item())
            return result

    dataset = StatefulWrapper(crop_dataset)
    model, metric = RecordingModel(), RecordingL1()
    figure = plot_predictions_grid_from_model(
        model, dataset, indices=[3, 0], metrics=[metric], device="cpu",
        target_scaling=scaling, target_limits=limits, show_plot=False,
    )
    assert dataset.calls == [3, 0]
    for row, (raw_value, crop_xy) in enumerate([(3, (5, 5)), (1, (0, 0))]):
        axes = figure.axes[row * 6:(row + 1) * 6]
        displayed_input = axes[2].images[0].get_array()
        displayed_target = axes[4].images[0].get_array()
        displayed_prediction = axes[5].images[0].get_array()
        np.testing.assert_allclose(displayed_input, model.inputs[row][0, 0].numpy())
        np.testing.assert_allclose(displayed_input, raw_value / 10 + row + 1)
        np.testing.assert_allclose(displayed_target, ((2 if row == 0 else 1) / 10) ** 0.3)
        np.testing.assert_allclose(displayed_prediction, displayed_input + 0.25)
        assert metric.values[row] == pytest.approx(np.abs(displayed_prediction - displayed_target).mean())
        assert f"RecordingL1: {metric.values[row]:.3f}" in axes[5].get_title()
        np.testing.assert_array_equal(axes[0].images[0].get_array(), raw_value)
        assert axes[0].patches[0].get_xy() == crop_xy


def test_dataset_wrapper_forwards_display_options(basic_dataset):
    figure = plot_dataset_grid(
        basic_dataset, [0, 1], target_scaling="fixed", target_limits=(0, 10),
        input_scaling="channel", show_plot=False,
    )
    assert figure.axes[2].images[0].get_clim() == (0, 10)
    assert figure.axes[5].images[0].get_clim() == (0, 10)
    assert figure.axes[0].images[0].get_clim() == figure.axes[3].images[0].get_clim()


def test_plot_callback_forwards_scaling_options(basic_dataset, tmp_path, monkeypatch):
    from virtual_stain_flow.vsf_logging.callbacks.PlotCallback import PlotPredictionCallback

    callback = PlotPredictionCallback(
        "test", basic_dataset, save_path=tmp_path, indices=[0],
        target_scaling="fixed", target_limits=(0, 1), input_scaling="channel",
        show_plot=False, panel_width=0.5,
    )
    monkeypatch.setattr(callback, "get_model", lambda **kwargs: torch.nn.Conv2d(2, 1, 1))
    monkeypatch.setattr(callback, "get_epoch", lambda: 0)
    assert callback._plot().is_file()


def test_inplace_model_does_not_change_displayed_snapshot(basic_dataset):
    class InplaceModel(torch.nn.Module):
        def forward(self, inputs):
            inputs.add_(100)
            return inputs[:, :1]

    figure = plot_predictions_grid_from_model(
        InplaceModel(), basic_dataset, [0], [], device="cpu", show_plot=False,
    )
    np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), 1)
    np.testing.assert_array_equal(figure.axes[1].images[0].get_array(), 2)
    np.testing.assert_array_equal(figure.axes[2].images[0].get_array(), 1)
    np.testing.assert_array_equal(figure.axes[3].images[0].get_array(), 101)
