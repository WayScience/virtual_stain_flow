"""Regression tests for the actual Matplotlib image mappings, not just titles."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from virtual_stain_flow.evaluation.visualization import plot_predictions_grid


def _images(figure):
    return [axis.images[0] for axis in figure.axes]


def _grid(**kwargs):
    targets = np.array([0.2, 0.3, 0.5, 0.6]).reshape(1, 1, 2, 2)
    predictions = np.array([-0.1, 0.3, 0.5, 1.2]).reshape(1, 1, 2, 2)
    return plot_predictions_grid(
        targets * 10, targets, predictions, show_plot=False, **kwargs
    )


def test_default_is_target_only_and_clips_prediction_display_not_data():
    figure = _grid()
    input_image, target, prediction = _images(figure)
    assert target.norm is prediction.norm
    assert target.get_clim() == (0.2, 0.6)
    assert input_image.get_clim() == (2.0, 6.0)
    assert prediction.norm(-0.1) == 0
    assert prediction.norm(1.2) == 1
    assert prediction.cmap(prediction.norm(-0.1))[:3] == (0, 0, 0)
    assert prediction.cmap(prediction.norm(1.2))[:3] == (1, 1, 1)
    assert prediction.get_array().min() == -0.1
    assert prediction.get_array().max() == 1.2
    # Force rendering too, including the shared norm's interaction with imshow.
    figure.canvas.draw()
    assert target.get_clim() == prediction.get_clim() == (0.2, 0.6)


@pytest.mark.parametrize(
    "mode,limits,target_range,prediction_range",
    [
        ("target", None, (0.2, 0.6), (0.2, 0.6)),
        ("joint", None, (-0.1, 1.2), (-0.1, 1.2)),
        ("independent", None, (0.2, 0.6), (-0.1, 1.2)),
        ("fixed", (0, 1), (0, 1), (0, 1)),
    ],
)
def test_selectable_target_modes(mode, limits, target_range, prediction_range):
    _, target, prediction = _images(_grid(target_scaling=mode, target_limits=limits))
    assert target.get_clim() == target_range
    assert prediction.get_clim() == prediction_range


@pytest.mark.parametrize("mode", ["target", "joint", "independent"])
def test_channel_scope_shares_rows_but_not_channels(mode):
    targets = np.array([
        [[[1, 2], [3, 4]], [[10, 20], [30, 40]]],
        [[[2, 3], [4, 5]], [[20, 30], [40, 50]]],
    ], dtype=float)
    predictions = targets * 2
    figure = plot_predictions_grid(
        targets[:, :1], targets, predictions, target_scaling=mode,
        target_scale_scope="channel", show_plot=False,
    )
    images = _images(figure)
    target_ranges = [(1, 5), (10, 50)]
    prediction_ranges = [(2, 10), (20, 100)]
    if mode == "joint":
        target_ranges = [(1, 10), (10, 100)]
    if mode != "independent":
        prediction_ranges = target_ranges
    for channel in range(2):
        target_column = 1 + 2 * channel
        assert images[target_column].norm is images[target_column + 5].norm
        assert images[target_column + 1].norm is images[target_column + 6].norm
        assert images[target_column].get_clim() == target_ranges[channel]
        assert images[target_column + 1].get_clim() == prediction_ranges[channel]


def test_image_scope_keeps_rows_independent_by_default():
    targets = np.arange(8, dtype=float).reshape(2, 1, 2, 2)
    images = _images(plot_predictions_grid(targets, targets, targets * 2, show_plot=False))
    assert images[1].get_clim() == images[2].get_clim() == (0, 3)
    assert images[4].get_clim() == images[5].get_clim() == (4, 7)


def test_fixed_limits_follow_selected_channel_order():
    images = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
    artists = _images(plot_predictions_grid(
        images, images, images, raw_images=images,
        input_channel_indices=[2, 0], raw_channel_indices=[1],
        target_channel_indices=[2, 0], prediction_channel_indices=[0, 1],
        target_scaling="fixed", target_limits=[(0, 100), (-1, 1)],
        input_scaling="fixed", input_limits=[(0, 30), (0, 10)],
        raw_scaling="fixed", raw_limits=(0, 65535), show_plot=False,
    ))
    expected = [(0, 65535), (0, 30), (0, 10), (0, 100), (0, 100), (-1, 1), (-1, 1)]
    assert [image.get_clim() for image in artists] == expected * 2
    np.testing.assert_array_equal(artists[3].get_array(), images[0, 2])
    np.testing.assert_array_equal(artists[4].get_array(), images[0, 0])


@pytest.mark.parametrize("scaling", ["independent", "channel"])
def test_input_and_raw_scaling_are_separate_from_each_other_and_targets(scaling):
    images = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    artists = _images(plot_predictions_grid(
        images, images[:, :1] / 100, raw_images=images * 100,
        input_scaling=scaling, raw_scaling=scaling, show_plot=False,
    ))
    if scaling == "channel":
        ranges = [(0, 1100), (400, 1500), (0, 11), (4, 15)]
        for start in (0, 5):
            assert [image.get_clim() for image in artists[start:start + 4]] == ranges
    else:
        assert artists[0].get_clim() == (0, 300)
        assert artists[5].get_clim() == (800, 1100)
        assert artists[2].get_clim() == (0, 3)
        assert artists[7].get_clim() == (8, 11)


def test_constant_target_still_exposes_under_and_over_prediction():
    target = np.full((1, 1, 2, 2), 0.5)
    prediction = np.array([0.4, 0.5, 0.6, np.nan]).reshape(1, 1, 2, 2)
    figure = plot_predictions_grid(target, target, prediction, show_plot=False)
    _, target_artist, pred_artist = _images(figure)
    assert pred_artist.norm is target_artist.norm
    assert pred_artist.get_clim() == (0.5, 0.5)
    np.testing.assert_array_equal(pred_artist.norm([0.4, 0.5, 0.6]), [0, 0.5, 1])
    assert np.ma.is_masked(pred_artist.norm(np.nan))
    mapped = pred_artist.cmap(pred_artist.norm(prediction[0, 0]))
    np.testing.assert_allclose(
        mapped,
        [[[0, 0, 0, 1], [0.5, 0.5, 0.5, 1]],
         [[1, 1, 1, 1], [0, 0, 0, 0]]],
        atol=1 / 255,
    )


def test_legacy_constant_images_keep_matplotlib_behavior():
    images = np.ones((1, 1, 2, 2))
    artists = _images(plot_predictions_grid(
        images, images, images, target_scaling="independent", show_plot=False,
    ))
    assert artists[1].norm(1.0) == 0


def test_auto_limits_ignore_nonfinite_values():
    targets = np.array([np.nan, np.inf, 0.2, 0.8]).reshape(1, 1, 2, 2)
    artists = _images(plot_predictions_grid(np.zeros_like(targets), targets, show_plot=False))
    assert artists[1].get_clim() == (0.2, 0.8)
    with pytest.raises(ValueError, match="no finite values"):
        plot_predictions_grid(np.zeros_like(targets), targets * np.nan, show_plot=False)


def test_joint_limits_use_finite_target_when_prediction_channel_is_all_nonfinite():
    targets = np.array([np.nan, np.inf, 0.2, 0.8]).reshape(1, 1, 2, 2)
    predictions = np.full_like(targets, np.nan)
    artists = _images(plot_predictions_grid(
        np.zeros_like(targets), targets, predictions,
        target_scaling="joint", show_plot=False,
    ))
    assert artists[1].get_clim() == artists[2].get_clim() == (0.2, 0.8)

    predictions = np.array([-0.1, np.inf, 0.5, np.nan]).reshape(1, 1, 2, 2)
    artists = _images(plot_predictions_grid(
        np.zeros_like(targets), np.full_like(targets, np.nan), predictions,
        target_scaling="joint", show_plot=False,
    ))
    assert artists[1].get_clim() == artists[2].get_clim() == (-0.1, 0.5)

    predictions = np.full_like(targets, np.nan)
    with pytest.raises(ValueError, match="no finite values"):
        plot_predictions_grid(
            np.zeros_like(targets), predictions, predictions,
            target_scaling="joint", show_plot=False,
        )


@pytest.mark.parametrize("kwargs", [
    {"target_scaling": "invalid"},
    {"target_scale_scope": "invalid"},
    {"target_scaling": "fixed"},
    {"target_limits": (0, 1)},
    {"target_scaling": "fixed", "target_limits": (1, 0)},
    {"target_scaling": "fixed", "target_limits": (0, 0)},
    {"target_scaling": "fixed", "target_limits": (0, np.inf)},
    {"target_scaling": "fixed", "target_limits": [(0, 1), (0, 2)]},
    {"input_scaling": "invalid"},
    {"input_scaling": "fixed"},
    {"input_limits": (0, 1)},
    {"raw_scaling": "invalid", "raw_images": np.ones((1, 1, 2, 2))},
])
def test_invalid_scaling_settings_raise(kwargs):
    with pytest.raises(ValueError):
        _grid(**kwargs)


def test_without_predictions_joint_uses_target_range_and_does_not_mutate_arrays():
    inputs = np.arange(4, dtype=float).reshape(1, 1, 2, 2)
    targets = inputs / 10
    original = targets.copy()
    artists = _images(plot_predictions_grid(inputs, targets, target_scaling="joint", show_plot=False))
    assert artists[1].get_clim() == (0, 0.3)
    np.testing.assert_array_equal(targets, original)
