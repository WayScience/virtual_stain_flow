"""Tests for fixed interval gamma correction and its analytic inverse."""


import numpy as np
import pytest

from virtual_stain_flow.transforms import (
    ChannelwiseTransform,
    ContinuousGammaTransform,
)


def test_defaults_and_repr():
    transform = ContinuousGammaTransform(gamma=0.5)
    assert (transform.gamma, transform.lower, transform.upper, transform.p) == (
        0.5, 0.0, 1.0, 1.0
    )
    assert repr(transform) == (
        "ContinuousGammaTransform(name=ContinuousGammaTransform, gamma=0.5, "
        "lower=0.0, upper=1.0, p=1.0)"
    )


@pytest.mark.parametrize("gamma", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_default_interval_is_power_law(gamma):
    image = np.linspace(0, 1, 101).reshape(1, 101)
    actual = ContinuousGammaTransform(gamma=gamma)(image=image)["image"]
    np.testing.assert_allclose(actual, image**gamma, rtol=1e-14, atol=0)


def test_selected_interval_and_known_values():
    image = np.array([0, 0.1, 0.125, 0.2, 0.325, 0.5, 0.8, 1.0])
    transform = ContinuousGammaTransform(gamma=0.5, lower=0.1, upper=0.5)
    actual = transform.apply(image)
    np.testing.assert_allclose(actual, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1])
    np.testing.assert_array_equal(actual[[0, 1, 5, 6, 7]], image[[0, 1, 5, 6, 7]])


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_strict_monotonicity_continuity_and_fixed_points(gamma):
    transform = ContinuousGammaTransform(gamma=gamma, lower=0.1, upper=0.7)
    x = np.linspace(0, 1, 10001)
    y = transform.apply(x)
    assert np.all(np.diff(y) > 0)
    assert y.min() == 0 and y.max() == 1
    for boundary in (0.1, 0.7):
        adjacent = np.array([
            np.nextafter(boundary, 0), boundary, np.nextafter(boundary, 1)
        ])
        result = transform.apply(adjacent)
        assert result[1] == boundary
        np.testing.assert_allclose(result, boundary, atol=1e-8, rtol=0)


@pytest.mark.parametrize("gamma", [0.25, 0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.1, 0.7)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inverse_round_trip_both_directions(gamma, bounds, dtype):
    # Stay away from a nonzero lower endpoint for large powers: rounding there
    # can discard the tiny offset, a finite-precision rather than analytic limit.
    image = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1], dtype=dtype)
    transform = ContinuousGammaTransform(gamma, lower=bounds[0], upper=bounds[1])
    inverse = transform.inverse()
    assert inverse.gamma == 1 / gamma
    assert (inverse.lower, inverse.upper, inverse.p) == (*bounds, 1.0)
    tolerance = 2e-6 if dtype == np.float32 else 1e-13
    np.testing.assert_allclose(
        inverse.apply(transform.apply(image)), image, atol=tolerance, rtol=0
    )
    np.testing.assert_allclose(
        transform.apply(inverse.apply(image)), image, atol=tolerance, rtol=0
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.01, 1.01])
def test_invalid_pixels_rejected_not_clipped(value):
    with pytest.raises(ValueError, match="finite values|normalized"):
        ContinuousGammaTransform(0.5).apply(np.array([0.0, value, 1.0]))


def test_channelwise_correction_and_unaffected_channel():
    image = np.linspace(0, 1, 48).reshape(3, 4, 4)
    gamma = ContinuousGammaTransform(0.5)
    wrapper = ChannelwiseTransform([gamma, None, ContinuousGammaTransform(2)])
    corrected = wrapper.apply(image)
    np.testing.assert_allclose(corrected[0], np.sqrt(image[0]))
    np.testing.assert_array_equal(corrected[1], image[1])
    np.testing.assert_allclose(corrected[2], image[2] ** 2)
