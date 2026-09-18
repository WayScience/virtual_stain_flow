"""Display-only intensity scaling; never modifies image or metric data."""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.colors import Normalize

DisplayLimits = Union[Tuple[float, float], Sequence[Tuple[float, float]]]


class _ClippedNormalize(Normalize):
    """Handle a constant reference without hiding brighter predictions.

    For a zero-width range, equal values are middle gray, lower values are
    black and higher values are white (with the gray colormap).
    """

    def __call__(self, value, clip=None):
        if self.vmin != self.vmax:
            return super().__call__(value, clip=clip)
        result, is_scalar = self.process_value(value)
        normalized = np.ma.array(
            np.where(result.data < self.vmin, 0.0,
                     np.where(result.data > self.vmax, 1.0, 0.5)),
            mask=np.ma.getmaskarray(result) | np.isnan(result.data),
        )
        return normalized[0] if is_scalar else normalized


def _fixed_limits(limits: Optional[DisplayLimits], channels: int, name: str) -> np.ndarray:
    """Validate a common pair or one pair per *displayed* channel."""
    try:
        values = np.asarray(limits, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a (min, max) pair or one pair per displayed channel.") from error
    if values.shape == (2,):
        values = np.tile(values, (channels, 1))
    if values.shape != (channels, 2) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite (min, max) pairs for {channels} displayed channels.")
    if np.any(values[:, 0] >= values[:, 1]):
        raise ValueError(f"{name} requires min < max.")
    return values


def _finite_range(images: np.ndarray) -> Tuple[float, float]:
    finite = images[np.isfinite(images)]
    if not finite.size:
        raise ValueError("Cannot determine display limits from an image/channel with no finite values.")
    return float(finite.min()), float(finite.max())


def image_norms(
    images: np.ndarray,
    *,
    scope: str = "image",
    limits: Optional[DisplayLimits] = None,
    other: Optional[np.ndarray] = None,
    legacy: bool = False,
) -> List[List[Normalize]]:
    """Construct per-row/channel norms, optionally pooling another image stack."""
    if scope not in ("image", "channel"):
        raise ValueError("Display scale scope must be 'image' or 'channel'.")
    fixed = _fixed_limits(limits, images.shape[1], "Display limits") if limits is not None else None
    rows: List[List[Normalize]] = []
    for row in range(images.shape[0]):
        if row and (scope == "channel" or fixed is not None):
            rows.append(rows[0])
            continue
        norms = []
        for channel in range(images.shape[1]):
            if fixed is not None:
                lower, upper = fixed[channel]
            else:
                reference = images[:, channel] if scope == "channel" else images[row, channel]
                lower, upper = _finite_range(reference)
                if other is not None:
                    comparison = other[:, channel] if scope == "channel" else other[row, channel]
                    other_lower, other_upper = _finite_range(comparison)
                    lower, upper = min(lower, other_lower), max(upper, other_upper)
            norm_type = Normalize if legacy else _ClippedNormalize
            norms.append(norm_type(vmin=lower, vmax=upper, clip=True))
        rows.append(norms)
    return rows


def unpaired_norms(
    images: np.ndarray, scaling: str, limits: Optional[DisplayLimits], name: str
) -> List[List[Normalize]]:
    """Input/raw panels have their own independent, channel-shared or fixed scale."""
    if scaling not in ("independent", "channel", "fixed"):
        raise ValueError(f"{name}_scaling must be 'independent', 'channel', or 'fixed'.")
    if (scaling == "fixed") != (limits is not None):
        raise ValueError(f"{name}_limits must be provided if and only if {name}_scaling='fixed'.")
    return image_norms(
        images, scope="channel" if scaling == "channel" else "image",
        limits=limits, legacy=scaling == "independent",
    )
