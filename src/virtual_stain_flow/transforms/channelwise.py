"""
channelwise.py

Defines channel-specific transform wrappers.
Useful if the values of different channels take different ranges or distributions
and needs separate transformations/normalizations for each channel.
"""

from typing import List, Optional, Sequence

import numpy as np

from .base_transform import LoggableTransform


class ChannelwiseTransform(LoggableTransform):
	"""
	Apply a list of transforms to a channel-first image, one transform per channel.

	A potential use case of this transform would be two modal input combining 
		two channels of different images (image augmentations). 
		An example would be brightfield image + phase retrieved image from brightfield z-stacks.
		The former would take the range of a grayscale image, while the latter
		would be in unit of radians for phase delay, centered around zero. 
		A reasonable channelwise transform would be a bitdepth max normalization
		applied to the brightfield image channel and a z-score/radian normalization
		applied to the phase retrieved image channel.	
	"""

	def __init__(
		self,
		transforms: Sequence[Optional[LoggableTransform]],
		name: str = "ChannelwiseTransform",
		p: float = 1.0,
		channel_axis: int = 0,
	):
		super().__init__(name=name, p=p)

		if channel_axis != 0:
			raise ValueError("Only channel-first images with channel_axis=0 are supported.")

		if not isinstance(transforms, Sequence) or len(transforms) == 0:
			raise ValueError("Expected a non-empty sequence of LoggableTransform or None.")
		if not all((t is None) or isinstance(t, LoggableTransform) for t in transforms):
			raise ValueError("All transforms must be instances of LoggableTransform or None.")

		self._transforms: List[Optional[LoggableTransform]] = list(transforms)
		self._channel_axis = channel_axis

	@property
	def transforms(self) -> List[Optional[LoggableTransform]]:
		return self._transforms

	@property
	def channel_axis(self) -> int:
		return self._channel_axis

	def apply(self, img: np.ndarray, **params) -> np.ndarray:
		if not isinstance(img, np.ndarray):
			raise TypeError(
				"Expected input image to be a NumPy array, "
				f"got {type(img).__name__} instead."
			)

		if img.ndim != 3:
			raise ValueError(
				"Expected a channel-first image with shape (C, H, W)."
			)

		channel_count = img.shape[self._channel_axis]
		if channel_count != len(self._transforms):
			raise ValueError(
				"Number of transforms must match number of channels. "
				f"Got {len(self._transforms)} transforms for {channel_count} channels."
			)

		transformed_channels = []
		for channel_idx, transform in enumerate(self._transforms):
			channel = img[channel_idx:channel_idx + 1, ...]
			if transform is None:
				transformed_channels.append(channel)
			else:
				transformed_channels.append(transform.apply(img=channel))

		return np.concatenate(transformed_channels, axis=0)

	def __repr__(self) -> str:
		return (
			f"{self.__class__.__name__}(name={self._name}, "
			f"channels={len(self._transforms)}, p={self.p})"
		)

	def to_config(self) -> dict:
		"""
		Returns a dictionary containing the configuration of the transform.

		Helps with training reproducibility by allowing the logger to export
			the configuration of the transform for later use.
		"""
		return {
			"class": self.__class__.__name__,
			"name": self._name,
			"params": {
				"channel_axis": self._channel_axis,
				"p": self.p,
				"transforms": [t.to_config() if t is not None else None for t in self._transforms],
			},
		}
