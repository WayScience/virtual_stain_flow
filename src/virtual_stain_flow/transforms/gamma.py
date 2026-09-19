"""Fixed, reversible gamma correction of normalized microscopy intensities."""

from numbers import Real

import numpy as np

from .base_transform import LoggableTransform


class ContinuousGammaTransform(LoggableTransform):
	"""
	Apply a continuous, strictly increasing power curve on a fixed interval.

	For ``lower < x < upper``, with ``t = (x - lower) / (upper - lower)``,
	the output is ``lower + (upper - lower) * t**gamma``. All other values
	are unchanged. The interval endpoints are fixed, making the mapping
	continuous (without jumps in pixel values). 

	``gamma < 1`` expands contrast near ``lower`` and compresses it near
	``upper``; ``gamma > 1`` does the reverse; ``gamma == 1`` is identity.
	No image statistics are estimated. The inverse uses ``1 / gamma`` on
	the same interval. Extreme powers and finite precision can lose detail.

	:param gamma: Positive, finite correction exponent.
	:param lower: Lower fixed point of the curve.
	:param upper: Upper fixed point of the curve.
	"""

	def __init__(
		self,
		gamma: float,
		lower: float = 0.0,
		upper: float = 1.0,
		name: str = "ContinuousGammaTransform",
		p: float = 1.0,
	):
		"""
		Initialize the continuous gamma transform with the given parameters.

		:param gamma: Positive, finite correction exponent.
		:param lower: Lower fixed point of the curve.
		:param upper: Upper fixed point of the curve.
		:param name: Name used for logging.
		:param p: Must be 1.0 for deterministic, reversible correction.
		"""
		gamma = self._finite_real(gamma, "gamma")
		lower = self._finite_real(lower, "lower")
		upper = self._finite_real(upper, "upper")
		p = self._finite_real(p, "p")
		if gamma <= 0 or not np.isfinite(1.0 / gamma):
			raise ValueError("gamma must be positive with a finite reciprocal.")
		if not 0.0 <= lower < upper <= 1.0:
			raise ValueError("Expected 0 <= lower < upper <= 1.")
		if p != 1.0:
			raise ValueError("p must be 1.0 for deterministic, reversible correction.")

		super().__init__(name=name, p=p)
		self._gamma = gamma
		self._lower = lower
		self._upper = upper

	@staticmethod
	def _finite_real(value: float, name: str) -> float:
		"""
		Ensure value is a finite real number.

		:param value: The value to check.
		:param name: Name of the parameter (for error messages).
		:return: The validated finite real number.
		"""
		if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
			raise TypeError(f"{name} must be a real number.")
		value = float(value)
		if not np.isfinite(value):
			raise ValueError(f"{name} must be finite.")
		return value

	@property
	def gamma(self) -> float:
		"""Power applied within the interval."""
		return self._gamma

	@property
	def lower(self) -> float:
		"""Lower fixed point of the curve."""
		return self._lower

	@property
	def upper(self) -> float:
		"""Upper fixed point of the curve."""
		return self._upper

	def apply(self, img: np.ndarray, **params) -> np.ndarray:
		"""Correct normalized intensities without clipping or estimating statistics."""
		if not isinstance(img, np.ndarray):
			raise TypeError("Expected input image to be a NumPy array.")
		if not np.issubdtype(img.dtype, np.floating):
			raise TypeError("Expected a floating-point image; apply MaxScaleNormalize first.")
		if not np.all(np.isfinite(img)):
			raise ValueError("Input image must contain only finite values.")
		if np.any(img < 0.0) or np.any(img > 1.0):
			raise ValueError("Input image must be normalized to [0, 1]; values are not clipped.")

		output = img.astype(np.result_type(img.dtype, np.float32), copy=True)
		if self.gamma == 1.0:
			return output

		# Compute only the open interval, preserving fixed points and all
		# unaffected values exactly; fractional powers never see negatives.
		work = img.astype(np.result_type(img.dtype, np.float64), copy=False)
		# Use representable endpoints in both directions. Otherwise a rounded
		# float32 lower endpoint can be mistaken for an interior value, whose
		# tiny offset is greatly amplified by a fractional power.
		lower = output.dtype.type(self.lower).item()
		upper = output.dtype.type(self.upper).item()
		active = (work > lower) & (work < upper)
		if not np.any(active):
			return output
		width = upper - lower
		t = (work[active] - lower) / width
		output[active] = lower + width * np.power(t, self.gamma)
		return output

	def inverse(self) -> "ContinuousGammaTransform":
		"""
		Return the deterministic analytic inverse, callable like any transform.

		This undoes gamma correction only. To undo preceding max scaling,
		multiply the recovered normalized image by its normalization factor.
		Values clipped during max scaling cannot be recovered.
		"""
		return type(self)(
			gamma=1.0 / self.gamma,
			lower=self.lower,
			upper=self.upper,
			name=f"{self.name}_inverse",
		)

	def __repr__(self) -> str:
		return (
			f"{self.__class__.__name__}(name={self.name}, gamma={self.gamma}, "
			f"lower={self.lower}, upper={self.upper}, p={self.p})"
		)

	def to_config(self) -> dict:
		"""Return JSON-serializable parameters for inherited from_config()."""
		return {
			"class": self.__class__.__name__,
			"name": self.name,
			"params": {
				"gamma": self.gamma,
				"lower": self.lower,
				"upper": self.upper,
				"p": self.p,
			},
		}
