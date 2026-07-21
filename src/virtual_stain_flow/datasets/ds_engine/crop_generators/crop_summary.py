"""
crop_summary.py

Helper module for formatting crop generation summaries, 
shared by crop generating modules
"""

from collections.abc import Mapping
import warnings


class CropSummaryWarning(UserWarning):
	"""Warning category for crop generation summary reports."""


def _stringify_metric_value(value: object) -> str:
	"""Convert metric values to compact, human-readable strings."""
	if isinstance(value, float):
		return f"{value:.2f}"
	return str(value)


def build_stats_table(metrics: Mapping[str, object]) -> str:
	"""
	Build a simple ASCII table from metric-value key-value pairs.

	:param metrics: Ordered metric mapping of label -> value.
	:return: Multi-line ASCII table.
	"""
	metric_header = "Metric"
	value_header = "Value"

	rows = [(metric, _stringify_metric_value(value)) for metric, value in metrics.items()]
	if not rows:
		rows = [("No metrics", "n/a")]

	metric_width = max(len(metric_header), *(len(metric) for metric, _ in rows))
	value_width = max(len(value_header), *(len(value) for _, value in rows))

	separator = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
	header = f"| {metric_header:<{metric_width}} | {value_header:<{value_width}} |"
	body = [
		f"| {metric:<{metric_width}} | {value:>{value_width}} |"
		for metric, value in rows
	]

	return "\n".join([separator, header, separator, *body, separator])


def format_crop_summary(
	title: str,
	metrics: Mapping[str, object],
	detail_line: str | None = None,
) -> str:
	"""
	Format a crop summary title, optional detail line, and metric table.

	:param title: Summary title line.
	:param metrics: Ordered metric mapping of label -> value.
	:param detail_line: Optional descriptive one-liner for context.
	:return: Formatted summary text.
	"""
	stats_table = build_stats_table(metrics)
	if detail_line:
		return f"{title}\n{detail_line}\n{stats_table}"
	return f"{title}\n{stats_table}"


def warn_formatted_crop_summary(
	title: str,
	metrics: Mapping[str, object],
	detail_line: str | None = None,
) -> None:
	"""
	Emit a module-scoped crop summary warning without file/line prefixes.

	:param title: Summary title line.
	:param metrics: Ordered metric mapping of label -> value.
	:param detail_line: Optional descriptive one-liner for context.
	"""
	message = format_crop_summary(title=title, metrics=metrics, detail_line=detail_line)

	original_formatwarning = warnings.formatwarning
	try:
		warnings.formatwarning = (
			lambda msg, category, filename, lineno, line=None:
			f"{category.__name__}: {msg}\n"
		)
		warnings.warn(message, category=CropSummaryWarning, stacklevel=2)
	finally:
		warnings.formatwarning = original_formatwarning
