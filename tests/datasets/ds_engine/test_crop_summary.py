"""
Minimal tests for crop summary formatting helpers.
"""

import pytest

from virtual_stain_flow.datasets.ds_engine.crop_generators.crop_summary import (
    CropSummaryWarning,
    build_stats_table,
    format_crop_summary,
    warn_formatted_crop_summary,
)


def test_build_stats_table_renders_headers_and_rows():
    """Should render a compact ASCII table from metric key-value pairs."""
    table = build_stats_table({"Total accepted": 12, "Total rejected": 3})

    assert "Metric" in table
    assert "Value" in table
    assert "Total accepted" in table
    assert "Total rejected" in table


def test_warn_formatted_crop_summary_emits_warning_with_formatted_body():
    """Should emit CropSummaryWarning containing title, detail, and table rows."""
    with pytest.warns(CropSummaryWarning, match="Example summary") as record:
        warn_formatted_crop_summary(
            title="Example summary",
            detail_line="Example criterion line.",
            metrics={"Total": 2, "Mean": 1.0},
        )

    message = str(record[0].message)
    assert "Example criterion line." in message
    assert "Total" in message
    assert "Mean" in message


def test_format_crop_summary_without_detail_still_includes_table():
    """Should format title and table when detail line is omitted."""
    summary = format_crop_summary(
        title="No detail summary",
        metrics={"Success": "3/3"},
    )

    assert "No detail summary" in summary
    assert "Success" in summary
