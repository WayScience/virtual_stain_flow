import pandas as pd
import pytest
import torch

from virtual_stain_flow.evaluation.evaluation_utils import evaluate_per_image_metric


class MeanAbsoluteError(torch.nn.Module):
    def forward(self, target, prediction):
        return torch.abs(target - prediction).mean()


def test_evaluate_per_image_metric_returns_one_row_per_image():
    targets = torch.tensor([[[[1.0]]], [[[3.0]]]])
    predictions = torch.tensor([[[[2.0]]], [[[1.0]]]])

    result = evaluate_per_image_metric(predictions, targets, [MeanAbsoluteError()])

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"MeanAbsoluteError": [1.0, 2.0]}),
    )


def test_evaluate_per_image_metric_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluate_per_image_metric(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 3, 3), [])
