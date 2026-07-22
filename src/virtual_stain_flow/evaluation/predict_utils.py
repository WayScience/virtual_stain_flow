from typing import Any, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset


def _move_to_device(value: Any, device: Union[str, torch.device]) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(item, device) for item in value)
    return value


def _validate_image_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, received {type(value).__name__}.")
    if value.ndim != 4:
        raise ValueError(
            f"{name} must have shape (N, C, H, W), received {tuple(value.shape)}."
        )
    if value.shape[0] == 0:
        raise ValueError(f"{name} cannot have an empty batch dimension.")
    return value


def _validate_batch(
    input_batch: Any,
    target_batch: Any,
    expected_multi_input: Optional[bool],
    expected_input_count: Optional[int],
) -> Tuple[bool, int]:
    _validate_image_tensor(target_batch, "Target batch")

    if isinstance(input_batch, (list, tuple)):
        if not input_batch:
            raise ValueError("Multi-input batches cannot be empty.")
        if expected_multi_input is False:
            raise ValueError("Input structure changed from single-input to multi-input during prediction.")
        if expected_input_count is not None and expected_input_count != len(input_batch):
            raise ValueError("Multi-input batch count changed during prediction.")

        for input_index, input_tensor in enumerate(input_batch):
            _validate_image_tensor(input_tensor, f"Input batch {input_index}")
            if input_tensor.shape[0] != target_batch.shape[0]:
                raise ValueError("Input and target batch sizes must match.")
        return True, len(input_batch)

    _validate_image_tensor(input_batch, "Input batch")
    if input_batch.shape[0] != target_batch.shape[0]:
        raise ValueError("Input and target batch sizes must match.")
    if expected_multi_input is True:
        raise ValueError("Input structure changed from multi-input to single-input during prediction.")
    return False, 1


def predict_image(
    dataset: Dataset,
    model: torch.nn.Module,
    batch_size: int = 1,
    device: Union[str, torch.device] = "cpu",
    num_workers: int = 0,
    indices: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Runs a model on a dataset, performing a forward pass on all (or a subset of) input images 
    in evaluation mode and returning a stacked tensor of predictions.
    DOES NOT check if the dataset dimensions are compatible with the model. 

    :param dataset: A dataset that returns (input_tensor, target_tensor) tuples, 
                    where input_tensor has shape (C, H, W).
    :param model: A PyTorch model that is compatible with the dataset inputs.
    :param batch_size: The number of samples per batch (default is 1).
    :param device: The device to run inference on, e.g., "cpu" or "cuda".
    :param num_workers: Number of workers for the DataLoader (default is 0).
    :param indices: Optional list of dataset indices to subset the dataset before inference.

    :return: Tuple of stacked target, prediction, and input tensors. For multi-input
        datasets, the third element is a list of stacked input tensors.
    :raises ValueError: If the selected dataset is empty, a batch is malformed, or
        its input structure changes during prediction.
    :raises TypeError: If inputs, targets, or model predictions are not tensors.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    # Subset the dataset if indices are provided
    if indices is not None:
        if not indices:
            raise ValueError("indices cannot be empty.")
        if min(indices) < 0 or max(indices) >= len(dataset):
            raise IndexError(
                f"Index out of range. Dataset length: {len(dataset)}, "
                f"requested indices: {indices}"
            )
        dataset = Subset(dataset, indices)
    elif len(dataset) == 0:
        raise ValueError("dataset cannot be empty.")

    # Create DataLoader for efficient batch processing
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.to(device)
    model.eval()

    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    input_batches: List[torch.Tensor] = []
    multi_input_batches: Optional[List[List[torch.Tensor]]] = None
    expected_multi_input: Optional[bool] = None
    expected_input_count: Optional[int] = None

    with torch.no_grad():
        for input_batch, target_batch in dataloader:
            is_multi_input, input_count = _validate_batch(
                input_batch, target_batch, expected_multi_input, expected_input_count
            )
            expected_multi_input = is_multi_input
            if is_multi_input:
                expected_input_count = input_count

            input_batch = _move_to_device(input_batch, device)

            # Forward pass
            if is_multi_input:
                prediction = model(*input_batch)
            else:
                prediction = model(input_batch)
            _validate_image_tensor(prediction, "Model prediction")
            if prediction.shape[0] != target_batch.shape[0]:
                raise ValueError("Prediction and target batch sizes must match.")

            targets.append(target_batch.cpu())
            predictions.append(prediction.cpu())

            if is_multi_input:
                if multi_input_batches is None:
                    multi_input_batches = [[] for _ in range(input_count)]
                for input_index, input_tensor in enumerate(input_batch):
                    multi_input_batches[input_index].append(input_tensor.cpu())
            else:
                input_batches.append(input_batch.cpu())

    if not targets or not predictions:
        raise RuntimeError("Prediction did not retrieve any batches.")

    if multi_input_batches is not None:
        inputs_stacked = [torch.cat(batch, dim=0) for batch in multi_input_batches]
    else:
        if not input_batches:
            raise RuntimeError("Prediction did not accumulate any input batches.")
        inputs_stacked = torch.cat(input_batches, dim=0)

    return (
        torch.cat(targets, dim=0),
        torch.cat(predictions, dim=0),
        inputs_stacked,
    )
