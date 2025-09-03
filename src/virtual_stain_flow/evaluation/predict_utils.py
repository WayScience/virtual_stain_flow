from typing import Optional, List, Tuple, Callable

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from albumentations import ImageOnlyTransform, Compose

def predict_image(
    dataset: Dataset,
    model: torch.nn.Module,
    batch_size: int = 1,
    device: str = "cpu",
    num_workers: int = 0,
    indices: Optional[List[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs a model on a dataset, performing a forward pass on all (or a subset of) input images 
    in evaluation mode and returning a stacked tensor of predictions.
    DOES NOT check if the dataset dimensions are compatible with the model. 

    :param dataset: A dataset that returns (input_tensor, target_tensor) tuples, 
                    where input_tensor has shape (C, H, W).
    :type dataset: torch.utils.data.Dataset
    :param model: A PyTorch model that is compatible with the dataset inputs.
    :type model: torch.nn.Module
    :param batch_size: The number of samples per batch (default is 1).
    :type batch_size: int, optional
    :param device: The device to run inference on, e.g., "cpu" or "cuda".
    :type device: str, optional
    :param num_workers: Number of workers for the DataLoader (default is 0).
    :type num_workers: int, optional
    :param indices: Optional list of dataset indices to subset the dataset before inference.
    :type indices: Optional[List[int]], optional

    :return: Tuple of stacked target and prediction tensors.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    # Subset the dataset if indices are provided
    if indices is not None:
        dataset = Subset(dataset, indices)

    # Create DataLoader for efficient batch processing
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.to(device)
    model.eval()

    predictions, targets = [], []

    with torch.no_grad():
        for inputs, target in dataloader:  # Unpacking (input_tensor, target_tensor)
            inputs = inputs.to(device)  # Move input data to the specified device

            # Forward pass
            outputs = model(inputs)
            
            # output both target and prediction tensors for metric
            targets.append(target.cpu())
            predictions.append(outputs.cpu())  # Move to CPU for stacking

    return torch.cat(targets, dim=0), torch.cat(predictions, dim=0) 

def process_tensor_image(
    img_tensor: torch.Tensor,
    dtype: Optional[np.dtype] = None,
    dataset: Optional[Dataset] = None,
    invert_function: Optional[Callable] = None
) -> np.ndarray:
    """
    Processes model output/other image tensor by casting to numpy, applying an optional dtype casting, 
    and inverting target transformations if a dataset with `target_transform` is provided.

    :param img_tensor: Tensor stack of model-predicted images with shape (N, C, H, W).
    :type img_tensor: torch.Tensor
    :param dtype: Optional numpy dtype to cast the output array (default: None).
    :type dtype: Optional[np.dtype], optional
    :param dataset: Optional dataset object with `target_transform` to invert transformations.
    :type dataset: Optional[torch.utils.data.Dataset], optional
    :param invert_function: Optional function to invert transformations applied to the images.
        If provided, overrides the invert function call from dataset transform.
    :type invert_function: Optional[Callable], optional

    :return: Processed numpy array of images with shape (N, C, H, W).
    :rtype: np.ndarray
    """
    # Convert img_tensor to CPU and NumPy
    output_images = img_tensor.cpu().numpy()

    # Optionally cast to specified dtype
    if dtype is not None:
        output_images = output_images.astype(dtype)

    # Apply invert function when supplied or transformation if invert function is supplied
    if invert_function is not None and isinstance(invert_function, Callable):
        output_images = np.array([invert_function(img) for img in output_images])
    elif dataset is not None and hasattr(dataset, "target_transform"):
        # Apply inverted target transformation if available
        target_transform = dataset.target_transform
        if isinstance(target_transform, (ImageOnlyTransform, Compose)):
            # Apply the transformation on each image
            output_images = np.array([target_transform.invert(img) for img in output_images])

    return output_images