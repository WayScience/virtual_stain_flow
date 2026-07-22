import pytest
import torch
from torch.utils.data import Dataset

from virtual_stain_flow.evaluation.predict_utils import predict_image


class ImageDataset(Dataset):
    def __init__(self, inputs, targets=None):
        self.inputs = inputs
        self.targets = targets or [torch.zeros(1, 4, 4) for _ in inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return inputs


class AddInputsModel(torch.nn.Module):
    def forward(self, first_input, second_input):
        return first_input + second_input


class InvalidOutputModel(torch.nn.Module):
    def forward(self, inputs):
        return "not a tensor"


def test_predict_image_stacks_single_input_batches():
    inputs = [torch.full((1, 4, 4), float(index)) for index in range(3)]

    targets, predictions, stacked_inputs = predict_image(
        ImageDataset(inputs), IdentityModel(), batch_size=2, device="cpu"
    )

    assert targets.shape == (3, 1, 4, 4)
    assert torch.equal(predictions, torch.stack(inputs))
    assert torch.equal(stacked_inputs, torch.stack(inputs))


def test_predict_image_stacks_each_multi_input_component():
    first_inputs = [torch.ones(1, 4, 4), torch.full((1, 4, 4), 2.0)]
    second_inputs = [torch.full((1, 4, 4), 3.0), torch.full((1, 4, 4), 4.0)]
    dataset = ImageDataset(list(zip(first_inputs, second_inputs)))

    targets, predictions, stacked_inputs = predict_image(
        dataset, AddInputsModel(), batch_size=2, device="cpu"
    )

    assert targets.shape == (2, 1, 4, 4)
    assert isinstance(stacked_inputs, list)
    assert len(stacked_inputs) == 2
    assert torch.equal(stacked_inputs[0], torch.stack(first_inputs))
    assert torch.equal(stacked_inputs[1], torch.stack(second_inputs))
    assert torch.equal(predictions, torch.stack(first_inputs) + torch.stack(second_inputs))


def test_predict_image_rejects_empty_dataset():
    with pytest.raises(ValueError, match="dataset cannot be empty"):
        predict_image(ImageDataset([]), IdentityModel(), device="cpu")


def test_predict_image_rejects_empty_indices():
    dataset = ImageDataset([torch.zeros(1, 4, 4)])

    with pytest.raises(ValueError, match="indices cannot be empty"):
        predict_image(dataset, IdentityModel(), indices=[], device="cpu")


def test_predict_image_rejects_empty_multi_input_before_forward_pass():
    model = IdentityModel()
    dataset = ImageDataset([[]])

    with pytest.raises(ValueError, match="Multi-input batches cannot be empty"):
        predict_image(dataset, model, device="cpu")

    assert model.forward_calls == 0


def test_predict_image_rejects_non_tensor_input_before_forward_pass():
    model = IdentityModel()
    dataset = ImageDataset(["invalid input"])

    with pytest.raises(TypeError, match="Input batch 0 must be a torch.Tensor"):
        predict_image(dataset, model, device="cpu")

    assert model.forward_calls == 0


def test_predict_image_rejects_input_structure_change_before_second_forward_pass():
    model = IdentityModel()
    single_input = torch.ones(1, 4, 4)
    multi_input = (torch.ones(1, 4, 4), torch.ones(1, 4, 4))
    dataset = ImageDataset([single_input, multi_input])

    with pytest.raises(ValueError, match="single-input to multi-input"):
        predict_image(dataset, model, batch_size=1, device="cpu")

    assert model.forward_calls == 1


def test_predict_image_rejects_non_tensor_model_output():
    dataset = ImageDataset([torch.zeros(1, 4, 4)])

    with pytest.raises(TypeError, match="Model prediction must be a torch.Tensor"):
        predict_image(dataset, InvalidOutputModel(), device="cpu")
