import torch

from .AbstractLoss import AbstractLoss

class DiscriminatorLoss(AbstractLoss):
    def __init__(self, _metric_name):
        super().__init__(_metric_name)

    def forward(self, real_output, fake_output):
        
        # If the probability output is more than Scalar, take the mean of the output
        if real_output.dim() >= 3:
            real_output = torch.mean(real_output, tuple(range(2, real_output.dim())))
        if fake_output.dim() >= 3:
            fake_output = torch.mean(fake_output, tuple(range(2, fake_output.dim())))

        return (fake_output - real_output).mean()