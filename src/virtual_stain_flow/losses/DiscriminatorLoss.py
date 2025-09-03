import torch

from .AbstractLoss import AbstractLoss

class WassersteinLoss(AbstractLoss):
    """
    This class implements the loss function for the discriminator in a Wasserstein Generative Adversarial Network (wGAN). 
    The discriminator loss measures how well the discriminator is able to distinguish between real (ground expected_truth) 
    images and fake (expected_generated) images produced by the generator.
    """
    def __init__(self, _metric_name):
        super().__init__(_metric_name)

    def forward(self, expected_truth, expected_generated):
        """
        Computes the Wasserstein Discriminator Loss given probability scores expected_truth and expected_generated from the discriminator

        :param expected_truth: The tensor containing the ground expected_truth 
        probability score predicted by the discriminator over a batch of real images (input target pair),
        should be of shape [batch_size, 1].
        :type expected_truth: torch.Tensor
        :param expected_generated: The tensor containing model expected_generated 
        probability score predicted by the discriminator over a batch of generated images (input generated pair),
        should be of shape [batch_size, 1].
        :type expected_generated: torch.Tensor
        :return: The computed metric as a float value.
        :rtype: float
        """
        
        # If the probability output is more than Scalar, take the mean of the output
        # For compatibility with both a Discriminator class that would output a scalar probability (currently implemented)
        # and a Discriminator class that would output a 2d matrix of probabilities (currently not implemented) 
        if expected_truth.dim() >= 3:
            expected_truth = torch.mean(expected_truth, tuple(range(2, expected_truth.dim())))
        if expected_generated.dim() >= 3:
            expected_generated = torch.mean(expected_generated, tuple(range(2, expected_generated.dim())))

        return (expected_generated - expected_truth).mean()