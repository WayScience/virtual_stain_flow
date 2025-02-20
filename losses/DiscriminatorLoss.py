import torch

from .AbstractLoss import AbstractLoss

class DiscriminatorLoss(AbstractLoss):
    """
    This class implements the loss function for the discriminator in a Generative Adversarial Network (GAN). 
    The discriminator loss measures how well the discriminator is able to distinguish between real (ground truth) 
    images and fake (generated) images produced by the generator.
    """
    def __init__(self, _metric_name):
        super().__init__(_metric_name)

    def forward(self, truth, generated):
        """
        Computes the GaN discriminator loss given ground truth image and generated image

        :param truth: The tensor containing the ground truth image, 
            should be of shape [batch_size, channel_number, img_height, img_width].
        :type truth: torch.Tensor
        :param generated: The tensor containing model generated image, 
            should be of shape [batch_size, channel_number, img_height, img_width].
        :type generated: torch.Tensor
        :return: The computed metric as a float value.
        :rtype: float
        """
        
        # If the probability output is more than Scalar, take the mean of the output
        if truth.dim() >= 3:
            truth = torch.mean(truth, tuple(range(2, truth.dim())))
        if generated.dim() >= 3:
            generated = torch.mean(generated, tuple(range(2, generated.dim())))

        return (generated - truth).mean()