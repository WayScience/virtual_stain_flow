from typing import Optional

import torch
from torch.nn import L1Loss

from .AbstractLoss import AbstractLoss

class GeneratorLoss(AbstractLoss):    
    """
    Computes the loss for the GaN generator. 
    Combines an adversarial loss component with an image reconstruction loss.
    """
    def __init__(self, 
                 _metric_name: str, 
                reconstruction_loss: Optional[torch.tensor] = L1Loss()
                ):
        """
        :param reconstruction_loss: The image reconstruction loss, 
        defaults to L1Loss(reduce=False)
        :type reconstruction_loss: torch.tensor
        """
        
        super().__init__(_metric_name)

        self._reconstruction_loss = reconstruction_loss

    def forward(self, 
                discriminator_probs: torch.tensor,
                truth: torch.tensor, 
                generated: torch.tensor,
                epoch: int = 0                
                ):
        """
        Computes the loss for the GaN generator.

        :param discriminator_probs: The probabilities of the discriminator for the fake images being real.
        :type discriminator_probs: torch.tensor
        :param truth: The tensor containing the ground truth image, 
            should be of shape [batch_size, channel_number, img_height, img_width].
        :type truth: torch.Tensor
        :param generated: The tensor containing model generated image, 
            should be of shape [batch_size, channel_number, img_height, img_width].
        :type generated: torch.Tensor
        :param epoch: The current epoch number. 
        Used for a smoothing weight for the adversarial loss component
        Defaults to 0.
        :type epoch: int
        :return: The computed metric as a float value.
        :rtype: float
        """

        # Adversarial loss
        adversarial_loss = -torch.mean(discriminator_probs)
        
        adversarial_loss = 0.01 * adversarial_loss/(epoch + 1)

        image_loss = self._reconstruction_loss(generated, truth)
        
        return adversarial_loss + image_loss.mean()