"""
wgan_losses.py

Wasserstein GAN loss implementations.
"""

import torch
from torch import Tensor

from .BaseLoss import BaseLoss


class WassersteinLoss(BaseLoss):
    """
    This class implements the loss function for the discriminator in a 
    Wasserstein Generative Adversarial Network (wGAN). 
    The discriminator loss measures how well the discriminator is able to 
        distinguish between real (ground expected_truth) 
        images and fake (expected_generated) images produced by the generator.
    """
    def __init__(self, _metric_name='WassersteinLoss'):
        super().__init__(_metric_name=_metric_name)

    def forward(
        self, 
        p_real_as_real: Tensor, 
        p_fake_as_real: Tensor
    ) -> Tensor:
        """
        Computes the Wasserstein Discriminator Loss given probability scores
        """
        
        # Ensure reduction of p tensors to [batch_size, 1]
        if p_real_as_real.dim() >= 3:
            p_real_as_real = torch.mean(p_real_as_real, tuple(range(2, p_real_as_real.dim())))
        if p_fake_as_real.dim() >= 3:
            p_fake_as_real = torch.mean(p_fake_as_real, tuple(range(2, p_fake_as_real.dim())))

        return (p_fake_as_real - p_real_as_real).mean()
    

class GradientPenaltyLoss(BaseLoss):
    """
    This class implements the gradient penalty loss for the discriminator in a 
    Wasserstein Generative Adversarial Network (wGAN-GP). 
    The gradient penalty is used to enforce the Lipschitz constraint on the discriminator,
    which helps stabilize the training of the GAN.
    """
    def __init__(self, _metric_name='GradientPenaltyLoss'):
        super().__init__(_metric_name=_metric_name)

    def forward(
        self, 
        real_stack: torch.Tensor, 
        fake_stack: torch.Tensor,
        discriminator: torch.nn.Module,
    ):
        """
        Computes the Gradient Penalty Loss given the gradients of the discriminator's output
        with respect to its input.
        """

        device = next(discriminator.parameters()).device
        batch_size = real_stack.size(0)
        eta = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_stack)
        interpolated = (eta * real_stack + (1 - eta) * fake_stack).requires_grad_(True)
        p_interpolated = discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=p_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(p_interpolated, device=device),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


class AdversarialLoss(BaseLoss):
    """
    Adversarial loss for the generator in a Wasserstein Generative Adversarial Network (wGAN).
    """

    def __init__(self, _metric_name='AdversarialLoss'):
        super().__init__(_metric_name=_metric_name)

    def forward(self, p_fake_as_real: torch.Tensor):
        """
        Computes the Adversarial Loss for the generator given the probability scores
        assigned by the discriminator to the fake (generated) images.
        """
        
        # Ensure reduction of p tensors to [batch_size, 1]
        if p_fake_as_real.dim() >= 3:
            p_fake_as_real = torch.mean(p_fake_as_real, tuple(range(2, p_fake_as_real.dim())))

        return -p_fake_as_real.mean()
