
import torch
import torch.autograd as autograd

from .AbstractLoss import AbstractLoss

class GradientPenaltyLoss(AbstractLoss):
    def __init__(self, _metric_name, discriminator, weight=10.0):
        super().__init__(_metric_name)

        ## TODO: add a wrapper class for GaN loss functions to 
        # dynamically access discriminator from the trainer class
        self._discriminator = discriminator
        self._weight = weight

    def forward(self, real_imgs, fake_imgs):

        device = self.trainer.device

        batch_size = real_imgs.size(0)
        ## TODO: check if expand_as behaves as expected
        eta = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_imgs)
        interpolated = (eta * real_imgs + (1 - eta) * fake_imgs).requires_grad_(True)
        prob_interpolated = self._discriminator(interpolated)

        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self._weight * gradient_penalty