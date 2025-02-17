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

    def forward(self, truth, generated):
        """
        Computes Gradient Penalty Loss for wGaN GP

        :param truth: The tensor containing the ground truth image, 
            should be of shape [batch_size, channel_number, img_height, img_width].
        :type truth: torch.Tensor
        :param generated: The tensor containing model generated image, 
            should be of shape [batch_size, channel_number, img_height, img_width].
        :type generated: torch.Tensor
        :return: The computed metric as a float value.
        :rtype: float
        """

        device = self.trainer.device

        batch_size = truth.size(0)
        eta = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(truth)
        interpolated = (eta * truth + (1 - eta) * generated).requires_grad_(True)
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