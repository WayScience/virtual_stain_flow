"""
Logging GAN Trainer

This module defines the LoggingGANTrainer class, which extends the
AbstractTrainer to provide training and evaluation functionalities for a GAN
model using the engine subpackage for forward passes and loss computations.
"""

import pathlib
from typing import Dict, List, Union, Optional

import torch

from .AbstractTrainer import AbstractTrainer
from ..engine.loss_group import LossGroup, LossItem
from ..losses.wgan_losses import (
    WassersteinLoss,
    GradientPenaltyLoss,
    AdversarialLoss
)
from ..engine.forward_groups import GeneratorForwardGroup, DiscriminatorForwardGroup
from ..engine.orchestrators import GANOrchestrator

Scalar = Union[int, float]


class BaseGANTrainer(AbstractTrainer):
    """
    Flexible trainer class for GAN models with logging.
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        generator_loss_group: LossGroup,
        discriminator_loss_group: LossGroup,
        n_discriminator_steps: int = 3,
        **kwargs
    ):
        """
        Initialize the trainer with the GAN orchestrator and loss groups.
        
        :param generator: The generator model to be trained.
        :param discriminator: The discriminator model to be trained.
        :param generator_optimizer: The optimizer for the generator.
        :param discriminator_optimizer: The optimizer for the discriminator.
        :param generator_loss_group: The loss group for the generator.
        :param discriminator_loss_group: The loss group for the discriminator.
        :param n_discriminator_steps: Number of discriminator steps per generator step.
        :kwargs: Additional arguments for the AbstractTrainer
        """

        device = kwargs.get('device', torch.device('cpu'))

        # Registry for logging model parameters
        self._models: List[torch.nn.Module] = [generator, discriminator]

        generator_fg = GeneratorForwardGroup(
            generator=generator,
            optimizer=generator_optimizer,
            device=device
        )

        discriminator_fg = DiscriminatorForwardGroup(
            discriminator=discriminator,
            optimizer=discriminator_optimizer,
            device=device
        )

        self._orchestrator = GANOrchestrator(
            generator_fg=generator_fg,
            discriminator_fg=discriminator_fg
        )
        
        self._generator_loss_group: LossGroup = generator_loss_group
        self._discriminator_loss_group: LossGroup = discriminator_loss_group

        # Internal counters for update frequencies
        self._n_discriminator_steps: int = n_discriminator_steps
        self._global_step: int = 0

        super().__init__(
            model=generator, # register generator as main model for early stopping
            optimizer=generator_optimizer,
            losses=generator_loss_group,
            **kwargs
        )

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step for both the generator and discriminator.

        :param inputs: The input tensor for the models.
        :param targets: The target tensor for the models.
        :return: A dictionary containing the loss values for both generator and discriminator.
        """
        # Always update discriminator
        #disc_ctx = self._discriminator_forward_group(
        disc_ctx = self._orchestrator.discriminator_step(
            train=True,
            inputs=inputs,
            targets=targets
        )
        disc_weighted_total, disc_logs = self._discriminator_loss_group(
            train=True,
            context=disc_ctx
        )
        disc_weighted_total.backward()
        #self._discriminator_forward_group.step()
        self._orchestrator.discriminator_step.step()

        # always evaluate metrics on discriminator context
        # which will always represent the most updated generator state
        ctx = disc_ctx
        for _, metric in self.metrics.items():
            metric.update(*ctx.as_metric_args(), validation=True)

        # Update generator 1 in every n_discriminator_steps
        if (self._global_step % self._n_discriminator_steps) == 0:
            
            # Generator step
            gen_ctx = self._orchestrator.generator_step(
                train=True,
                inputs=inputs,
                targets=targets
            )
            gen_weighted_total, gen_logs = self._generator_loss_group(
                train=True,
                context=gen_ctx
            )
            gen_weighted_total.backward()
            self._orchestrator.generator_step.step()
        else:
            gen_ctx = None
            gen_logs = {}
        
        self._global_step += 1

        # if generator logs are not computed this step (due to skipped update), 
        # compute from discriminator context
        if not gen_logs:
            _, gen_logs = self._generator_loss_group(
                train=True,
                context=ctx
            )

        return gen_logs | disc_logs

    def evaluate_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single evaluation step for both the generator and discriminator.

        :param inputs: The input tensor for the models.
        :param targets: The target tensor for the models.
        :return: A dictionary containing the loss values for both generator and discriminator.
        """
        
        # Taking a shortcut here by only evaluating through
        # the discriminator forward group, which will also
        # contain the generator outputs
        ctx = self._orchestrator.discriminator_step(
            train=False,
            inputs=inputs,
            targets=targets
        )
        _, gen_logs = self._generator_loss_group(
            train=False,
            context=ctx
        )
        _, disc_logs = self._discriminator_loss_group(
            train=False,
            context=ctx
        )

        for _, metric in self.metrics.items():
            metric.update(*ctx.as_metric_args(), validation=True)

        return gen_logs | disc_logs
    
    def save_model(
        self,
        save_path: pathlib.Path,
        file_name_prefix: Optional[str] = None,
        file_name_suffix: Optional[str] = None,
        file_ext: str = '.pth',
        best_model: bool = True
    ) -> Optional[List[pathlib.Path]]:

        if file_name_suffix is None:
            file_name_suffix = 'weights_' + (
                'best' if best_model else str(self.epoch)
            )

        gen_path = self.model.save_weights(
            filename=f"generator_{file_name_suffix}{file_ext}",
            dir=save_path
        )

        return [gen_path]

class LoggingWGANTrainer(BaseGANTrainer):
    """
    Predefined WGAN trainer needing only drop-in generator losses and weights.

    Under default settings, this trainer:

    trains the generator with:
    - AdverserialLoss() operating on p_fake_as_real from the discriminator
    - + [any additional provided generator losses]

    trains the discriminator with:
    - WassersteinLoss() operating on p_real_as_real and p_fake_as_real
    - GradientPenaltyLoss() operating on real_stack, fake_stack, and discriminator
    """

    def __init__(
        self,
        *,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        generator_losses: Union[torch.nn.Module, List[torch.nn.Module]],
        generator_loss_weights: Optional[Union[Scalar, List[Scalar]]] = None,
        generator_adverserial_loss: torch.nn.Module = AdversarialLoss(),
        generator_adverserial_loss_weight: Scalar = 1.0,
        discriminator_loss: torch.nn.Module = WassersteinLoss(),
        discriminator_loss_weight: Scalar = 1.0,
        discriminator_gradient_penalty_loss: torch.nn.Module = GradientPenaltyLoss(),
        discriminator_gradient_penalty_loss_weight: Scalar = 10.0,
        n_discriminator_steps: int = 3,
        **kwargs
    ):
        """
        Initialize the WGAN trainer with the GAN orchestrator and loss groups.
        
        :param generator: The generator model to be trained.
        :param discriminator: The discriminator model to be trained.
        :param generator_optimizer: The optimizer for the generator.
        :param discriminator_optimizer: The optimizer for the discriminator.
        :param generator_losses: The loss function(s) for the generator.
        :param generator_loss_weights: The weight(s) for the generator loss function(s).
        :param discriminator_loss: The loss function for the discriminator.
        :param discriminator_loss_weight: The weight for the discriminator loss function.
        :param discriminator_gradient_penalty_loss: The gradient penalty loss function for the discriminator.
        :param discriminator_loss_weight: The weight for the discriminator loss function.
        :param discriminator_gradient_penalty_loss: The gradient penalty loss function for the discriminator.
        :param discriminator_gradient_penalty_loss_weight: The weight for the gradient penalty loss function.
        :param n_discriminator_steps: Number of discriminator steps per generator step.
        :kwargs: Additional arguments for the AbstractTrainer
        """

        device = kwargs.get('device', torch.device('cpu'))

        generator_losses = generator_losses if isinstance(
            generator_losses,
            list
        ) else [generator_losses]
        if generator_loss_weights is None:
            generator_loss_weights = [1.0] * len(generator_losses)
        elif isinstance(generator_loss_weights, Scalar):
            generator_loss_weights = [generator_loss_weights] * len(generator_losses)
        elif isinstance(generator_loss_weights, list):
            if len(generator_loss_weights) != len(generator_losses):
                raise ValueError(
                    "Length of generator_loss_weights must match length of generator_losses."
                )
        else:
            raise TypeError(
                "generator_loss_weights must be a float or list of floats."
            )

        generator_loss_group = LossGroup(
            items=[
                LossItem(
                    module=loss,
                    weight=weight,
                    args=('preds', 'targets'),
                    device=device
                )
                for loss, weight in zip(
                    generator_losses,
                    generator_loss_weights
                )
            ] + [
                LossItem(
                    module=generator_adverserial_loss,
                    weight=generator_adverserial_loss_weight,
                    args=('p_fake_as_real',),
                    device=device
                )
            ]
        )

        discriminator_loss_group = LossGroup(
            items=[
                LossItem(
                    module=discriminator_loss,
                    weight=discriminator_loss_weight,
                    args=('p_real_as_real', 'p_fake_as_real'),
                    device=device
                ),
                LossItem(
                    module=discriminator_gradient_penalty_loss,
                    weight=discriminator_gradient_penalty_loss_weight,
                    args=('real_stack', 'fake_stack', 'discriminator'),
                    device=device
                )
            ]
        )

        super().__init__(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_loss_group=generator_loss_group,
            discriminator_loss_group=discriminator_loss_group,
            n_discriminator_steps=n_discriminator_steps,
            **kwargs
        )
