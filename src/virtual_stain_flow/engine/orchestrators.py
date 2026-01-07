"""
orchestrators.py

Collection of orchestrator classes that manages training flow
for complex models involving multiple components, such as GANs.

This is constrasted with ForwardGroup classes, which handle
the forward pass and optimization of single model components.
The addition of orchestrators helps keep ForwardGroup classes simple.

Internally, an orchestrator manages multiple ForwardGroups
and defines coordinated training steps that involve forward passes
through several components in a specific sequence.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor
from torch import optim

from .forward_groups import GeneratorForwardGroup, DiscriminatorForwardGroup
from .context import Context


@dataclass
class OrchestratedStep:
    """
    Thin wrapper around orchestrator methods to present step-like objects
    to trainers with the same interface as ForwardGroups, exposing:
      - __call__(train=..., **batch) for forward pass, and
      - .step() to step the optimizer
    """

    forward_fn: Callable[..., Context]
    optimizer: Optional[optim.Optimizer] = None

    def __call__(self, train: bool, **batch) -> Context:
        return self.forward_fn(train=train, **batch)

    def step(self) -> None:
        if self.optimizer is not None:
            self.optimizer.step()
        

class GANOrchestrator:
    """
    Orchestrator for a GAN-style setup with separate generator and discriminator
    training steps.

    Stores GeneratorForwardGroup and a DiscriminatorForwardGroup:    
    The GeneratorForwardGroup and DiscriminatorForwardGroup are the 
        simplified building blocks that conducts exclusively the forward pass of 
        either generator or discriminator). 
    The Orchestrator._discriminator_forward and Orchestrator._generator_forward 
        methods is uses these simple forward groups to build more complex steps that
        enable GAN training, which requires a coordinated forward pass through both
        the discriminator and generator.
    """

    def __init__(
        self,
        generator_fg: GeneratorForwardGroup,
        discriminator_fg: DiscriminatorForwardGroup,
    ):
        """
        Initialize from already-constructed forward groups.

        This keeps concerns separated: forward groups manage single-module
        behavior; the orchestrator manages their composition.
        """
        # simple forward group storage
        if not isinstance(generator_fg, GeneratorForwardGroup):
            raise TypeError("generator_fg must be a GeneratorForwardGroup")
        self._gen_fg: GeneratorForwardGroup = generator_fg
        if not isinstance(discriminator_fg, DiscriminatorForwardGroup):
            raise TypeError("discriminator_fg must be a DiscriminatorForwardGroup")
        self._disc_fg: DiscriminatorForwardGroup = discriminator_fg

        # Public step-like objects that trainers can use directly
        self.discriminator_step = OrchestratedStep(
            forward_fn=self._discriminator_forward,
            optimizer=self._disc_fg.optimizer,
        )
        self.generator_step = OrchestratedStep(
            forward_fn=self._generator_forward,
            optimizer=self._gen_fg.optimizer,
        )
        
    def _build_real_fake_contexts(
        self,
        train: bool,
        gen_ctx: Context,
    ) -> Context:
        """
        Given a generator context containing inputs / targets / preds,
            generates the real and fake stacks by concatenating the true
            input with the true target or predicted target along the 
            channel dimension. The result stacks serve as direct inputs
            to the discriminator.
            
        The discriminator is then run on both stacks to produce
            outputs scores of if it thinks the provided stack is real.

        :param train: Whether the model is in training mode.
        :param gen_ctx: The Context produced by the generator forward pass,
            containing at least INPUTS, TARGETS, and PREDS tensors.
        :return: A merged Context containing outputs from both
            the real and fake discriminator passes, as well as
            the original generator context.
        """
        # Stack along channel dim: [inputs, targets] vs [inputs, preds]
        # Context objects handles type checking before the reserved key
        # context values (inputs, targets, preds) are tenors, so no
        # further type checking is needed here.
        real_stack: Tensor = torch.cat(tensors=[gen_ctx.inputs, gen_ctx.targets], dim=1)
        fake_stack: Tensor = torch.cat(tensors=[gen_ctx.inputs, gen_ctx.preds], dim=1)

        # Real batch: D(x, y_true)
        ctx_real: Context = self._disc_fg(train=train, stack=real_stack)
        ctx_real["real_stack"] = ctx_real.pop(key="stack")
        ctx_real["p_real_as_real"] = ctx_real.pop(key="p")

        # Fake batch: D(x, y_fake)
        ctx_fake: Context = self._disc_fg(train=train, stack=fake_stack)
        ctx_fake["fake_stack"] = ctx_fake.pop(key="stack")
        ctx_fake["p_fake_as_real"] = ctx_fake.pop(key="p")

        # Merge: real info, fake info, and generator info
        return ctx_real | ctx_fake | gen_ctx

    def _discriminator_forward(
        self,
        train: bool,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Context:
        """
        Forward step to train only the discriminator.

        :param train: Whether the model is in training mode.
        :param inputs: The input tensor for the models.
        :param targets: The target tensor for the models.
        :return: A Context containing discriminator outputs for both
            real and fake stacks, as well as the original generator context.    
        """
        # Generator is always eval for discriminator updates
        gen_ctx: Context = self._gen_fg(train=False, inputs=inputs, targets=targets)
        
        return self._build_real_fake_contexts(train=train, gen_ctx=gen_ctx)

    def _generator_forward(
        self,
        train: bool,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Context:
        """
        Forward step to train only the generator.

        :param train: Whether the model is in training mode.
        :param inputs: The input tensor for the models.
        :param targets: The target tensor for the models.
        :return: A Context containing generator outputs plus
            p_fake_as_real from the discriminator.
        """

        # Generate predictions and then run discriminator on fake stack
        gen_ctx: Context = self._gen_fg(train=train,inputs=inputs,targets=targets)
        fake_stack: Tensor = torch.cat(tensors=[gen_ctx.inputs, gen_ctx.preds], dim=1)
        disc_ctx: Context = self._disc_fg(train=train, stack=fake_stack)

        # Attach discriminator score to the generator context and return.
        return gen_ctx.add(p_fake_as_real=disc_ctx["p"])

    @property
    def generator_forward_group(self) -> GeneratorForwardGroup:
        return self._gen_fg

    @property
    def discriminator_forward_group(self) -> DiscriminatorForwardGroup:
        return self._disc_fg
