from typing import Optional
from collections import defaultdict

import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader

from .AbstractTrainer import AbstractTrainer

class WGaNTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: torch.nn.Module, 
                 discriminator: torch.nn.Module, 
                 gen_optimizer: torch.optim.Optimizer, 
                 disc_optimizer: torch.optim.Optimizer, 
                 generator_loss_fn: torch.nn.Module, 
                 discriminator_loss_fn: torch.nn.Module, 
                 gradient_penalty_fn: Optional[torch.nn.Module]=None, 
                 discriminator_update_freq: int=1,
                 generator_update_freq: int=5,
                 # rest of the arguments are passed to and handled by the parent class
                    # - dataset
                    # - batch_size
                    # - epochs
                    # - patience
                    # - callbacks
                    # - metrics
                 **kwargs):
        """
        Initializes the WGaN Trainer class.

        :param generator: The image2image generator model (e.g., UNet)
        :type generator: torch.nn.Module
        :param discriminator: The discriminator model
        :type discriminator: torch.nn.Module
        :param gen_optimizer: Generator optimizer
        :type gen_optimizer: torch.optim.Optimizer
        :param disc_optimizer: Discriminator optimizer
        :type disc_optimizer: torch.optim.Optimizer
        :param generator_loss_fn: Generator loss function
        :type generator_loss_fn: torch.nn.Module
        :param discriminator_loss_fn: Adverserial loss function
        :type discriminator_loss_fn: torch.nn.Module
        :param gradient_penalty_fn: (optional) Gradient penalty loss function
        :type gradient_penalty_fn: torch.nn.Module
        :param discriminator_update_freq: How frequently to update the discriminator
        :type discriminator_update_freq: int
        :param generator_update_freq: How frequently to update the generator
        :type generator_update_freq: int
        :param kwargs: Additional arguments passed to the AbstractTrainer
        :type kwargs: dict
        """
        super().__init__(**kwargs)

        # Validate update frequencies
        if discriminator_update_freq > 1 and generator_update_freq > 1:
            raise ValueError(
                "Both discriminator_update_freq and generator_update_freq cannot be greater than 1. "
                "At least one network must update every epoch."
            )
        
        self._generator = generator
        self._discriminator = discriminator
        self._gen_optimizer = gen_optimizer
        self._disc_optimizer = disc_optimizer
        self._generator_loss_fn = generator_loss_fn
        self._generator_loss_fn.trainer = self
        self._discriminator_loss_fn = discriminator_loss_fn
        self._discriminator_loss_fn.trainer = self
        self._gradient_penalty_fn = gradient_penalty_fn
        if self._gradient_penalty_fn is not None:
            self._gradient_penalty_fn.trainer = self

        # Global step counter and update frequencies
        self._discriminator_update_freq = discriminator_update_freq
        self._generator_update_freq = generator_update_freq

        ## TODO: instead of memorizing the same loss, keep a running average of the losses
        # Memory for discriminator and generator losses from the most recent update
        self._last_discriminator_loss = torch.tensor(0.0, device=self.device).detach()
        self._last_gradient_penalty_loss = torch.tensor(0.0, device=self.device).detach()
        self._last_generator_loss = torch.tensor(0.0, device=self.device).detach()

    def train_step(self, 
                   inputs: torch.tensor, 
                   targets: torch.tensor
                   ):
        """
        Perform a single training step on batch.

        :param inputs: The input image data batch
        :type inputs: torch.tensor
        :param targets: The target image data batch
        :type targets: torch.tensor
        """
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        gp_loss = torch.tensor(0.0, device=self.device)

        # foward pass to generate image (shared by both updates)
        generated_images = self._generator(inputs)

        # Train Discriminator
        if self.epoch % self._discriminator_update_freq == 0:
            self._disc_optimizer.zero_grad()

            real_images = targets

            # Concatenate input channel and real/generated image channels along the 
            # channel dimension to feed full stacked multi-channel images to the discriminator
            real_input_pair = torch.cat((real_images, inputs), 1)
            generated_input_pair = torch.cat((generated_images.detach(), inputs), 1)

            discriminator_real_score = self._discriminator(real_input_pair).mean()
            discriminator_fake_score = self._discriminator(generated_input_pair).mean()

            # Adverserial loss
            discriminator_loss = self._discriminator_loss_fn(discriminator_real_score, discriminator_fake_score)

            # Compute Gradient penalty loss if fn is supplied
            if self._gradient_penalty_fn is not None:
                gp_loss = self._gradient_penalty_fn(real_input_pair, generated_input_pair)

            total_discriminator_loss = discriminator_loss + gp_loss
            total_discriminator_loss.backward()
            self._disc_optimizer.step()

            # memorize current discriminator loss until next discriminator update
            self._last_discriminator_loss = discriminator_loss.detach()
            self._last_gradient_penalty_loss = gp_loss
        else:
            # when not being updated, use the loss from previus update
            discriminator_loss = self._last_discriminator_loss
            gp_loss = self._last_gradient_penalty_loss

        # Train Generator
        if self.epoch % self._generator_update_freq == 0:
            self._gen_optimizer.zero_grad()

            discriminator_fake_score = self._discriminator(torch.cat((generated_images, inputs), 1)).mean()
            generator_loss = self._generator_loss_fn(discriminator_fake_score, real_images, generated_images, self.epoch)
            generator_loss.backward()
            self._gen_optimizer.step()

            # memorize current generator loss until next generator update
            self._last_generator_loss = generator_loss.detach()
        else:
            # when not being updated, set the loss to zero
            generator_loss = self._last_generator_loss            

        for _, metric in self.metrics.items():
            ## TODO: centralize the update of metrics
            # compute the generated fake targets regardless for use with metrics
            generated_images = self._generator(inputs).detach()
            metric.update(generated_images, targets, validation=False)
            ## After each batch -> after each epoch
        
        loss = {type(self._discriminator_loss_fn).__name__: discriminator_loss.item(),
                type(self._generator_loss_fn).__name__: generator_loss.item()}
        if self._gradient_penalty_fn is not None:
            loss = {
                **loss,
                **{type(self._gradient_penalty_fn).__name__: gp_loss.item()} 
            }                     

        return loss

    def evaluate_step(self, 
                      inputs: torch.tensor, 
                      targets: torch.tensor
                      ):
        """
        Perform a single evaluation step on batch.

        :param inputs: The input image data batch
        :type inputs: torch.tensor
        :param targets: The target image data batch
        :type targets: torch.tensor
        """
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self._generator.eval()
        self._discriminator.eval()
        with torch.no_grad():

            real_images = targets
            generated_images = self._generator(inputs)
            
            # Concatenate input channel and real/generated image channels along the 
            # channel dimension to feed full stacked multi-channel images to the discriminator
            real_input_pair = torch.cat((real_images, inputs), 1)
            generated_input_pair = torch.cat((generated_images, inputs), 1)

            discriminator_real_score = self._discriminator(real_input_pair).mean()
            discriminator_fake_score = self._discriminator(generated_input_pair).mean()

            # Compute losses
            discriminator_loss = self._discriminator_loss_fn(discriminator_real_score, discriminator_fake_score)

            ## TODO: decide if gradient loss computation during eval mode is meaningful
            gp_loss = torch.tensor(0.0, device=self.device)

            generator_loss = self._generator_loss_fn(discriminator_fake_score, generated_images, real_images, self.epoch)

            for _, metric in self.metrics.items():
                metric.update(generated_images, targets, validation=True)
            
        loss = {type(self._discriminator_loss_fn).__name__: discriminator_loss.item(),
                type(self._generator_loss_fn).__name__: generator_loss.item()}
        if self._gradient_penalty_fn is not None:
            loss = {
                **loss,
                **{type(self._gradient_penalty_fn).__name__: gp_loss.item()} 
            }                     

        return loss

    def train_epoch(self):
        
        self._generator.train()
        self._discriminator.train()

        epoch_losses = defaultdict(list)
        for inputs, targets in self._train_loader:
            losses = self.train_step(inputs, targets)
            for key, value in losses.items():
                epoch_losses[key].append(value)

        for key, _ in epoch_losses.items():
            epoch_losses[key] = sum(epoch_losses[key])/len(self._train_loader)
            
        return epoch_losses

    def evaluate_epoch(self):

        self._generator.eval()
        self._discriminator.eval()

        epoch_losses = defaultdict(list)
        for inputs, targets in self._val_loader:
            losses = self.evaluate_step(inputs, targets)
            for key, value in losses.items():
                epoch_losses[key].append(value)
        
        for key, _ in epoch_losses.items():
            epoch_losses[key] = sum(epoch_losses[key])/len(self._val_loader)
        
        return epoch_losses
    
    def train(self):

        self._discriminator.to(self.device)

        super().train()
    
    @property
    def model(self) -> torch.nn.Module:
        """
        return the generator
        """
        return self._generator
    
    @property
    def discriminator(self) -> torch.nn.Module:
        """
        returns the discriminator
        """
        return self._discriminator