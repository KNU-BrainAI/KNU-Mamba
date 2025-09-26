"""Shared utilities for all main scripts."""

import os
import pickle
import random
import logging
import wandb
from typing import Any
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import multiprocessing as mp

from MambaVLA.utils.scaler import Scaler, ActionScaler, MinMaxScaler
from MambaVLA.utils.ema import ExponentialMovingAverage

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

log = logging.getLogger(__name__)


class Trainer:
    """Basic train/test class to be inherited."""

    def __init__(
            self,
            training_dataset: Any,
            validation_dataset: Any,
            training_batch_size: int = 512,
            validation_batch_size: int = 512,
            dataloader_workers: int = 8,
            device: str = 'cpu',
            total_epochs: int = 100,
            enable_data_scaling: bool = True,
            data_scaler_type: str = "minmax",
            evaluation_frequency: int = 50,
            observation_sequence_length: int = 1,
            ema_decay_rate: float = 0.999,
            enable_ema: bool = False,
            checkpoint_frequency: int = 10
    ):
        """Initialize."""
        
        # Dataset and data loading configuration
        self.trainset = training_dataset
        self.valset = validation_dataset
        self.train_batch_size = training_batch_size
        self.val_batch_size = validation_batch_size
        self.num_workers = dataloader_workers
        
        # Training configuration
        self.epoch = total_epochs
        self.perception_seq_len = observation_sequence_length
        self.eval_every_n_epochs = evaluation_frequency
        self.save_every_n_epochs = checkpoint_frequency
        
        # Device and environment configuration
        self.device = device
        self.working_dir = os.getcwd()
        
        # Data scaling configuration
        self.scale_data = enable_data_scaling
        self.scaling_type = data_scaler_type
        
        # EMA configuration
        self.decay_ema = ema_decay_rate
        self.if_use_ema = enable_ema
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize scaler
        self._setup_scaler()

        log.info("Number of training samples: {}".format(len(self.trainset)))

    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        self.train_dataloader = DataLoader(
            self.trainset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

        # self.test_dataloader = DataLoader(
        #     self.valset,
        #     batch_size=self.val_batch_size,
        #     shuffle=False,
        #     num_workers=0,
        #     pin_memory=True,
        #     drop_last=False
        # )

    def _setup_scaler(self):
        """Setup data scaler based on configuration."""
        if self.scaling_type == 'minmax':
            self.scaler = MinMaxScaler(self.trainset.get_all_actions(), self.scale_data, self.device)
        else:
            self.scaler = ActionScaler(self.trainset.get_all_actions(), self.scale_data, self.device)

    def main(self, model):
        """Run main training/testing pipeline."""
        self._setup_training_components(model)
        self._run_training_loop(model)
        self._finalize_training(model)

    def _setup_training_components(self, model):
        """Setup scaler, EMA, and optimizer for training."""
        # assign scaler to model calss
        model.set_scaler(self.scaler)

        if self.if_use_ema:
            self.ema_helper = ExponentialMovingAverage(model.parameters(), self.decay_ema, self.device)

        # define optimizer
        if model.use_lr_scheduler:
            self.optimizer, self.scheduler = model.configure_optimizers()
        else:
            self.optimizer = model.configure_optimizers()

    def _run_training_loop(self, model):
        """Execute the main training loop over all epochs."""
        for num_epoch in tqdm(range(self.epoch), desc="Epochs", dynamic_ncols=True):
            epoch_loss = self._train_single_epoch(model, num_epoch)
            self._log_epoch_results(num_epoch, epoch_loss)
            self._save_checkpoint_if_needed(model, num_epoch)

    def _train_single_epoch(self, model, num_epoch):
        """Train for a single epoch and return the average loss."""
        epoch_loss = torch.tensor(0.0).to(self.device)

        for data in tqdm(self.train_dataloader, desc="Batches", leave=False, dynamic_ncols=True):
            obs_dict, action, mask = data
            obs_dict, action = self._prepare_batch_data(obs_dict, action)
            batch_loss = self.train_one_step(model, obs_dict, action)
            epoch_loss += batch_loss

        return epoch_loss / len(self.train_dataloader)

    def _prepare_batch_data(self, obs_dict, action):
        """Prepare observation and action data for training."""
        # put data on cuda
        for camera in obs_dict.keys():
            if camera == 'lang':
                continue
            
            obs_dict[camera] = obs_dict[camera].to(self.device)

            if 'rgb' not in camera and 'image' not in camera:
                continue
            obs_dict[camera] = obs_dict[camera][:, :self.perception_seq_len].contiguous()

        action = self.scaler.scale_output(action)
        action = action[:, self.perception_seq_len - 1:, :].contiguous()

        return obs_dict, action

    def _log_epoch_results(self, num_epoch, epoch_loss):
        """Log epoch results to wandb and console."""
        wandb.log({"train_loss": epoch_loss.item()})
        log.info("Epoch {}: Mean train loss is {}".format(num_epoch, epoch_loss.item()))

    def _save_checkpoint_if_needed(self, model, num_epoch):
        """Save model checkpoint if it's time to do so."""
        if (num_epoch + 1) % self.save_every_n_epochs == 0:
            try:
                model.store_model_weights(self.working_dir, sv_name=f"epoch_{num_epoch + 1:05d}")
            except Exception as e:
                log.warning(f"Failed to save checkpoint at epoch {num_epoch + 1}: {e}")

    def _finalize_training(self, model):
        """Finalize training by applying EMA and saving final model."""
        log.info("training done")

        if self.if_use_ema:
            self.ema_helper.store(model.parameters())
            self.ema_helper.copy_to(model.parameters())

        model.store_model_weights(model.working_dir, sv_name='final_model')
        # or send weight out of the class

    def train_one_step(self, model, obs_dict, action):
        """Run a single training step."""
        model.train()

        loss = model(obs_dict, action)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if model.use_lr_scheduler:
            self.scheduler.step()

        if self.if_use_ema:
            self.ema_helper.update(model.parameters())

        return loss

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        return None

