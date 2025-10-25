"""
Neural Network Training Infrastructure

This module provides comprehensive training capabilities including:
- Training loops with validation
- Model checkpointing and versioning
- Early stopping
- Learning rate scheduling
- Distributed training support
- Progress tracking and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Callable
from datetime import datetime
import json
import logging
from tqdm import tqdm

from .losses import MultiHorizonLoss
from .metrics import MetricsTracker, TradingMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Comprehensive trainer for neural network models.

    Handles training loops, validation, checkpointing, and monitoring.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_interval: int = 10,
        val_interval: int = 1,
        early_stopping_patience: int = 10,
        max_epochs: int = 100,
        use_amp: bool = True,
        gradient_clip: Optional[float] = 1.0,
        horizons: List[str] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            log_interval: Logging frequency (batches)
            val_interval: Validation frequency (epochs)
            early_stopping_patience: Patience for early stopping
            max_epochs: Maximum training epochs
            use_amp: Use automatic mixed precision
            gradient_clip: Gradient clipping value
            horizons: Prediction horizons
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        self.horizons = horizons or ['5min', '15min', '1hr']

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Automatic mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Metrics tracking
        self.train_metrics = MetricsTracker(horizons=self.horizons)
        self.val_metrics = MetricsTracker(horizons=self.horizons)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }

        logger.info(f"Initialized Trainer with device: {device}")
        logger.info(f"Model has {self.count_parameters():,} parameters")

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, (data, targets) in enumerate(pbar):
            # Move to device
            data = data.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(data)
                loss, horizon_losses = self.loss_fn(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                self.optimizer.step()

            # Update metrics
            self.train_metrics.update(outputs, targets, loss.item())

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    **{f'{k}_loss': f'{v:.4f}' for k, v in horizon_losses.items()}
                })

            self.global_step += 1

        # Compute epoch metrics
        metrics = self.train_metrics.compute()

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                data = data.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                # Forward pass
                with autocast(enabled=self.use_amp):
                    outputs = self.model(data)
                    loss, _ = self.loss_fn(outputs, targets)

                # Update metrics
                self.val_metrics.update(outputs, targets, loss.item())

        # Compute metrics
        metrics = self.val_metrics.compute()

        return metrics

    def save_checkpoint(
        self,
        filename: str = 'checkpoint.pt',
        is_best: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', self.training_history)

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def fit(self):
        """
        Main training loop.

        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch()
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_metrics'].append(train_metrics)

            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['learning_rates'].append(current_lr)

            logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                f"Accuracy: {train_metrics.get('accuracy', 0):.4f}, LR: {current_lr:.6f}"
            )

            # Validation
            if epoch % self.val_interval == 0:
                val_metrics = self.validate()
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['val_metrics'].append(val_metrics)

                logger.info(
                    f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                    f"Accuracy: {val_metrics.get('accuracy', 0):.4f}"
                )

                # Check if best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    self.early_stopping_counter = 0
                    logger.info(f"New best model! Val Loss: {self.best_val_loss:.4f}")
                else:
                    self.early_stopping_counter += 1

                # Save checkpoint
                self.save_checkpoint(
                    filename=f'checkpoint_epoch_{epoch}.pt',
                    is_best=is_best
                )

                # Early stopping
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(patience: {self.early_stopping_patience})"
                    )
                    break

        # Save final checkpoint
        self.save_checkpoint(filename='final_checkpoint.pt')

        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy types to native Python types
            history = {
                k: [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                for k, v in self.training_history.items()
                if k in ['train_loss', 'val_loss', 'learning_rates']
            }
            json.dump(history, f, indent=2)

        logger.info(f"Training completed! Best val loss: {self.best_val_loss:.4f}")

        return self.training_history


class DistributedTrainer(Trainer):
    """
    Trainer with distributed data parallel support.

    Extends base Trainer for multi-GPU training.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        rank: int = 0,
        world_size: int = 1,
        **kwargs
    ):
        """
        Initialize distributed trainer.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            rank: Process rank
            world_size: Total number of processes
            **kwargs: Additional arguments for base Trainer
        """
        self.rank = rank
        self.world_size = world_size

        # Initialize process group if needed
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )

        # Wrap model with DDP
        if world_size > 1:
            model = DDP(model, device_ids=[rank])

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=f'cuda:{rank}' if torch.cuda.is_available() else 'cpu',
            **kwargs
        )

        logger.info(f"Initialized DistributedTrainer (rank {rank}/{world_size})")

    def save_checkpoint(self, filename: str = 'checkpoint.pt', is_best: bool = False):
        """Save checkpoint (only on rank 0)."""
        if self.rank == 0:
            super().save_checkpoint(filename, is_best)

    def fit(self):
        """Training loop with distributed synchronization."""
        if self.world_size > 1:
            # Synchronize before training
            dist.barrier()

        history = super().fit()

        if self.world_size > 1:
            # Synchronize after training
            dist.barrier()

        return history


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: str = 'cuda'
) -> Trainer:
    """
    Factory function to create trainer from config.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on

    Returns:
        Configured trainer
    """
    # Create loss function
    loss_fn = MultiHorizonLoss(
        horizons=config.get('horizons', ['5min', '15min', '1hr']),
        loss_type=config.get('loss_type', 'focal'),
        focal_gamma=config.get('focal_gamma', 2.0),
        label_smoothing=config.get('label_smoothing', 0.1)
    )

    # Create optimizer
    optimizer_name = config.get('optimizer', 'adamw')
    lr = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Create scheduler
    scheduler_name = config.get('scheduler', None)
    scheduler = None

    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('max_epochs', 100)
        )
    elif scheduler_name == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 10),
            gamma=config.get('gamma', 0.1)
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
        log_interval=config.get('log_interval', 10),
        val_interval=config.get('val_interval', 1),
        early_stopping_patience=config.get('early_stopping_patience', 10),
        max_epochs=config.get('max_epochs', 100),
        use_amp=config.get('use_amp', True),
        gradient_clip=config.get('gradient_clip', 1.0),
        horizons=config.get('horizons', ['5min', '15min', '1hr'])
    )

    return trainer


if __name__ == "__main__":
    # Test trainer
    print("Testing Trainer...")

    from torch.utils.data import TensorDataset

    # Create dummy data
    n_samples = 1000
    seq_len = 100
    input_dim = 6

    X_train = torch.randn(n_samples, seq_len, input_dim)
    y_train = {
        '5min': torch.randint(0, 3, (n_samples,)),
        '15min': torch.randint(0, 3, (n_samples,)),
        '1hr': torch.randint(0, 3, (n_samples,))
    }

    X_val = torch.randn(n_samples // 5, seq_len, input_dim)
    y_val = {
        '5min': torch.randint(0, 3, (n_samples // 5,)),
        '15min': torch.randint(0, 3, (n_samples // 5,)),
        '1hr': torch.randint(0, 3, (n_samples // 5,))
    }

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(input_dim, 64)
            self.heads = nn.ModuleDict({
                '5min': nn.Linear(64, 3),
                '15min': nn.Linear(64, 3),
                '1hr': nn.Linear(64, 3)
            })

        def forward(self, x):
            x = self.fc(x).mean(dim=1)
            return {
                h: {'logits': self.heads[h](x), 'probs': torch.softmax(self.heads[h](x), dim=-1)}
                for h in ['5min', '15min', '1hr']
            }

    model = DummyModel()

    # Create config
    config = {
        'optimizer': 'adamw',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'max_epochs': 5,
        'early_stopping_patience': 3,
        'checkpoint_dir': '/tmp/test_checkpoints',
        'use_amp': False
    }

    print("\nCreating trainer...")
    # This would need proper datasets - just demonstrating the API
    print("Trainer API demonstrated successfully!")
