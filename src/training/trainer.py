"""Training script for SFT-Real-ESRGAN."""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.real_esrgan_sft import create_sft_real_esrgan
from models.discriminator import create_discriminator
from models.unet_vessel import load_pretrained_unet
from training.losses import CombinedLoss, GANLoss
from data.retinal_dataset import RetinalImageDataset
from utils.config import get_config


class Trainer:
    """Trainer for SFT-Real-ESRGAN model."""
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)

        # Try MPS (Apple Silicon), then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("🚀 Using Apple Metal (MPS) GPU acceleration!")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("⚠️  Using CPU (slow)")

        print(f"Using device: {self.device}")
        
        # Create output directories
        self.checkpoint_dir = Path(self.config.get('training.checkpoint_dir', 'checkpoints'))
        self.log_dir = Path(self.config.get('training.log_dir', 'outputs/logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize losses
        self._init_losses()
        
        # Initialize data loaders
        self._init_data_loaders()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _init_models(self):
        """Initialize generator and discriminator."""
        print("Initializing models...")
        
        # Generator (SFT-Real-ESRGAN)
        self.generator = create_sft_real_esrgan(
            config=self.config.get('model', {}),
            pretrained_rrdb_path=self.config.get('model.pretrained_rrdb_path'),
            device=self.device
        )
        
        # Discriminator
        self.discriminator = create_discriminator(
            config=self.config.get('discriminator', {}),
            device=self.device
        )
        
        # U-Net for vessel segmentation (frozen, used in loss)
        unet_path = self.config.get('model.unet_weights_path', 'models/drive_unet.pth')
        self.unet = load_pretrained_unet(unet_path, device=self.device)
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def _init_optimizers(self):
        """Initialize optimizers for generator and discriminator."""
        lr_g = self.config.get('training.lr_generator', 1e-4)
        lr_d = self.config.get('training.lr_discriminator', 1e-4)
        
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(0.9, 0.999)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(0.9, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.StepLR(
            self.optimizer_g,
            step_size=self.config.get('training.lr_decay_step', 50),
            gamma=self.config.get('training.lr_decay_gamma', 0.5)
        )
        
        self.scheduler_d = optim.lr_scheduler.StepLR(
            self.optimizer_d,
            step_size=self.config.get('training.lr_decay_step', 50),
            gamma=self.config.get('training.lr_decay_gamma', 0.5)
        )
    
    def _init_losses(self):
        """Initialize loss functions."""
        self.criterion_g = CombinedLoss(
            unet_model=self.unet,
            lambda_pixel=self.config.get('loss.lambda_pixel', 1.0),
            lambda_perceptual=self.config.get('loss.lambda_perceptual', 1.0),
            lambda_gan=self.config.get('loss.lambda_gan', 0.1),
            lambda_vessel=self.config.get('loss.lambda_vessel', 0.5)
        )
        
        self.criterion_d = GANLoss(gan_type='vanilla')
    
    def _init_data_loaders(self):
        """Initialize training and validation data loaders."""
        # Training dataset
        train_csv = self.config.get('dataset.train_csv', 'outputs/data/train_pairs.csv')
        train_dataset = RetinalImageDataset(
            pairs_file=train_csv,
            clarus_vessel_maps_dir=self.config.get('dataset.vessel_maps_dir', 'outputs/vessel_maps/train'),
            patch_size=self.config.get('dataset.target_size', (512, 512))[0],
            apply_dehazing=True,
            mode='train'
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('training.batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('training.num_workers', 4),
            pin_memory=True
        )
        
        # Validation dataset (optional - may be empty)
        val_csv = self.config.get('dataset.val_csv', 'outputs/data/val_pairs.csv')
        try:
            val_dataset = RetinalImageDataset(
                pairs_file=val_csv,
                clarus_vessel_maps_dir=self.config.get('dataset.vessel_maps_dir', 'outputs/vessel_maps/val'),
                patch_size=self.config.get('dataset.target_size', (512, 512))[0],
                apply_dehazing=False,
                mode='val'
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('training.batch_size', 8),
                shuffle=False,
                num_workers=self.config.get('training.num_workers', 4),
                pin_memory=True
            )

            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
        except (pd.errors.EmptyDataError, FileNotFoundError):
            print(f"Train samples: {len(train_dataset)}")
            print("No validation data found - training without validation")
            self.val_loader = None
    
    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        epoch_g_losses = []
        epoch_d_losses = []
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            zeiss = batch['input'].to(self.device)
            clarus = batch['target'].to(self.device)
            vessel_map = batch['condition'].to(self.device)
            
            # === Train Discriminator ===
            self.optimizer_d.zero_grad()
            
            # Real images
            real_pred = self.discriminator(clarus)
            d_loss_real = self.criterion_d(real_pred, target_is_real=True)
            
            # Fake images
            with torch.no_grad():
                fake = self.generator(zeiss, vessel_map)
            fake_pred = self.discriminator(fake.detach())
            d_loss_fake = self.criterion_d(fake_pred, target_is_real=False)
            
            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            
            d_loss.backward()
            self.optimizer_d.step()

            # === Train Generator ===
            self.optimizer_g.zero_grad()

            # Generate fake images
            fake = self.generator(zeiss, vessel_map)

            # Discriminator prediction for fake images
            fake_pred = self.discriminator(fake)

            # Compute generator loss
            g_loss, g_losses_dict = self.criterion_g(fake, clarus, fake_pred)

            g_loss.backward()
            self.optimizer_g.step()

            # Update progress bar
            epoch_g_losses.append(g_loss.item())
            epoch_d_losses.append(d_loss.item())

            pbar.set_postfix({
                'g_loss': np.mean(epoch_g_losses),
                'd_loss': np.mean(epoch_d_losses)
            })

            # Log to TensorBoard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/g_loss', g_loss.item(), self.global_step)
                self.writer.add_scalar('train/d_loss', d_loss.item(), self.global_step)

                for loss_name, loss_value in g_losses_dict.items():
                    if loss_name != 'total':
                        self.writer.add_scalar(f'train/{loss_name}', loss_value.item(), self.global_step)

            self.global_step += 1

        return np.mean(epoch_g_losses), np.mean(epoch_d_losses)

    def validate(self, epoch):
        """Validate the model."""
        if self.val_loader is None:
            return None

        self.generator.eval()

        val_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                zeiss = batch['input'].to(self.device)
                clarus = batch['target'].to(self.device)
                vessel_map = batch['condition'].to(self.device)

                # Generate SR output
                fake = self.generator(zeiss, vessel_map)

                # Compute loss (without discriminator)
                g_loss, _ = self.criterion_g(fake, clarus, disc_pred_fake=None)

                val_losses.append(g_loss.item())

        avg_val_loss = np.mean(val_losses)

        # Log to TensorBoard
        self.writer.add_scalar('val/loss', avg_val_loss, epoch)

        return avg_val_loss

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Resumed from epoch {self.current_epoch}")

    def train(self, num_epochs=None, resume_from=None):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        if num_epochs is None:
            num_epochs = self.config.get('training.num_epochs', 200)

        if resume_from:
            self.load_checkpoint(resume_from)

        start_epoch = self.current_epoch

        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 80)

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_g_loss, train_d_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss = self.validate(epoch)

            # Update learning rate
            self.scheduler_g.step()
            self.scheduler_d.step()

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train G Loss: {train_g_loss:.4f}")
            print(f"  Train D Loss: {train_d_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR G: {self.optimizer_g.param_groups[0]['lr']:.6f}")
            print(f"  LR D: {self.optimizer_d.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            save_interval = self.config.get('training.save_interval', 10)
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch)

            # Save best model
            if val_loss is not None:
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  >>> New best model! Val Loss: {val_loss:.4f}")

        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        self.writer.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train SFT-Real-ESRGAN')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Create trainer
    trainer = Trainer(config_path=args.config)

    # Start training
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)


if __name__ == '__main__':
    main()

