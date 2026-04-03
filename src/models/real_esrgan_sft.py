"""SFT-conditioned Real-ESRGAN for retinal image super-resolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rrdb_net import RRDB
from .sft_layer import SFTLayer


class SFTConditionedRRDB(nn.Module):
    """RRDB block with SFT conditioning."""
    
    def __init__(self, num_channels: int = 64, cond_channels: int = 1):
        """
        Initialize SFT-conditioned RRDB.
        
        Args:
            num_channels: Number of feature channels
            cond_channels: Number of conditioning channels (1 for vessel map)
        """
        super().__init__()
        
        # Original RRDB block
        self.rrdb = RRDB(num_channels)
        
        # SFT layer for conditioning
        self.sft = SFTLayer(num_channels, cond_channels)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditioning.
        
        Args:
            x: Input features (B, C, H, W)
            cond: Conditioning input - vessel map (B, 1, H', W')
            
        Returns:
            Conditioned features (B, C, H, W)
        """
        # RRDB processing
        out = self.rrdb(x)
        
        # SFT modulation
        out = self.sft(out, cond)
        
        return out


class SFTRealESRGAN(nn.Module):
    """
    SFT-conditioned Real-ESRGAN for vessel-guided super-resolution.
    
    Architecture:
    1. Initial conv
    2. Stack of SFT-conditioned RRDB blocks
    3. Trunk conv
    4. Upsampling
    5. Output conv
    
    The vessel segmentation map conditions the RRDB blocks via SFT layers.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_channels: int = 64,
        num_blocks: int = 23,
        scale_factor: int = 4,
        cond_channels: int = 1,
        freeze_backbone: bool = True
    ):
        """
        Initialize SFT-Real-ESRGAN.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_channels: Number of base channels
            num_blocks: Number of RRDB blocks
            scale_factor: Upsampling scale (2 or 4)
            cond_channels: Conditioning channels (1 for vessel map)
            freeze_backbone: Whether to freeze RRDB weights
        """
        super().__init__()
        
        self.scale_factor = scale_factor
        self.freeze_backbone = freeze_backbone
        
        # Initial convolution
        self.conv_first = nn.Conv2d(in_channels, num_channels, 3, 1, 1)
        
        # SFT-conditioned RRDB blocks
        self.rrdb_blocks = nn.ModuleList([
            SFTConditionedRRDB(num_channels, cond_channels) 
            for _ in range(num_blocks)
        ])
        
        # Trunk convolution
        self.conv_trunk = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        
        # Upsampling
        self.upconv1 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        if scale_factor == 4:
            self.upconv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        
        # High-resolution conv
        self.conv_hr = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        
        # Output conv
        self.conv_last = nn.Conv2d(num_channels, out_channels, 3, 1, 1)
        
        # Activation
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def load_rrdb_weights(self, weights_path: str):
        """
        Load pretrained RRDB weights into the backbone.
        
        Args:
            weights_path: Path to Real-ESRGAN pretrained weights
        """
        import os
        if not os.path.exists(weights_path):
            print(f"Warning: RRDB weights not found at {weights_path}")
            return
        
        # Load state dict
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Map to current model (only RRDB blocks, not SFT layers)
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # Skip SFT layers (they don't exist in pretrained model)
            if 'rrdb.rrdb' in k or 'conv_first' in k or 'conv_trunk' in k:
                pretrained_dict[k] = v
        
        # Update model weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded pretrained RRDB weights from {weights_path}")
        
        # Freeze RRDB backbone if specified
        if self.freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze RRDB blocks, train only SFT layers and upsampler."""
        for block in self.rrdb_blocks:
            # Freeze RRDB parameters
            for param in block.rrdb.parameters():
                param.requires_grad = False
            
            # Keep SFT parameters trainable
            for param in block.sft.parameters():
                param.requires_grad = True
        
        # Freeze initial conv and trunk conv
        for param in self.conv_first.parameters():
            param.requires_grad = False
        for param in self.conv_trunk.parameters():
            param.requires_grad = False
        
        # Keep upsampling and output layers trainable
        for param in self.upconv1.parameters():
            param.requires_grad = True
        if self.scale_factor == 4:
            for param in self.upconv2.parameters():
                param.requires_grad = True
        for param in self.conv_hr.parameters():
            param.requires_grad = True
        for param in self.conv_last.parameters():
            param.requires_grad = True
        
        print("RRDB backbone frozen. Training only SFT layers and upsampler.")
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with vessel map conditioning.
        
        Args:
            x: Input image (B, 3, H, W)
            cond: Vessel segmentation map (B, 1, H, W)
            
        Returns:
            Super-resolved output (B, 3, H*scale, W*scale)
        """
        # Initial feature extraction
        feat = self.conv_first(x)
        trunk = feat
        
        # SFT-conditioned RRDB blocks
        for rrdb_block in self.rrdb_blocks:
            trunk = rrdb_block(trunk, cond)
        
        # Trunk conv with global residual
        trunk = self.conv_trunk(trunk)
        feat = feat + trunk
        
        # Upsampling (2x)
        feat = F.interpolate(feat, scale_factor=2, mode='nearest')
        feat = self.lrelu(self.upconv1(feat))
        
        # Upsampling (2x) for 4x total
        if self.scale_factor == 4:
            feat = F.interpolate(feat, scale_factor=2, mode='nearest')
            feat = self.lrelu(self.upconv2(feat))
        
        # High-resolution processing
        feat = self.lrelu(self.conv_hr(feat))

        # Output
        out = self.conv_last(feat)

        # Clamp output to [0, 1] range for image generation
        out = torch.clamp(out, 0, 1)

        return out


def create_sft_real_esrgan(
    config: dict,
    pretrained_rrdb_path: str = None,
    device: str = 'cuda'
) -> SFTRealESRGAN:
    """
    Factory function to create SFT-Real-ESRGAN model.
    
    Args:
        config: Configuration dictionary
        pretrained_rrdb_path: Path to pretrained Real-ESRGAN weights
        device: Device to create model on
        
    Returns:
        Initialized SFT-Real-ESRGAN model
    """
    model = SFTRealESRGAN(
        in_channels=3,
        out_channels=3,
        num_channels=config.get('num_channels', 64),
        num_blocks=config.get('num_rrdb_blocks', 23),
        scale_factor=config.get('scale_factor', 4),
        cond_channels=1,
        freeze_backbone=config.get('freeze_backbone', True)
    )
    
    # Load pretrained weights if provided
    if pretrained_rrdb_path:
        model.load_rrdb_weights(pretrained_rrdb_path)
    
    model = model.to(device)
    
    return model

