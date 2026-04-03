"""Residual-in-Residual Dense Block (RRDB) Network for Real-ESRGAN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """
    Dense block with 5 convolutions.
    Each layer receives features from all preceding layers.
    """
    
    def __init__(self, num_channels: int = 64, growth_channels: int = 32):
        """
        Initialize dense block.
        
        Args:
            num_channels: Number of input/output channels
            growth_channels: Growth rate for dense connections
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_channels + 4 * growth_channels, num_channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Residual scaling factor
        self.scaling = 0.2
    
    def forward(self, x):
        """Forward pass with dense connections."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        
        # Residual scaling
        return x5 * self.scaling + x


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block.
    Contains 3 dense blocks with residual connections.
    """
    
    def __init__(self, num_channels: int = 64, growth_channels: int = 32):
        """
        Initialize RRDB.
        
        Args:
            num_channels: Number of channels
            growth_channels: Growth rate for dense blocks
        """
        super().__init__()
        
        self.db1 = DenseBlock(num_channels, growth_channels)
        self.db2 = DenseBlock(num_channels, growth_channels)
        self.db3 = DenseBlock(num_channels, growth_channels)
        
        # Residual scaling factor
        self.scaling = 0.2
    
    def forward(self, x):
        """Forward pass through 3 dense blocks."""
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        
        # Residual scaling
        return out * self.scaling + x


class RRDBNet(nn.Module):
    """
    RRDB Network for super-resolution.
    
    Architecture:
    1. Initial conv layer
    2. Stack of RRDB blocks
    3. Trunk conv layer
    4. Upsampling layers
    5. Final conv layers
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_channels: int = 64,
        num_blocks: int = 23,
        scale_factor: int = 4
    ):
        """
        Initialize RRDB network.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_channels: Number of base channels
            num_blocks: Number of RRDB blocks
            scale_factor: Upsampling scale factor (2 or 4)
        """
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.conv_first = nn.Conv2d(in_channels, num_channels, 3, 1, 1)
        
        # RRDB blocks
        self.rrdb_blocks = nn.ModuleList([
            RRDB(num_channels) for _ in range(num_blocks)
        ])
        
        # Trunk convolution
        self.conv_trunk = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        
        # Upsampling layers
        self.upconv1 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        
        # Final layers
        self.conv_hr = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_channels, out_channels, 3, 1, 1)
        
        # Activation
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        """Forward pass."""
        # Initial feature extraction
        feat = self.conv_first(x)
        trunk = feat
        
        # RRDB blocks
        for rrdb in self.rrdb_blocks:
            trunk = rrdb(trunk)
        
        # Trunk conv with global residual
        trunk = self.conv_trunk(trunk)
        feat = feat + trunk
        
        # Upsampling (2x)
        feat = self.lrelu(self.upconv1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        # Upsampling (2x) - total 4x if scale_factor=4
        if self.scale_factor == 4:
            feat = self.lrelu(self.upconv2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        # Final convolution
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        
        return out


def load_pretrained_rrdb(weights_path: str, device: str = 'cpu') -> RRDBNet:
    """
    Load pretrained RRDB network.
    
    Args:
        weights_path: Path to pretrained weights
        device: Device to load model on
        
    Returns:
        Pretrained RRDBNet
    """
    model = RRDBNet(
        in_channels=3,
        out_channels=3,
        num_channels=64,
        num_blocks=23,
        scale_factor=4
    )
    
    import os
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained RRDB weights from {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}")
    
    model = model.to(device)
    return model

