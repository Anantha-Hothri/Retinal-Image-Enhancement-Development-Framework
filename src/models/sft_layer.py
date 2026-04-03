"""Spatial Feature Transform (SFT) layer for conditional image generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFTLayer(nn.Module):
    """
    Spatial Feature Transform layer.
    
    Applies affine transformation to feature maps conditioned on external input:
    F_out = gamma(C) * F + beta(C)
    
    where gamma and beta are learned from conditioning input C (vessel map).
    """
    
    def __init__(self, in_channels: int, cond_channels: int = 1, num_layers: int = 3):
        """
        Initialize SFT layer.
        
        Args:
            in_channels: Number of feature map channels
            cond_channels: Number of conditioning input channels (1 for vessel map)
            num_layers: Number of convolutional layers in gamma/beta networks
        """
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        
        # Network to compute gamma (scale)
        gamma_layers = []
        current_channels = cond_channels
        
        for i in range(num_layers - 1):
            gamma_layers.extend([
                nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
            current_channels = in_channels
        
        # Final layer for gamma
        gamma_layers.append(nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1))
        self.gamma_net = nn.Sequential(*gamma_layers)
        
        # Network to compute beta (shift)
        beta_layers = []
        current_channels = cond_channels
        
        for i in range(num_layers - 1):
            beta_layers.extend([
                nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
            current_channels = in_channels
        
        # Final layer for beta
        beta_layers.append(nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1))
        self.beta_net = nn.Sequential(*beta_layers)
        
        # Initialize final layers with proper bias
        # gamma initialized to produce 1.0 (identity scale)
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        
        # beta initialized to produce 0.0 (identity shift)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply SFT transformation.
        
        Args:
            x: Feature map (B, C, H, W)
            cond: Conditioning input (B, 1, H', W') - vessel map
            
        Returns:
            Transformed feature map (B, C, H, W)
        """
        # Resize conditioning to match feature map spatial dimensions
        if cond.size(2) != x.size(2) or cond.size(3) != x.size(3):
            cond = F.interpolate(cond, size=(x.size(2), x.size(3)), 
                               mode='bilinear', align_corners=False)
        
        # Compute gamma and beta
        gamma = self.gamma_net(cond)
        beta = self.beta_net(cond)
        
        # Apply affine transformation
        out = gamma * x + beta
        
        return out


class SFTResidualBlock(nn.Module):
    """
    Residual block with SFT modulation.
    
    Applies SFT after the residual computation.
    """
    
    def __init__(self, num_channels: int, cond_channels: int = 1):
        """
        Initialize SFT residual block.
        
        Args:
            num_channels: Number of feature channels
            cond_channels: Number of conditioning channels
        """
        super().__init__()
        
        # Residual path
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
        # SFT modulation
        self.sft = SFTLayer(num_channels, cond_channels)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditioning.
        
        Args:
            x: Input features (B, C, H, W)
            cond: Conditioning input (B, 1, H', W')
            
        Returns:
            Output features (B, C, H, W)
        """
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Add residual
        out = out + residual
        
        # Apply SFT modulation
        out = self.sft(out, cond)
        
        return out


def test_sft_layer():
    """Test SFT layer functionality."""
    # Create dummy inputs
    batch_size = 2
    channels = 64
    height, width = 64, 64
    
    x = torch.randn(batch_size, channels, height, width)
    cond = torch.randn(batch_size, 1, height, width)  # Vessel map
    
    # Test SFT layer
    sft = SFTLayer(channels, cond_channels=1)
    out = sft(x, cond)
    
    print(f"Input shape: {x.shape}")
    print(f"Condition shape: {cond.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape mismatch"
    
    # Test SFT residual block
    sft_block = SFTResidualBlock(channels)
    out = sft_block(x, cond)
    
    print(f"SFT Residual output shape: {out.shape}")
    assert out.shape == x.shape, "SFT Residual output shape mismatch"
    
    print("SFT layer tests passed!")


if __name__ == "__main__":
    test_sft_layer()

