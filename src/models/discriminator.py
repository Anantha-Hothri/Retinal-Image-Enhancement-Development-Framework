"""PatchGAN Discriminator for adversarial training."""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator that classifies image patches as real or fake.
    
    Architecture based on pix2pix paper (Isola et al., 2017).
    Outputs a matrix of predictions, one per patch of the input.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 64,
        num_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        """
        Initialize PatchGAN discriminator.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_filters: Base number of filters
            num_layers: Number of discriminator layers
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()
        
        # Choose normalization
        if use_spectral_norm:
            norm_layer = lambda x: nn.utils.spectral_norm(x)
        else:
            norm_layer = lambda x: x
        
        # Build discriminator layers
        layers = []
        
        # First layer: no normalization
        layers.append(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            layers.extend([
                norm_layer(nn.Conv2d(
                    num_filters * nf_mult_prev,
                    num_filters * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Penultimate layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        layers.extend([
            norm_layer(nn.Conv2d(
                num_filters * nf_mult_prev,
                num_filters * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1
            )),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Final layer: output single channel
        layers.append(
            nn.Conv2d(num_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            Patch predictions (B, 1, H/16, W/16) for num_layers=3
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator that operates on multiple resolutions.
    
    Helps capture both global structure and fine details.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_discriminators: int = 2,
        **disc_kwargs
    ):
        """
        Initialize multi-scale discriminator.
        
        Args:
            in_channels: Number of input channels
            num_discriminators: Number of discriminators at different scales
            **disc_kwargs: Arguments passed to each PatchGAN discriminator
        """
        super().__init__()
        
        self.num_discriminators = num_discriminators
        
        # Create discriminators
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(in_channels, **disc_kwargs)
            for _ in range(num_discriminators)
        ])
        
        # Downsampling for multi-scale
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass through multiple scales.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            List of predictions from each scale
        """
        outputs = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))
        
        return outputs


def create_discriminator(config: dict, device: str = 'cuda') -> nn.Module:
    """
    Factory function to create discriminator.
    
    Args:
        config: Configuration dictionary
        device: Device to create model on
        
    Returns:
        Discriminator model
    """
    use_multiscale = config.get('use_multiscale_discriminator', False)
    
    disc_config = {
        'in_channels': 3,
        'num_filters': config.get('discriminator_filters', 64),
        'num_layers': config.get('discriminator_layers', 3),
        'use_spectral_norm': config.get('use_spectral_norm', True)
    }
    
    if use_multiscale:
        model = MultiScaleDiscriminator(
            num_discriminators=config.get('num_discriminators', 2),
            **disc_config
        )
    else:
        model = PatchGANDiscriminator(**disc_config)
    
    model = model.to(device)
    
    return model


def test_discriminator():
    """Test discriminator functionality."""
    # Test PatchGAN
    disc = PatchGANDiscriminator(in_channels=3, num_filters=64, num_layers=3)
    
    x = torch.randn(2, 3, 256, 256)
    out = disc(x)
    
    print(f"PatchGAN input: {x.shape}")
    print(f"PatchGAN output: {out.shape}")
    
    # Test MultiScale
    multi_disc = MultiScaleDiscriminator(in_channels=3, num_discriminators=2)
    outputs = multi_disc(x)
    
    print(f"\nMultiScale outputs:")
    for i, out in enumerate(outputs):
        print(f"  Scale {i}: {out.shape}")
    
    print("\nDiscriminator tests passed!")


if __name__ == "__main__":
    test_discriminator()

