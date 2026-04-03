"""Loss functions for SFT-Real-ESRGAN training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PixelLoss(nn.Module):
    """Simple L1 pixel-wise loss between SR output and target."""
    
    def __init__(self, loss_type='l1'):
        """
        Initialize pixel loss.
        
        Args:
            loss_type: 'l1' or 'l2'
        """
        super().__init__()
        
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred, target):
        """Compute pixel loss."""
        return self.loss(pred, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    
    Computes L1 loss between feature maps extracted from pretrained VGG19.
    """
    
    def __init__(self, layer_weights=None):
        """
        Initialize perceptual loss.
        
        Args:
            layer_weights: Dict mapping layer names to weights
        """
        super().__init__()
        
        # Default layers and weights from ESRGAN paper
        if layer_weights is None:
            layer_weights = {
                'conv1_2': 0.1,
                'conv2_2': 0.1,
                'conv3_4': 1.0,
                'conv4_4': 1.0,
                'conv5_4': 1.0,
            }
        
        self.layer_weights = layer_weights
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        vgg.eval()
        
        # Disable gradients
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract feature extraction layers
        self.feature_layers = {}
        layer_mapping = {
            'conv1_2': 2,
            'conv2_2': 7,
            'conv3_4': 16,
            'conv4_4': 25,
            'conv5_4': 34,
        }
        
        for layer_name, layer_idx in layer_mapping.items():
            if layer_name in self.layer_weights:
                self.feature_layers[layer_name] = nn.Sequential(*list(vgg.children())[:layer_idx + 1])
        
        # Register as buffers
        for name, layer in self.feature_layers.items():
            self.add_module(f'vgg_{name}', layer)
    
    def forward(self, pred, target):
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
            
        Returns:
            Perceptual loss
        """
        loss = 0.0
        
        for layer_name, weight in self.layer_weights.items():
            if layer_name in self.feature_layers:
                layer = getattr(self, f'vgg_{layer_name}')
                
                pred_feat = layer(pred)
                target_feat = layer(target)
                
                loss += weight * F.l1_loss(pred_feat, target_feat)
        
        return loss


class GANLoss(nn.Module):
    """
    GAN loss for adversarial training.
    
    Supports vanilla, lsgan, and wgan loss types.
    """
    
    def __init__(self, gan_type='vanilla', real_label=1.0, fake_label=0.0):
        """
        Initialize GAN loss.
        
        Args:
            gan_type: Type of GAN loss ('vanilla', 'lsgan', 'wgan')
            real_label: Label value for real images
            fake_label: Label value for fake images
        """
        super().__init__()
        
        self.gan_type = gan_type
        self.real_label = real_label
        self.fake_label = fake_label
        
        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'wgan':
            self.loss = None
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")
    
    def forward(self, pred, target_is_real):
        """
        Compute GAN loss.
        
        Args:
            pred: Discriminator predictions
            target_is_real: Whether target should be real or fake
            
        Returns:
            GAN loss
        """
        if target_is_real:
            target = torch.ones_like(pred) * self.real_label
        else:
            target = torch.zeros_like(pred) * self.fake_label
        
        if self.gan_type == 'wgan':
            loss = -pred.mean() if target_is_real else pred.mean()
        else:
            loss = self.loss(pred, target)
        
        return loss


class VesselSegmentationLoss(nn.Module):
    """
    Vessel segmentation consistency loss.
    
    Compares vessel maps extracted from SR output vs target.
    Encourages preservation of vascular structure.
    """
    
    def __init__(self, unet_model):
        """
        Initialize vessel segmentation loss.
        
        Args:
            unet_model: Pretrained U-Net for vessel segmentation
        """
        super().__init__()
        
        self.unet = unet_model
        self.unet.eval()
        
        # Freeze U-Net parameters
        for param in self.unet.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        """
        Compute vessel segmentation loss.
        
        Args:
            pred: SR output (B, 3, H, W)
            target: Target Clarus image (B, 3, H, W)
            
        Returns:
            Vessel segmentation loss
        """
        with torch.no_grad():
            # Extract vessel maps
            pred_vessels = torch.sigmoid(self.unet(pred))
            target_vessels = torch.sigmoid(self.unet(target))
        
        # L1 loss between vessel maps
        loss = F.l1_loss(pred_vessels, target_vessels)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for SFT-Real-ESRGAN training.
    
    Combines:
    - Pixel loss (L1)
    - Perceptual loss (VGG features)
    - Adversarial loss (GAN)
    - Vessel segmentation loss
    """
    
    def __init__(
        self,
        unet_model=None,
        lambda_pixel=1.0,
        lambda_perceptual=1.0,
        lambda_gan=0.1,
        lambda_vessel=0.5
    ):
        """
        Initialize combined loss.
        
        Args:
            unet_model: Pretrained U-Net for vessel loss
            lambda_pixel: Weight for pixel loss
            lambda_perceptual: Weight for perceptual loss
            lambda_gan: Weight for adversarial loss
            lambda_vessel: Weight for vessel segmentation loss
        """
        super().__init__()
        
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_gan = lambda_gan
        self.lambda_vessel = lambda_vessel
        
        # Initialize loss functions
        self.pixel_loss = PixelLoss(loss_type='l1')
        self.perceptual_loss = PerceptualLoss()
        self.gan_loss = GANLoss(gan_type='vanilla')
        
        if unet_model is not None and lambda_vessel > 0:
            self.vessel_loss = VesselSegmentationLoss(unet_model)
        else:
            self.vessel_loss = None
    
    def forward(self, pred, target, disc_pred_fake=None):
        """
        Compute combined loss.
        
        Args:
            pred: SR output (B, 3, H, W)
            target: Target image (B, 3, H, W)
            disc_pred_fake: Discriminator prediction for fake image
            
        Returns:
            Total loss and dict of individual losses
        """
        losses = {}
        total_loss = 0.0
        
        # Pixel loss
        if self.lambda_pixel > 0:
            losses['pixel'] = self.pixel_loss(pred, target)
            total_loss += self.lambda_pixel * losses['pixel']
        
        # Perceptual loss
        if self.lambda_perceptual > 0:
            losses['perceptual'] = self.perceptual_loss(pred, target)
            total_loss += self.lambda_perceptual * losses['perceptual']
        
        # GAN loss
        if self.lambda_gan > 0 and disc_pred_fake is not None:
            losses['gan'] = self.gan_loss(disc_pred_fake, target_is_real=True)
            total_loss += self.lambda_gan * losses['gan']
        
        # Vessel segmentation loss
        if self.lambda_vessel > 0 and self.vessel_loss is not None:
            losses['vessel'] = self.vessel_loss(pred, target)
            total_loss += self.lambda_vessel * losses['vessel']
        
        losses['total'] = total_loss
        
        return total_loss, losses

