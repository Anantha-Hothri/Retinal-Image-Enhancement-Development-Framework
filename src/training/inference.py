"""Inference script for SFT-Real-ESRGAN."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.real_esrgan_sft import create_sft_real_esrgan
from models.unet_vessel import load_pretrained_unet
from utils.preprocessing import dehaze_image
from utils.config import get_config


class RetinalEnhancer:
    """Inference wrapper for retinal image enhancement."""
    
    def __init__(
        self,
        checkpoint_path,
        unet_path='models/drive_unet.pth',
        config_path='configs/config.yaml',
        device=None
    ):
        """
        Initialize enhancer.
        
        Args:
            checkpoint_path: Path to trained SFT-Real-ESRGAN checkpoint
            unet_path: Path to pretrained U-Net weights
            config_path: Path to config file
            device: Device to run inference on
        """
        self.config = get_config(config_path)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load models
        self._load_models(checkpoint_path, unet_path)
    
    def _load_models(self, checkpoint_path, unet_path):
        """Load SFT-Real-ESRGAN and U-Net models."""
        print("Loading models...")
        
        # Load U-Net for vessel segmentation
        self.unet = load_pretrained_unet(unet_path, device=self.device)
        
        # Load SFT-Real-ESRGAN
        self.generator = create_sft_real_esrgan(
            config=self.config.get('model', {}),
            device=self.device
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    def preprocess_zeiss_image(self, image_path, apply_dehazing=True):
        """
        Preprocess Zeiss Visuscout image.
        
        Args:
            image_path: Path to Zeiss image
            apply_dehazing: Whether to apply DCP dehazing
            
        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply dehazing if requested
        if apply_dehazing:
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img_dehazed = dehaze_image(img_lab)
            img = cv2.cvtColor(img_dehazed, cv2.COLOR_LAB2RGB)
        
        # Normalize to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor, img.shape[:2]
    
    def extract_vessel_map(self, image_tensor):
        """
        Extract vessel segmentation map using U-Net.
        
        Args:
            image_tensor: Image tensor (1, 3, H, W)
            
        Returns:
            Vessel map tensor (1, 1, H, W)
        """
        with torch.no_grad():
            vessel_logits = self.unet(image_tensor)
            vessel_map = torch.sigmoid(vessel_logits)
        
        return vessel_map
    
    def enhance_image(
        self,
        zeiss_path,
        output_path=None,
        apply_dehazing=True,
        return_intermediates=False
    ):
        """
        Enhance Zeiss Visuscout image to Clarus quality.
        
        Args:
            zeiss_path: Path to Zeiss image
            output_path: Path to save enhanced image
            apply_dehazing: Whether to apply DCP dehazing
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Enhanced image (H, W, 3) as numpy array
            If return_intermediates=True, returns dict with all intermediate steps
        """
        # Preprocess input
        zeiss_tensor, original_size = self.preprocess_zeiss_image(
            zeiss_path,
            apply_dehazing=apply_dehazing
        )
        zeiss_tensor = zeiss_tensor.to(self.device)
        
        # Extract vessel map
        vessel_map = self.extract_vessel_map(zeiss_tensor)
        
        # Run super-resolution
        with torch.no_grad():
            enhanced_tensor = self.generator(zeiss_tensor, vessel_map)
        
        # Convert to numpy
        enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_np = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), enhanced_bgr)
            print(f"Saved enhanced image to {output_path}")
        
        # Return results
        if return_intermediates:
            vessel_np = vessel_map.squeeze().cpu().numpy()
            vessel_np = (vessel_np * 255).astype(np.uint8)
            
            return {
                'enhanced': enhanced_np,
                'vessel_map': vessel_np,
                'input_tensor': zeiss_tensor,
                'enhanced_tensor': enhanced_tensor
            }
        else:
            return enhanced_np


def main():
    """Main entry point for inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance retinal images')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input Zeiss image')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save enhanced image')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--unet-weights', type=str, default='models/drive_unet.pth',
                       help='Path to U-Net weights')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--no-dehazing', action='store_true',
                       help='Disable DCP dehazing')
    
    args = parser.parse_args()
    
    # Create enhancer
    enhancer = RetinalEnhancer(
        checkpoint_path=args.checkpoint,
        unet_path=args.unet_weights,
        config_path=args.config
    )
    
    # Enhance image
    print(f"Processing {args.input}...")
    enhanced = enhancer.enhance_image(
        zeiss_path=args.input,
        output_path=args.output,
        apply_dehazing=not args.no_dehazing,
        return_intermediates=False
    )
    
    print(f"Enhanced image shape: {enhanced.shape}")
    print("Done!")


if __name__ == '__main__':
    main()

