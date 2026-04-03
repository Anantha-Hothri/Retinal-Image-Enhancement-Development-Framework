"""Generate vessel segmentation maps for all Clarus images using pretrained U-Net."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet_vessel import load_pretrained_unet
from utils.config import get_config


def preprocess_for_unet(image_path, target_size=(584, 565)):
    """
    Preprocess image for U-Net vessel segmentation.
    
    Args:
        image_path: Path to image
        target_size: Target size for U-Net input
        
    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to U-Net input size
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
    
    # Add batch dimension (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, img.shape[:2]


def postprocess_vessel_map(vessel_map, original_size, threshold=0.5):
    """
    Postprocess vessel segmentation output.
    
    Args:
        vessel_map: Raw U-Net output (1, 1, H, W)
        original_size: Original image size (H, W)
        threshold: Threshold for binary segmentation
        
    Returns:
        Binary vessel map (H, W) resized to original size
    """
    # Remove batch and channel dimensions
    vessel_map = vessel_map.squeeze().cpu().numpy()
    
    # Apply sigmoid if needed
    vessel_map = 1 / (1 + np.exp(-vessel_map))  # Sigmoid
    
    # Threshold
    vessel_binary = (vessel_map > threshold).astype(np.uint8) * 255
    
    # Resize to original size
    vessel_resized = cv2.resize(
        vessel_binary,
        (original_size[1], original_size[0]),  # (W, H)
        interpolation=cv2.INTER_LINEAR
    )
    
    return vessel_resized


def generate_vessel_maps_from_csv(
    csv_path,
    output_dir,
    unet_model,
    device='cuda'
):
    """
    Generate vessel maps for all images in CSV.
    
    Args:
        csv_path: Path to CSV with image pairs
        output_dir: Directory to save vessel maps
        unet_model: Loaded U-Net model
        device: Device to run inference on
    """
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print(f"No images found in {csv_path}")
            return
        print(f"Processing {len(df)} images from {csv_path}")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"CSV file is empty or not found: {csv_path}")
        return
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating vessel maps"):
        clarus_path = row['clarus_path']
        pair_id = row['pair_id']
        
        try:
            # Preprocess
            img_tensor, original_size = preprocess_for_unet(clarus_path)
            img_tensor = img_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                vessel_logits = unet_model(img_tensor)
            
            # Postprocess
            vessel_map = postprocess_vessel_map(vessel_logits, original_size)
            
            # Save
            vessel_output_path = output_dir / f"{pair_id}_vessel.png"
            cv2.imwrite(str(vessel_output_path), vessel_map)
            
        except Exception as e:
            print(f"Error processing {pair_id}: {e}")
            continue
    
    print(f"Vessel maps saved to {output_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate vessel segmentation maps')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--unet-weights', type=str, default='models/drive_unet.pth',
                       help='Path to pretrained U-Net weights')
    parser.add_argument('--output-dir', type=str, default='outputs/vessel_maps',
                       help='Output directory for vessel maps')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load U-Net
    print(f"Loading U-Net from {args.unet_weights}")
    unet = load_pretrained_unet(args.unet_weights, device=device)
    
    # Get CSV paths
    train_csv = config.get('dataset.train_csv', 'outputs/data/train_pairs.csv')
    val_csv = config.get('dataset.val_csv', 'outputs/data/val_pairs.csv')
    
    # Generate vessel maps for training set
    print("\n=== Processing Training Set ===")
    train_output_dir = Path(args.output_dir) / 'train'
    if Path(train_csv).exists():
        generate_vessel_maps_from_csv(
            train_csv,
            train_output_dir,
            unet,
            device
        )
    else:
        print(f"Warning: Training CSV not found at {train_csv}")
    
    # Generate vessel maps for validation set
    print("\n=== Processing Validation Set ===")
    val_output_dir = Path(args.output_dir) / 'val'
    if Path(val_csv).exists():
        generate_vessel_maps_from_csv(
            val_csv,
            val_output_dir,
            unet,
            device
        )
    else:
        print(f"Warning: Validation CSV not found at {val_csv}")
    
    print("\n=== Complete ===")
    print(f"Vessel maps saved to {args.output_dir}")


if __name__ == '__main__':
    main()

