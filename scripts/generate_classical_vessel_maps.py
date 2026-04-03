"""Generate vessel maps using classical image processing (CLAHE + morphology)."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_vessels_classical(image_path, output_path):
    """
    Extract vessel-like features using classical image processing.
    
    Args:
        image_path: Path to input retinal image
        output_path: Path to save vessel map
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not load {image_path}")
        return False
    
    # Convert to grayscale (use green channel which has best vessel contrast)
    if len(img.shape) == 3:
        gray = img[:, :, 1]  # Green channel
    else:
        gray = img
    
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Morphological operations to enhance vessel structures
    # Use black-hat transform to enhance dark structures (vessels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(blackhat, (5, 5), 0)
    
    # Threshold to create binary vessel map
    _, vessel_map = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: Apply morphological closing to connect vessel fragments
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_map = cv2.morphologyEx(vessel_map, cv2.MORPH_CLOSE, kernel_small)
    
    # Save vessel map
    cv2.imwrite(str(output_path), vessel_map)
    return True


def generate_vessel_maps_from_csv(csv_path, output_dir):
    """Generate vessel maps for all images in CSV."""
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
    success_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating vessel maps"):
        pair_id = row['pair_id']
        clarus_path = row['clarus_path']
        
        # Output path
        output_path = output_dir / f"{pair_id}_vessel.png"
        
        # Skip if already exists
        if output_path.exists():
            success_count += 1
            continue
        
        # Generate vessel map
        if extract_vessels_classical(clarus_path, output_path):
            success_count += 1
    
    print(f"\nSuccessfully generated {success_count}/{len(df)} vessel maps")
    print(f"Saved to {output_dir}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate classical vessel maps')
    parser.add_argument('--train-csv', type=str, default='outputs/data/train_pairs.csv',
                       help='Path to training pairs CSV')
    parser.add_argument('--val-csv', type=str, default='outputs/data/val_pairs.csv',
                       help='Path to validation pairs CSV')
    parser.add_argument('--output-dir', type=str, default='outputs/vessel_maps',
                       help='Output directory for vessel maps')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Generating Classical Vessel Maps")
    print("=" * 80)
    
    # Generate vessel maps for training set
    print("\n=== Processing Training Set ===")
    train_output_dir = Path(args.output_dir) / 'train'
    if Path(args.train_csv).exists():
        generate_vessel_maps_from_csv(args.train_csv, train_output_dir)
    else:
        print(f"Warning: Training CSV not found at {args.train_csv}")
    
    # Generate vessel maps for validation set
    print("\n=== Processing Validation Set ===")
    val_output_dir = Path(args.output_dir) / 'val'
    if Path(args.val_csv).exists():
        generate_vessel_maps_from_csv(args.val_csv, val_output_dir)
    else:
        print(f"Warning: Validation CSV not found at {args.val_csv}")
    
    print("\n=== Complete ===")
    print(f"Vessel maps saved to {args.output_dir}")


if __name__ == '__main__':
    main()

