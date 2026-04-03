#!/usr/bin/env python3
"""
Register all patient images to prepare for training.
This script processes all patient folders and creates registered image pairs.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import ImagePreprocessor
from src.utils.config import get_config


def find_image_pair(patient_folder: Path, eye: str, centering: str):
    """Find matching Zeiss and Clarus image pair."""
    zeiss_folder = patient_folder / "ZEISS - LOW QUALITY"
    clarus_folder = patient_folder / "CLARUS - HIGH QUALITY"
    
    if not zeiss_folder.exists() or not clarus_folder.exists():
        return None, None
    
    # Find Zeiss image
    zeiss_pattern = f"{eye} {centering}"
    zeiss_img = None
    for ext in ['.JPG', '.jpg', '.PNG', '.png']:
        zeiss_path = zeiss_folder / f"{zeiss_pattern}{ext}"
        if zeiss_path.exists():
            zeiss_img = zeiss_path
            break
    
    # Find Clarus image  
    clarus_pattern = f"{eye} {centering}"
    clarus_img = None
    for ext in ['.jpg', '.JPG', '.png', '.PNG']:
        clarus_path = clarus_folder / f"{clarus_pattern}{ext}"
        if clarus_path.exists():
            clarus_img = clarus_path
            break
        # Try with (1) suffix
        clarus_path = clarus_folder / f"{clarus_pattern} (1){ext}"
        if clarus_path.exists():
            clarus_img = clarus_path
            break
    
    return zeiss_img, clarus_img


def simple_registration(zeiss_img: np.ndarray, clarus_img: np.ndarray):
    """Simple registration - just resize to match."""
    # Resize Zeiss to match Clarus dimensions
    h, w = clarus_img.shape[:2]
    zeiss_resized = cv2.resize(zeiss_img, (w, h), interpolation=cv2.INTER_CUBIC)
    return zeiss_resized


def process_patient_folder(patient_folder: Path, output_dir: Path, preprocessor: ImagePreprocessor):
    """Process one patient folder."""
    patient_id = patient_folder.name
    pairs_processed = 0
    
    for eye in ['LE', 'RE']:
        for centering in ['DC', 'MC']:
            zeiss_path, clarus_path = find_image_pair(patient_folder, eye, centering)
            
            if zeiss_path is None or clarus_path is None:
                continue
            
            # Read images
            zeiss_img = cv2.imread(str(zeiss_path))
            clarus_img = cv2.imread(str(clarus_path))
            
            if zeiss_img is None or clarus_img is None:
                continue
            
            # Apply DCP dehazing to Zeiss image
            zeiss_dehazed = preprocessor.apply_dcp_dehazing(zeiss_img)
            
            # Simple registration (resize to match)
            zeiss_registered = simple_registration(zeiss_dehazed, clarus_img)
            
            # Create output directory
            pair_id = f"{patient_id}_{eye}_{centering}"
            pair_output_dir = output_dir / pair_id
            pair_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save outputs
            cv2.imwrite(str(pair_output_dir / "registered_zeiss.png"), zeiss_registered)
            cv2.imwrite(str(pair_output_dir / "clarus_original.png"), clarus_img)
            
            pairs_processed += 1
    
    return pairs_processed


def main():
    """Main registration pipeline."""
    config = get_config()
    
    # Paths
    root_dir = Path(config.get('dataset.root_dir', 'patient_images'))
    output_dir = Path(config.get('dataset.registered_dir', 'outputs/registered'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(config.get('preprocessing', {}))
    
    # Find all patient folders
    patient_folders = sorted([
        f for f in root_dir.iterdir()
        if f.is_dir() and not f.name.startswith('.')
    ])
    
    print(f"Found {len(patient_folders)} patient folders")
    print(f"Output directory: {output_dir}")
    
    # Process each patient
    total_pairs = 0
    for patient_folder in tqdm(patient_folders, desc="Registering images"):
        pairs = process_patient_folder(patient_folder, output_dir, preprocessor)
        total_pairs += pairs
    
    print(f"\nRegistration complete!")
    print(f"Total pairs registered: {total_pairs}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

