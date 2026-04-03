"""Prepare dataset: run registration, create train/val split, generate CSV files."""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load patient images manifest."""
    print(f"Loading manifest from {manifest_path}")
    df = pd.read_excel(manifest_path)
    print(f"Loaded {len(df)} entries")
    return df


def find_patient_folders(root_dir: Path) -> List[Path]:
    """Find all patient folders in dataset root."""
    patient_folders = sorted([
        f for f in root_dir.iterdir()
        if f.is_dir() and not f.name.startswith('.')
    ])
    print(f"Found {len(patient_folders)} patient folders")
    return patient_folders


def split_patients_train_val(
    patient_folders: List[Path],
    train_size: int = 300,
    val_size: int = 52,
    random_seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """
    Split patient folders into train and validation sets.
    
    Args:
        patient_folders: List of patient folder paths
        train_size: Number of patients for training
        val_size: Number of patients for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_folders, val_folders)
    """
    np.random.seed(random_seed)
    
    # Shuffle patients
    shuffled = patient_folders.copy()
    np.random.shuffle(shuffled)
    
    # Note: This splits by patient count, not pair count
    # Each patient may have 1-2 pairs (left/right eyes)
    train_folders = shuffled[:train_size]
    val_folders = shuffled[train_size:train_size + val_size]
    
    print(f"Train: {len(train_folders)} patients")
    print(f"Val: {len(val_folders)} patients")
    
    return train_folders, val_folders


def create_pairs_csv(
    patient_folders: List[Path],
    output_csv: str,
    registered_dir: str = "outputs/registered"
):
    """
    Create CSV file listing all warped Zeiss and Clarus pairs.
    
    Assumes registration has already been run and outputs exist.
    
    Args:
        patient_folders: List of patient folders
        output_csv: Output CSV path
        registered_dir: Directory containing registration outputs
    """
    registered_path = Path(registered_dir)
    
    pairs = []
    
    for patient_folder in patient_folders:
        patient_id = patient_folder.name
        
        # Look for registration outputs for this patient
        # Format: outputs/registered/{patient_id}_{eye}/registered_zeiss.png
        for eye in ['LE', 'RE']:
            for centering in ['DC', 'MC']:
                pair_id = f"{patient_id}_{eye}_{centering}"
                pair_dir = registered_path / pair_id
                
                if not pair_dir.exists():
                    continue
                
                zeiss_warped = pair_dir / "registered_zeiss.png"
                clarus_original = pair_dir / "clarus_original.png"  # We'll copy this
                
                if zeiss_warped.exists():
                    # Find corresponding Clarus image
                    clarus_subfolder = None
                    for subfolder in patient_folder.iterdir():
                        if 'clarus' in subfolder.name.lower():
                            clarus_subfolder = subfolder
                            break
                    
                    if clarus_subfolder:
                        # Find matching Clarus image
                        for img_file in clarus_subfolder.glob('*.jpg'):
                            if eye in img_file.name and centering in img_file.name:
                                pairs.append({
                                    'pair_id': pair_id,
                                    'patient_id': patient_id,
                                    'eye': eye,
                                    'centering': centering,
                                    'zeiss_warped_path': str(zeiss_warped),
                                    'clarus_path': str(img_file),
                                })
                                break
    
    print(f"Found {len(pairs)} registered pairs")
    
    df = pd.DataFrame(pairs)
    df.to_csv(output_csv, index=False)
    print(f"Saved pairs CSV to {output_csv}")
    
    return df


def main():
    """Main data preparation pipeline."""
    config = get_config()
    
    # Paths
    root_dir = Path(config.get('dataset.root_dir', 'patient_images'))
    manifest_file = Path(config.get('dataset.manifest_file', 'patient_images/patient_images_report.xlsx'))
    output_dir = Path(config.get('dataset.output_dir', 'outputs'))
    
    # Parameters
    train_size = config.get('dataset.train_split', 300)
    val_size = config.get('dataset.val_split', 52)
    random_seed = config.get('dataset.random_seed', 42)
    
    # Create output directories
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all patient folders
    patient_folders = find_patient_folders(root_dir)
    
    # Split into train/val
    train_folders, val_folders = split_patients_train_val(
        patient_folders, train_size, val_size, random_seed
    )
    
    # Create CSV files
    print("\nCreating training pairs CSV...")
    train_csv = data_dir / 'train_pairs.csv'
    create_pairs_csv(train_folders, str(train_csv))
    
    print("\nCreating validation pairs CSV...")
    val_csv = data_dir / 'val_pairs.csv'
    create_pairs_csv(val_folders, str(val_csv))
    
    print(f"\nDataset preparation complete!")
    print(f"  Train CSV: {train_csv}")
    print(f"  Val CSV: {val_csv}")


if __name__ == "__main__":
    main()

