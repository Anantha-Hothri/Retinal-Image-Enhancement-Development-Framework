"""Dataset structure validation and pairing verification."""

import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import re


class DatasetValidator:
    """Validates patient_images dataset structure and manifest."""
    
    def __init__(self, root_dir: str = "patient_images", manifest_file: str = None):
        """
        Initialize dataset validator.
        
        Args:
            root_dir: Root directory containing patient folders
            manifest_file: Path to Excel manifest (default: root_dir/patient_images_report.xlsx)
        """
        self.root_dir = Path(root_dir)
        if manifest_file is None:
            manifest_file = self.root_dir / "patient_images_report.xlsx"
        self.manifest_file = Path(manifest_file)
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
        if not self.manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
    
    def load_manifest(self) -> pd.DataFrame:
        """Load Excel manifest file."""
        print(f"Loading manifest: {self.manifest_file}")
        df = pd.read_excel(self.manifest_file)
        print(f"Loaded {len(df)} entries from manifest")
        return df
    
    def tokenize_filename(self, filename: str) -> List[str]:
        """
        Split filename into tokens for matching.
        
        Args:
            filename: Filename to tokenize
            
        Returns:
            List of lowercase tokens
        """
        # Split on spaces, dashes, underscores
        tokens = re.split(r'[\s\-_]+', filename.lower())
        # Remove extension
        tokens = [t.replace('.jpg', '').replace('.jpeg', '').replace('.png', '') 
                  for t in tokens]
        return [t for t in tokens if t]
    
    def find_matching_file(self, folder: Path, label: str) -> str:
        """
        Find file in folder matching the label tokens.
        
        Args:
            folder: Folder to search
            label: Label from manifest
            
        Returns:
            Matched filename or None
        """
        if not folder.exists():
            return None
        
        label_tokens = self.tokenize_filename(label)
        
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                file_tokens = self.tokenize_filename(file.name)
                # Check if all label tokens appear in filename
                if all(token in file_tokens for token in label_tokens):
                    return file.name
        
        return None
    
    def find_subfolder(self, patient_folder: Path, keywords: List[str]) -> Path:
        """
        Find subfolder containing any of the keywords.
        
        Args:
            patient_folder: Patient folder to search
            keywords: List of keywords to match (case-insensitive)
            
        Returns:
            Matched subfolder path or None
        """
        if not patient_folder.exists():
            return None
        
        for subfolder in patient_folder.iterdir():
            if subfolder.is_dir():
                folder_name_lower = subfolder.name.lower()
                if any(keyword in folder_name_lower for keyword in keywords):
                    return subfolder
        
        return None
    
    def validate_dataset(self) -> Dict[str, any]:
        """
        Validate complete dataset structure.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_patients': 0,
            'valid_pairs': 0,
            'missing_clarus': 0,
            'missing_zeiss': 0,
            'errors': []
        }
        
        # Get all patient folders
        patient_folders = sorted([
            f for f in self.root_dir.iterdir() 
            if f.is_dir() and not f.name.startswith('.')
        ])
        
        results['total_patients'] = len(patient_folders)
        
        for patient_folder in patient_folders:
            # Find Clarus and Zeiss subfolders
            clarus_folder = self.find_subfolder(patient_folder, ['clarus', 'high quality'])
            zeiss_folder = self.find_subfolder(patient_folder, ['zeiss', 'visuscout', 'low quality'])
            
            if clarus_folder is None:
                results['missing_clarus'] += 1
                results['errors'].append(f"No Clarus folder in {patient_folder.name}")
                continue
            
            if zeiss_folder is None:
                results['missing_zeiss'] += 1
                results['errors'].append(f"No Zeiss folder in {patient_folder.name}")
                continue
            
            # Count valid image pairs (same eye laterality)
            clarus_images = list(clarus_folder.glob('*.jpg')) + list(clarus_folder.glob('*.png'))
            zeiss_images = list(zeiss_folder.glob('*.jpg')) + list(zeiss_folder.glob('*.png'))
            zeiss_images += list(zeiss_folder.glob('*.JPG'))  # Case-sensitive systems
            
            if len(clarus_images) > 0 and len(zeiss_images) > 0:
                results['valid_pairs'] += min(len(clarus_images), len(zeiss_images))
        
        return results
    
    def print_validation_report(self, results: Dict[str, any]):
        """Print formatted validation report."""
        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        print(f"Total patient folders: {results['total_patients']}")
        print(f"Valid image pairs found: {results['valid_pairs']}")
        print(f"Missing Clarus folders: {results['missing_clarus']}")
        print(f"Missing Zeiss folders: {results['missing_zeiss']}")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors'][:10]:  # Show first 10
                print(f"  - {error}")
            if len(results['errors']) > 10:
                print(f"  ... and {len(results['errors']) - 10} more")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    validator = DatasetValidator()
    results = validator.validate_dataset()
    validator.print_validation_report(results)

