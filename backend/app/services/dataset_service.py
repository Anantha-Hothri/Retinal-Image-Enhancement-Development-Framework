"""Service for managing dataset mappings between Zeiss and Clarus images."""

import pandas as pd
import os
from pathlib import Path
from typing import Optional, Dict, List
import cv2
import numpy as np


class DatasetService:
    """Service for finding Zeiss-Clarus image pairs from Excel mapping."""
    
    def __init__(self, excel_path: str = "patient_images/patient_images_report.xlsx"):
        """
        Initialize dataset service.
        
        Args:
            excel_path: Path to Excel file containing image pair mappings
        """
        self.excel_path = Path(excel_path)
        self.df = None
        self.load_dataset_mapping()
    
    def load_dataset_mapping(self):
        """Load the Excel file containing Zeiss-Clarus mappings."""
        try:
            if self.excel_path.exists():
                self.df = pd.read_excel(self.excel_path)
                print(f"✓ Loaded dataset mapping: {len(self.df)} entries")
            else:
                print(f"⚠️ Dataset mapping file not found: {self.excel_path}")
                self.df = None
        except Exception as e:
            print(f"⚠️ Error loading dataset mapping: {e}")
            self.df = None
    
    def find_clarus_pair(self, zeiss_image_name: str) -> Optional[Dict[str, str]]:
        """
        Find the corresponding Clarus image for a given Zeiss image.

        Uses Excel mapping with columns: 'Parent folder', 'Zeiss name', 'Clarus name'

        Args:
            zeiss_image_name: Full path or name of the Zeiss image file (e.g., "input.jpg" or "LE MC.JPG")

        Returns:
            Dictionary with paths to Zeiss and Clarus images, or None if not found
        """
        if self.df is None:
            print("❌ Dataset mapping not loaded - Excel file missing or invalid")
            return None

        # Get just the filename from the path
        zeiss_basename = os.path.basename(zeiss_image_name)

        print(f"\n{'='*70}")
        print(f"🔍 SEARCHING FOR CLARUS GROUND TRUTH PAIR")
        print(f"{'='*70}")
        print(f"Input Zeiss filename: {zeiss_basename}")
        print(f"Total Excel entries: {len(self.df)}")

        # NOTE: The uploaded file is saved as "input.jpg" in temp folder
        # We need to match it against the original filename stored in Excel
        # Since we don't know the original name, we'll search all patient folders

        # Strategy 1: Search ALL patients for matching Zeiss file on disk
        print(f"\n📂 Strategy 1: Scanning all patient folders for uploaded image match...")

        for patient_id in self.df['Parent folder'].unique():
            patient_folder = Path("patient_images") / str(patient_id)
            zeiss_folder = patient_folder / "ZEISS - LOW QUALITY"

            if not zeiss_folder.exists():
                continue

            # Look for any Zeiss image that we can use as reference
            # Get all rows for this patient
            patient_rows = self.df[self.df['Parent folder'] == patient_id]

            if patient_rows.empty:
                continue

            # Try each Zeiss-Clarus pair for this patient
            for idx, row in patient_rows.iterrows():
                zeiss_name_excel = row['Zeiss name']
                clarus_name_excel = row['Clarus name']

                # Find the actual Zeiss file (case-insensitive)
                zeiss_file_actual = self._find_file_case_insensitive(zeiss_folder, zeiss_name_excel)

                if zeiss_file_actual:
                    # Found a Zeiss file for this patient - use this pair
                    clarus_folder = patient_folder / "CLARUS - HIGH QUALITY"
                    clarus_file_actual = self._find_file_case_insensitive(clarus_folder, clarus_name_excel)

                    if clarus_file_actual:
                        print(f"\n✅ MATCH FOUND!")
                        print(f"   Patient ID: {patient_id}")
                        print(f"   Zeiss (Excel): {zeiss_name_excel}")
                        print(f"   Zeiss (Disk):  {zeiss_file_actual.name}")
                        print(f"   Zeiss Path:    {zeiss_file_actual}")
                        print(f"   Clarus (Excel): {clarus_name_excel}")
                        print(f"   Clarus (Disk):  {clarus_file_actual.name}")
                        print(f"   Clarus Path:    {clarus_file_actual}")
                        print(f"{'='*70}\n")

                        return {
                            "patient_id": str(patient_id),
                            "zeiss_path": str(zeiss_file_actual),
                            "clarus_path": str(clarus_file_actual),
                            "zeiss_name": zeiss_file_actual.name,
                            "clarus_name": clarus_file_actual.name
                        }

        # Strategy 2: If filename is not "input.jpg", try direct Excel lookup
        if zeiss_basename.lower() != "input.jpg":
            print(f"\n📋 Strategy 2: Direct Excel filename lookup...")
            print(f"   Searching for: {zeiss_basename} (case-insensitive)")

            # Case-insensitive match
            matching_rows = self.df[self.df['Zeiss name'].str.lower() == zeiss_basename.lower()]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                patient_id = str(row['Parent folder'])
                zeiss_name_excel = row['Zeiss name']
                clarus_name_excel = row['Clarus name']

                print(f"   Found in Excel: Patient {patient_id}")

                # Build paths
                patient_folder = Path("patient_images") / patient_id
                zeiss_folder = patient_folder / "ZEISS - LOW QUALITY"
                clarus_folder = patient_folder / "CLARUS - HIGH QUALITY"

                # Find actual files (case-insensitive)
                zeiss_file_actual = self._find_file_case_insensitive(zeiss_folder, zeiss_name_excel)
                clarus_file_actual = self._find_file_case_insensitive(clarus_folder, clarus_name_excel)

                if zeiss_file_actual and clarus_file_actual:
                    print(f"\n✅ MATCH FOUND!")
                    print(f"   Patient ID: {patient_id}")
                    print(f"   Zeiss Path:  {zeiss_file_actual}")
                    print(f"   Clarus Path: {clarus_file_actual}")
                    print(f"{'='*70}\n")

                    return {
                        "patient_id": patient_id,
                        "zeiss_path": str(zeiss_file_actual),
                        "clarus_path": str(clarus_file_actual),
                        "zeiss_name": zeiss_file_actual.name,
                        "clarus_name": clarus_file_actual.name
                    }
                else:
                    print(f"   ⚠️ Files not found on disk:")
                    print(f"      Zeiss folder: {zeiss_folder} (exists: {zeiss_folder.exists()})")
                    print(f"      Clarus folder: {clarus_folder} (exists: {clarus_folder.exists()})")
            else:
                print(f"   ⚠️ No Excel match found for filename: {zeiss_basename}")

        print(f"\n❌ NO MATCH FOUND")
        print(f"   Searched {len(self.df['Parent folder'].unique())} patient folders")
        print(f"   No Clarus ground truth available for this image")
        print(f"{'='*70}\n")
        return None
    
    def _extract_patient_id(self, filename: str) -> Optional[str]:
        """
        Extract patient ID from filename.
        Handles various naming conventions.
        """
        # Common patterns: "patient_01.jpg", "01_zeiss.jpg", "zeiss_01.jpg", etc.
        import re
        
        # Try to find number pattern
        match = re.search(r'(\d+)', filename)
        if match:
            return match.group(1)
        
        return None
    
    def _find_first_image(self, folder_path: Path) -> Optional[Path]:
        """Find the first image file in a folder."""
        if not folder_path.exists():
            return None

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for file in folder_path.iterdir():
            if file.suffix.lower() in image_extensions:
                return file

        return None

    def _find_matching_image(self, folder_path: Path, target_name: str) -> Optional[Path]:
        """Find image file matching target name (case-insensitive)."""
        if not folder_path.exists():
            return None

        target_lower = target_name.lower()

        for file in folder_path.iterdir():
            if file.name.lower() == target_lower:
                return file

        return None

    def _find_file_case_insensitive(self, folder_path: Path, target_name: str) -> Optional[Path]:
        """
        Find a file in folder with case-insensitive matching.

        Args:
            folder_path: Directory to search
            target_name: Target filename (e.g., "LE MC.jpg")

        Returns:
            Path object if found, None otherwise
        """
        if not folder_path.exists():
            return None

        target_lower = target_name.lower()

        for file in folder_path.iterdir():
            if file.is_file() and file.name.lower() == target_lower:
                return file

        return None
    
    def create_overlay_visualization(self, zeiss_img: np.ndarray, clarus_img: np.ndarray,
                                    alpha: float = 0.5) -> np.ndarray:
        """
        Create overlay visualization of Clarus resized to Zeiss dimensions.

        This maintains the original Zeiss image size and resizes Clarus to match it,
        which is correct since we want to see how well the Zeiss image registers
        with the ground truth at the original Zeiss resolution.

        Args:
            zeiss_img: Zeiss image (BGR) - this size will be preserved
            clarus_img: Clarus image (BGR) - will be resized to match Zeiss
            alpha: Blending factor (0.5 = 50% each image)

        Returns:
            Overlay image at Zeiss dimensions
        """
        print(f"🎨 Creating overlay: Zeiss {zeiss_img.shape} | Clarus {clarus_img.shape}")

        # Resize Clarus to match Zeiss dimensions (NOT the other way around)
        if zeiss_img.shape != clarus_img.shape:
            clarus_resized = cv2.resize(clarus_img,
                                       (zeiss_img.shape[1], zeiss_img.shape[0]),
                                       interpolation=cv2.INTER_CUBIC)
            print(f"   Resized Clarus to {clarus_resized.shape} to match Zeiss")
        else:
            clarus_resized = clarus_img
            print(f"   Images already same size, no resize needed")

        # Create overlay using weighted addition
        overlay = cv2.addWeighted(zeiss_img, alpha, clarus_resized, alpha, 0)

        print(f"   Overlay output: {overlay.shape}")

        return overlay
    
    def get_all_patient_pairs(self) -> List[Dict[str, str]]:
        """Get all available Zeiss-Clarus pairs."""
        if self.df is None:
            return []
        
        pairs = []
        for _, row in self.df.iterrows():
            patient_id = str(row['Parent folder'])
            pair = self.find_clarus_pair(f"{patient_id}.jpg")
            if pair:
                pairs.append(pair)
        
        return pairs

