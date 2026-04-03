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
        
        Args:
            zeiss_image_name: Name of the Zeiss image file
        
        Returns:
            Dictionary with paths to Zeiss and Clarus images, or None if not found
        """
        if self.df is None:
            return None
        
        # Search for matching Zeiss image name in the dataframe
        zeiss_basename = os.path.basename(zeiss_image_name).lower()
        
        # Try to find patient folder from filename
        patient_id = self._extract_patient_id(zeiss_basename)
        if patient_id is None:
            return None
        
        # Look for this patient in the dataset
        patient_row = self.df[self.df['Parent folder'].astype(str) == str(patient_id)]
        
        if patient_row.empty:
            print(f"⚠️ No matching patient found for ID: {patient_id}")
            return None
        
        # Get paths
        patient_folder = Path("patient_images") / str(patient_id)
        zeiss_folder = patient_folder / "ZEISS - LOW QUALITY"
        clarus_folder = patient_folder / "CLARUS - HIGH QUALITY"
        
        # Find first image in each folder
        zeiss_path = self._find_first_image(zeiss_folder)
        clarus_path = self._find_first_image(clarus_folder)
        
        if zeiss_path and clarus_path:
            return {
                "patient_id": patient_id,
                "zeiss_path": str(zeiss_path),
                "clarus_path": str(clarus_path)
            }
        
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
    
    def create_overlay_visualization(self, zeiss_img: np.ndarray, clarus_img: np.ndarray, 
                                    alpha: float = 0.5) -> np.ndarray:
        """
        Create overlay visualization of Zeiss on Clarus (similar to fixed_trial4_updated.py).
        
        Args:
            zeiss_img: Zeiss image (BGR)
            clarus_img: Clarus image (BGR)
            alpha: Blending factor (0.5 = 50% each image)
        
        Returns:
            Overlay image
        """
        # Resize Zeiss to match Clarus dimensions
        if zeiss_img.shape != clarus_img.shape:
            zeiss_resized = cv2.resize(zeiss_img, 
                                      (clarus_img.shape[1], clarus_img.shape[0]), 
                                      interpolation=cv2.INTER_CUBIC)
        else:
            zeiss_resized = zeiss_img
        
        # Create overlay using weighted addition
        overlay = cv2.addWeighted(clarus_img, alpha, zeiss_resized, alpha, 0)
        
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

