"""Metrics service for calculating image quality metrics."""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


class MetricsService:
    """Service for calculating image quality metrics."""
    
    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        Higher is better (typically 20-50 dB for images).
        """
        # Resize images to same size if needed
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return float(psnr)
    
    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        Range [0, 1], higher is better.
        """
        from skimage.metrics import structural_similarity as ssim
        
        # Resize images to same size if needed
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale if color
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            
        if len(img2.shape) == 3:
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = img2
        
        score = ssim(img1_gray, img2_gray, data_range=255)
        
        return float(score)
    
    @staticmethod
    def calculate_vessel_recovery(original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Calculate vessel structure preservation ratio.
        
        Extracts vessels from both images and compares overlap.
        Range [0, 1], higher is better.
        """
        def extract_vessels(img):
            """Extract vessels using classical method."""
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Black-hat morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
            
            # Threshold
            _, vessels = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return vessels
        
        # Resize enhanced to match original for comparison
        if original.shape != enhanced.shape:
            enhanced_resized = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
        else:
            enhanced_resized = enhanced
        
        # Extract vessels
        vessels_orig = extract_vessels(original)
        vessels_enhanced = extract_vessels(enhanced_resized)
        
        # Calculate overlap (Dice coefficient)
        intersection = np.logical_and(vessels_orig > 0, vessels_enhanced > 0).sum()
        union = (vessels_orig > 0).sum() + (vessels_enhanced > 0).sum()
        
        if union == 0:
            return 0.0
        
        dice = 2.0 * intersection / union
        
        return float(dice)
    
    @staticmethod
    def calculate_sharpness(img: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        Higher values indicate sharper images.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return float(sharpness)
    
    @staticmethod
    def calculate_contrast(img: np.ndarray) -> float:
        """
        Calculate image contrast (standard deviation of pixel intensities).
        Higher values indicate more contrast.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        contrast = gray.std()
        
        return float(contrast)
    
    @staticmethod
    def compare_images(img1_path: str, img2_path: str) -> Dict[str, Any]:
        """
        Comprehensive comparison between two images.
        
        Args:
            img1_path: Path to first image (e.g., original)
            img2_path: Path to second image (e.g., enhanced)
        
        Returns:
            Dictionary with all metrics
        """
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("Failed to load one or both images")
        
        metrics = {
            "psnr": MetricsService.calculate_psnr(img1, img2),
            "ssim": MetricsService.calculate_ssim(img1, img2),
            "vessel_recovery": MetricsService.calculate_vessel_recovery(img1, img2),
            "sharpness_original": MetricsService.calculate_sharpness(img1),
            "sharpness_enhanced": MetricsService.calculate_sharpness(img2),
            "contrast_original": MetricsService.calculate_contrast(img1),
            "contrast_enhanced": MetricsService.calculate_contrast(img2),
            "resolution_improvement": f"{img2.shape[0] / img1.shape[0]:.2f}x",
            "size_original": f"{img1.shape[1]}x{img1.shape[0]}",
            "size_enhanced": f"{img2.shape[1]}x{img2.shape[0]}"
        }
        
        return metrics

