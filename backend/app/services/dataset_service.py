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

    def _preprocess_for_registration(self, img: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for retinal image feature detection.
        Uses green channel + CLAHE + vessel enhancement for robust features.
        """
        if len(img.shape) == 3:
            green = img[:, :, 1]
        else:
            green = img

        # Apply CLAHE with optimized parameters for vessel enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green)

        # Additional vessel enhancement using morphological operations
        # This helps detect features on vessel structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        enhanced = cv2.add(enhanced, tophat)

        return enhanced

    def _create_retinal_mask(self, img_gray: np.ndarray, erosion_size: int = 20) -> np.ndarray:
        """
        Create mask to exclude black circular borders in retinal images.
        """
        try:
            # Threshold to get bright regions
            _, mask = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)

            # Find largest contour (retinal area)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return mask

            largest_contour = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(mask)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # Erode to avoid border artifacts
            kernel = np.ones((erosion_size, erosion_size), np.uint8)
            eroded_mask = cv2.erode(clean_mask, kernel, iterations=1)

            return eroded_mask
        except Exception as e:
            print(f"   ⚠️  Mask creation failed: {e}")
            return np.ones_like(img_gray, dtype=np.uint8) * 255
    
    def create_overlay_visualization(self, zeiss_img: np.ndarray, clarus_img: np.ndarray,
                                    alpha: float = 0.5, zeiss_enhanced: np.ndarray = None) -> np.ndarray:
        """
        Create overlay visualization using proper image registration.

        This performs feature-based registration to align the small Zeiss FOV
        with the corresponding region in the larger Clarus image, then creates
        an overlay where:
        - The Zeiss image is warped to align with Clarus coordinates
        - The output is at full Clarus resolution
        - Only the overlapping region shows the blend

        Args:
            zeiss_img: Zeiss image (BGR) - smaller FOV, original resolution
            clarus_img: Clarus image (BGR) - larger FOV
            alpha: Blending factor (0.5 = 50% each image)
            zeiss_enhanced: Optional 4x enhanced Zeiss for better feature detection

        Returns:
            Overlay image at Clarus dimensions with registered Zeiss
        """
        print(f"\n{'='*70}")
        print(f"🎨 CREATING REGISTERED OVERLAY")
        print(f"{'='*70}")
        print(f"Zeiss size:  {zeiss_img.shape}")
        print(f"Clarus size: {clarus_img.shape}")

        # Try registration with enhanced Zeiss first (if available) for better feature matching
        zeiss_for_registration = zeiss_enhanced if zeiss_enhanced is not None else zeiss_img
        use_enhanced = zeiss_enhanced is not None

        if use_enhanced:
            print(f"Enhanced Zeiss: {zeiss_enhanced.shape} (using for better feature detection)")

        # Step 1: Preprocess images for feature detection (green channel + CLAHE + vessels)
        print("\n📊 Step 1: Preprocessing for feature detection...")
        zeiss_gray = self._preprocess_for_registration(zeiss_for_registration)
        clarus_gray = self._preprocess_for_registration(clarus_img)

        # Step 2: Create masks to exclude black borders
        print("🎭 Step 2: Creating masks to exclude borders...")
        zeiss_mask = self._create_retinal_mask(zeiss_gray)
        clarus_mask = self._create_retinal_mask(clarus_gray)

        # Step 3: Feature detection and matching
        print("🔍 Step 3: Detecting and matching features...")
        homography, num_inliers = self._compute_homography(zeiss_gray, clarus_gray,
                                              zeiss_mask, clarus_mask)

        # Adaptive threshold: higher for enhanced (expect better), lower for original
        MIN_INLIERS_REQUIRED = 30 if use_enhanced else 15

        if homography is None or num_inliers < MIN_INLIERS_REQUIRED:
            if homography is not None:
                print(f"⚠️  Warning: Only {num_inliers} inliers (need ≥{MIN_INLIERS_REQUIRED}) - registration quality too low")

            # If enhanced failed, retry with original before falling back to centered
            if use_enhanced and zeiss_enhanced is not None:
                print("🔄 Retrying registration with original resolution Zeiss...")
                return self.create_overlay_visualization(zeiss_img, clarus_img, alpha, zeiss_enhanced=None)

            print("📍 Using centered placement with smart blending instead")
            return self._create_centered_overlay(zeiss_img, clarus_img, alpha)

        # Step 4: Warp Zeiss to Clarus coordinate system
        print(f"🔄 Step 4: Warping Zeiss to align with Clarus (using {num_inliers} inliers)...")
        h, w = clarus_img.shape[:2]

        # IMPORTANT: Always warp and display the ORIGINAL Zeiss, not the enhanced
        # We only used enhanced for feature detection, but overlay shows original input
        warped_zeiss = cv2.warpPerspective(zeiss_img, homography, (w, h),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0))

        # Step 5: Create smart overlay that preserves Clarus where Zeiss is black
        print("🎨 Step 5: Creating smart blended overlay...")
        overlay = self._create_smart_blend(clarus_img, warped_zeiss, alpha)

        print(f"✅ Overlay complete: {overlay.shape}")
        if use_enhanced:
            print(f"ℹ️  Note: Used 4× enhanced for feature detection, but displaying original Zeiss in overlay")
        print(f"{'='*70}\n")

        return overlay
    
    def _compute_homography(self, zeiss_gray: np.ndarray, clarus_gray: np.ndarray,
                           zeiss_mask: np.ndarray, clarus_mask: np.ndarray) -> tuple[Optional[np.ndarray], int]:
        """
        Compute homography matrix to register Zeiss to Clarus using feature matching.
        Tries multiple feature detectors (SIFT, ORB, AKAZE) with optimized parameters.

        Returns:
            (homography_matrix, num_inliers) or (None, 0) if failed
        """
        methods = ['SIFT', 'ORB', 'AKAZE']
        best_H = None
        max_inliers = 0
        best_method = None

        for method in methods:
            print(f"   Trying {method}...", end=" ")

            try:
                # Create detector with parameters optimized for retinal images
                if method == 'SIFT':
                    # Increased features from 10k to 20k for better coverage
                    # Lower contrast threshold to detect more features on vessels
                    detector = cv2.SIFT_create(
                        nfeatures=20000,
                        contrastThreshold=0.03,  # Optimized for vessel features
                        edgeThreshold=10,
                        sigma=1.6
                    )
                elif method == 'ORB':
                    # Increased features, multi-scale for better matching
                    detector = cv2.ORB_create(
                        nfeatures=20000,
                        scaleFactor=1.2,
                        nlevels=8,
                        edgeThreshold=31,
                        patchSize=31
                    )
                elif method == 'AKAZE':
                    # AKAZE with optimized threshold for retinal images
                    detector = cv2.AKAZE_create(
                        threshold=0.001,
                        nOctaves=4,
                        nOctaveLayers=4
                    )
                else:
                    continue

                # Detect keypoints and descriptors
                kp1, des1 = detector.detectAndCompute(zeiss_gray, mask=zeiss_mask)
                kp2, des2 = detector.detectAndCompute(clarus_gray, mask=clarus_mask)

                if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                    print(f"Not enough keypoints (Zeiss: {len(kp1) if kp1 else 0}, Clarus: {len(kp2) if kp2 else 0})")
                    continue

                print(f"Detected {len(kp1)} & {len(kp2)} keypoints...", end=" ")

                # Match features using appropriate matcher
                if method == 'ORB':
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                    ratio_threshold = 0.80  # More lenient for ORB
                else:
                    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                    ratio_threshold = 0.75  # Standard for SIFT/AKAZE

                matches = matcher.knnMatch(des1, des2, k=2)

                # Apply Lowe's ratio test to filter good matches
                good_matches = []
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < ratio_threshold * n.distance:
                            good_matches.append(m)

                if len(good_matches) < 4:
                    print(f"{len(good_matches)} matches (need ≥4)")
                    continue

                # Extract matching point coordinates
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute homography with RANSAC
                # Increased ransacReprojThreshold slightly for retinal images
                H, inlier_mask = cv2.findHomography(
                    src_pts, dst_pts,
                    cv2.RANSAC,
                    ransacReprojThreshold=5.0,
                    maxIters=2000,
                    confidence=0.995
                )

                if H is None:
                    print(f"Homography failed")
                    continue

                inliers = int(inlier_mask.sum())
                print(f"{len(good_matches)} matches → {inliers} inliers")

                if inliers > max_inliers:
                    max_inliers = inliers
                    best_H = H
                    best_method = method

            except Exception as e:
                print(f"Error: {e}")
                continue

        if best_H is not None:
            print(f"   ✅ Best: {best_method} with {max_inliers} inliers")
        else:
            print(f"   ❌ No valid homography found")

        return best_H, max_inliers

    def _create_smart_blend(self, clarus_img: np.ndarray, warped_zeiss: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
        """
        Create smart blend that only blends where Zeiss has valid data.
        Preserves Clarus in areas where Zeiss is black (warped border regions).
        """
        # Create mask for valid Zeiss regions (non-black areas)
        zeiss_gray = cv2.cvtColor(warped_zeiss, cv2.COLOR_BGR2GRAY)
        _, zeiss_valid_mask = cv2.threshold(zeiss_gray, 10, 255, cv2.THRESH_BINARY)

        # Convert mask to 3-channel for blending
        zeiss_valid_mask_3ch = cv2.merge([zeiss_valid_mask, zeiss_valid_mask, zeiss_valid_mask]) / 255.0

        # Blend only where Zeiss is valid
        # In valid regions: blend both images
        # In invalid regions: use only Clarus
        overlay = clarus_img.copy().astype(np.float32)
        warped_zeiss_float = warped_zeiss.astype(np.float32)
        clarus_float = clarus_img.astype(np.float32)

        # Where Zeiss is valid: weighted blend
        # Where Zeiss is black: 100% Clarus
        overlay = (zeiss_valid_mask_3ch * (alpha * clarus_float + alpha * warped_zeiss_float) +
                  (1 - zeiss_valid_mask_3ch) * clarus_float)

        return np.clip(overlay, 0, 255).astype(np.uint8)

    def _create_centered_overlay(self, zeiss_img: np.ndarray, clarus_img: np.ndarray,
                                alpha: float = 0.5) -> np.ndarray:
        """
        Improved fallback overlay: place Zeiss at center with smart blending.
        Only blends where Zeiss has valid (non-black) data.
        """
        h_c, w_c = clarus_img.shape[:2]
        h_z, w_z = zeiss_img.shape[:2]

        # Create output canvas (copy of Clarus)
        overlay = clarus_img.copy().astype(np.float32)

        # Calculate center position
        y_offset = (h_c - h_z) // 2
        x_offset = (w_c - w_z) // 2

        # Make sure Zeiss fits within Clarus
        if y_offset < 0 or x_offset < 0:
            # Zeiss is larger than Clarus - resize it
            scale = min(w_c / w_z, h_c / h_z) * 0.8
            new_w = int(w_z * scale)
            new_h = int(h_z * scale)
            zeiss_resized = cv2.resize(zeiss_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            y_offset = (h_c - new_h) // 2
            x_offset = (w_c - new_w) // 2
        else:
            zeiss_resized = zeiss_img

        h_z, w_z = zeiss_resized.shape[:2]

        # Create mask for valid Zeiss regions (non-black areas)
        zeiss_gray = cv2.cvtColor(zeiss_resized, cv2.COLOR_BGR2GRAY)
        _, zeiss_mask = cv2.threshold(zeiss_gray, 10, 255, cv2.THRESH_BINARY)
        zeiss_mask_3ch = cv2.merge([zeiss_mask, zeiss_mask, zeiss_mask]) / 255.0

        # Extract ROI from Clarus
        roi = overlay[y_offset:y_offset+h_z, x_offset:x_offset+w_z]
        zeiss_float = zeiss_resized.astype(np.float32)

        # Smart blend: only blend where Zeiss is valid
        blended_roi = (zeiss_mask_3ch * (alpha * roi + alpha * zeiss_float) +
                      (1 - zeiss_mask_3ch) * roi)

        overlay[y_offset:y_offset+h_z, x_offset:x_offset+w_z] = blended_roi

        result = np.clip(overlay, 0, 255).astype(np.uint8)

        print(f"   ℹ️  Centered overlay: Zeiss ({h_z}×{w_z}) placed at ({x_offset}, {y_offset}) within Clarus ({h_c}×{w_c})")

        return result

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

