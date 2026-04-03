"""Registration service for aligning and FOV extension."""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path


class RegistrationService:
    """Service for image registration and FOV extension."""
    
    @staticmethod
    def detect_and_match_features(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect features and match between two images.
        
        Returns:
            homography: 3x3 transformation matrix
            num_matches: Number of good matches found
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Use ORB detector (SIFT requires opencv-contrib)
        detector = cv2.ORB_create(nfeatures=5000)
        
        # Detect keypoints and descriptors
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            raise ValueError("Failed to detect features in one or both images")
        
        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Keep top matches
        num_good_matches = min(len(matches), 100)
        good_matches = matches[:num_good_matches]
        
        if len(good_matches) < 4:
            raise ValueError(f"Not enough matches found: {len(good_matches)}")
        
        # Extract matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        return H, len(good_matches)
    
    @staticmethod
    def warp_image(img: np.ndarray, H: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Warp image using homography matrix.
        
        Args:
            img: Input image
            H: 3x3 homography matrix
            output_shape: (height, width) of output
        
        Returns:
            Warped image
        """
        warped = cv2.warpPerspective(img, H, (output_shape[1], output_shape[0]))
        return warped
    
    @staticmethod
    def create_alpha_mask(footprint_mask: np.ndarray, sigma: float = 40.0) -> np.ndarray:
        """
        Create smooth alpha blending mask using Gaussian blur.

        Implements Methodology Section 7.4.2 Step 2:
        Apply Gaussian blur to binary footprint to create smooth gradient at boundary.

        Args:
            footprint_mask: Binary mask (0 or 255) indicating foreground region
            sigma: Gaussian blur sigma (controls blend zone width)
                  Recommended: 30-50 pixels for medical images
                  Larger sigma = wider, smoother transition
                  Smaller sigma = sharper transition, may show seams

        Returns:
            Alpha mask (0-1 range, single channel float32)
        """
        # Ensure mask is single channel uint8
        if len(footprint_mask.shape) == 3:
            footprint_mask = cv2.cvtColor(footprint_mask, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to create smooth transition
        # Convert to float32 and normalize to [0, 1]
        alpha = cv2.GaussianBlur(
            footprint_mask.astype(np.float32) / 255.0,
            (0, 0),  # Kernel size auto-calculated from sigma
            sigmaX=sigma
        )

        return alpha

    @staticmethod
    def gaussian_alpha_blend(
        background: np.ndarray,
        foreground: np.ndarray,
        sigma: float = 40.0,
        mask: Optional[np.ndarray] = None,
        return_alpha: bool = False
    ) -> np.ndarray:
        """
        Blend two images using Gaussian alpha blending for seamless FOV extension.

        Implements Methodology Section 7.4.2:
        Creates smooth transition zone between SR-enhanced narrow FOV and wide FOV images
        to eliminate visible seams due to intensity/color differences.

        Args:
            background: Background image (e.g., Clarus - wide FOV)
            foreground: Foreground image (e.g., SR-enhanced Zeiss - narrow FOV)
            sigma: Gaussian blur sigma for blend zone width (default: 40 pixels)
            mask: Optional binary mask. If None, auto-computed from foreground content
            return_alpha: If True, return (blended, alpha_mask) tuple

        Returns:
            Blended image (or tuple if return_alpha=True)
        """
        # Ensure same size
        if background.shape != foreground.shape:
            foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

        # Step 1: Compute footprint mask (Methodology 7.4.2 Step 1)
        if mask is None:
            # Convert to grayscale if needed
            if len(foreground.shape) == 3:
                gray_fg = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
            else:
                gray_fg = foreground

            # Create binary mask where foreground has content (threshold at 10)
            footprint_mask = (gray_fg > 10).astype(np.uint8) * 255
        else:
            footprint_mask = mask

        # Step 2: Create alpha blending mask (Methodology 7.4.2 Step 2)
        alpha = RegistrationService.create_alpha_mask(footprint_mask, sigma)

        # Step 3: Expand alpha to 3 channels (Methodology 7.4.2 Step 3)
        if len(background.shape) == 3:
            alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
        else:
            alpha_3ch = alpha

        # Step 4: Composite final image (Methodology 7.4.2 Step 4)
        # composite = alpha * foreground + (1 - alpha) * background
        # Inside footprint (alpha → 1.0): foreground dominates
        # Outside footprint (alpha = 0.0): background unchanged
        # Transition zone: smooth blend
        composite = alpha_3ch * foreground.astype(np.float32) + \
                   (1.0 - alpha_3ch) * background.astype(np.float32)
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        if return_alpha:
            return composite, alpha
        return composite

    @staticmethod
    def gaussian_blend(img1: np.ndarray, img2: np.ndarray, mask: Optional[np.ndarray] = None, sigma: float = 40.0) -> np.ndarray:
        """
        Blend two images using Gaussian alpha blending.

        Legacy wrapper for gaussian_alpha_blend.

        Args:
            img1: First image (background)
            img2: Second image (foreground)
            mask: Optional mask for blending regions
            sigma: Gaussian blur sigma (default: 40 pixels)

        Returns:
            Blended image
        """
        return RegistrationService.gaussian_alpha_blend(img1, img2, sigma=sigma, mask=mask)
    
    @staticmethod
    def extend_fov(
        narrow_img: np.ndarray,
        wide_img: np.ndarray,
        sigma: float = 40.0,
        return_diagnostics: bool = False
    ) -> Dict[str, Any]:
        """
        Extend field of view by registering narrow FOV to wide FOV with Gaussian blending.

        Implements Methodology Section 7.4 (FOV Extension and Gaussian Alpha Blending):
        Composites SR-enhanced narrow FOV onto full wide FOV canvas with seamless
        boundary blending.

        Args:
            narrow_img: Narrow FOV image (e.g., SR-enhanced Zeiss)
            wide_img: Wide FOV image (e.g., Clarus)
            sigma: Gaussian blur sigma for blend zone (default: 40 pixels)
                  Range 20-50 recommended. See config.yaml blending.sigma_candidates
            return_diagnostics: If True, include alpha mask and overlap metrics

        Returns:
            Dictionary with registration results:
                - success: bool
                - homography: 3x3 matrix as list
                - num_matches: int (feature matches found)
                - warped_image: Warped narrow image aligned to wide canvas
                - blended_image: Final composite with FOV extension
                - alpha_mask: (optional) Alpha blending mask
                - overlap_fraction: (optional) Fraction of canvas with narrow FOV content
        """
        try:
            # Detect and match features
            H, num_matches = RegistrationService.detect_and_match_features(narrow_img, wide_img)

            # Warp narrow image to align with wide image
            warped_narrow = RegistrationService.warp_image(
                narrow_img,
                H,
                (wide_img.shape[0], wide_img.shape[1])
            )

            # Blend images with Gaussian alpha blending
            blended, alpha = RegistrationService.gaussian_alpha_blend(
                wide_img,
                warped_narrow,
                sigma=sigma,
                return_alpha=True
            )

            result = {
                "success": True,
                "homography": H.tolist(),
                "num_matches": num_matches,
                "warped_image": warped_narrow,
                "blended_image": blended
            }

            # Add diagnostics if requested
            if return_diagnostics:
                # Compute overlap fraction
                if len(warped_narrow.shape) == 3:
                    gray_warped = cv2.cvtColor(warped_narrow, cv2.COLOR_BGR2GRAY)
                else:
                    gray_warped = warped_narrow

                overlap_pixels = np.count_nonzero(gray_warped > 10)
                total_pixels = gray_warped.shape[0] * gray_warped.shape[1]
                overlap_fraction = overlap_pixels / total_pixels

                result["alpha_mask"] = alpha
                result["overlap_fraction"] = float(overlap_fraction)
                result["blend_sigma"] = sigma

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "num_matches": 0
            }
    
    @staticmethod
    def tune_sigma(
        narrow_img: np.ndarray,
        wide_img: np.ndarray,
        sigma_candidates: list = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test multiple sigma values for Gaussian blending to find optimal transition.

        Implements Methodology Section 7.4.2 Step 6:
        Run blending on test pairs with multiple sigma values to select
        the one producing most visually seamless boundaries.

        Args:
            narrow_img: Narrow FOV image
            wide_img: Wide FOV image
            sigma_candidates: List of sigma values to test (default: [20, 30, 40, 50])
            output_dir: Optional directory to save comparison images

        Returns:
            Dictionary with results for each sigma value
        """
        if sigma_candidates is None:
            sigma_candidates = [20, 30, 40, 50]

        results = {}

        for sigma in sigma_candidates:
            result = RegistrationService.extend_fov(
                narrow_img,
                wide_img,
                sigma=sigma,
                return_diagnostics=True
            )

            if result["success"]:
                results[f"sigma_{sigma}"] = {
                    "sigma": sigma,
                    "blended_image": result["blended_image"],
                    "alpha_mask": result["alpha_mask"],
                    "overlap_fraction": result["overlap_fraction"]
                }

                # Save if output directory provided
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    # Save blended result
                    blend_path = output_path / f"blended_sigma_{sigma}.png"
                    cv2.imwrite(str(blend_path), result["blended_image"])

                    # Save alpha mask as heatmap
                    alpha_heatmap = (result["alpha_mask"] * 255).astype(np.uint8)
                    alpha_colored = cv2.applyColorMap(alpha_heatmap, cv2.COLORMAP_JET)
                    alpha_path = output_path / f"alpha_sigma_{sigma}.png"
                    cv2.imwrite(str(alpha_path), alpha_colored)

        return results

    @staticmethod
    def register_images(
        img1_path: str,
        img2_path: str,
        output_dir: str,
        sigma: float = 40.0,
        save_diagnostics: bool = True
    ) -> Dict[str, Any]:
        """
        Register two images and save results with FOV extension.

        Args:
            img1_path: Path to first image (narrow FOV)
            img2_path: Path to second image (wide FOV)
            output_dir: Directory to save results
            sigma: Gaussian blur sigma for blending (default: 40)
            save_diagnostics: Whether to save alpha mask visualization

        Returns:
            Registration results with paths to saved images
        """
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            raise ValueError("Failed to load one or both images")

        # Perform registration with FOV extension
        result = RegistrationService.extend_fov(
            img1,
            img2,
            sigma=sigma,
            return_diagnostics=save_diagnostics
        )

        if result["success"]:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save warped image
            warped_path = output_path / "warped.png"
            cv2.imwrite(str(warped_path), result["warped_image"])

            # Save blended image (FOV-extended composite)
            blended_path = output_path / "composite_enhanced.png"
            cv2.imwrite(str(blended_path), result["blended_image"])

            result["warped_path"] = str(warped_path)
            result["blended_path"] = str(blended_path)
            result["composite_path"] = str(blended_path)  # Alias for clarity

            # Save diagnostics if requested
            if save_diagnostics and "alpha_mask" in result:
                # Save alpha mask as grayscale
                alpha_path = output_path / "alpha_mask.png"
                alpha_img = (result["alpha_mask"] * 255).astype(np.uint8)
                cv2.imwrite(str(alpha_path), alpha_img)

                # Save alpha mask as heatmap overlay
                alpha_heatmap = cv2.applyColorMap(alpha_img, cv2.COLORMAP_JET)
                alpha_overlay = cv2.addWeighted(result["blended_image"], 0.7, alpha_heatmap, 0.3, 0)
                heatmap_path = output_path / "alpha_heatmap.png"
                cv2.imwrite(str(heatmap_path), alpha_overlay)

                result["alpha_path"] = str(alpha_path)
                result["alpha_heatmap_path"] = str(heatmap_path)

        return result

