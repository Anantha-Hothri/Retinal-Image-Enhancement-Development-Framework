"""Image preprocessing utilities including DCP dehazing and CLAHE enhancement."""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """Handles preprocessing for retinal images including dehazing and enhancement."""
    
    def __init__(self, config: dict = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary with DCP and CLAHE parameters
        """
        if config is None:
            config = {}
        
        # DCP parameters
        self.dcp_patch_size = config.get('dcp_patch_size', 15)
        self.dcp_omega = config.get('dcp_omega', 0.95)
        self.dcp_atm_percentile = config.get('dcp_atm_percentile', 0.1)
        self.dcp_guided_radius = config.get('dcp_guided_radius', 60)
        self.dcp_guided_eps = config.get('dcp_guided_eps', 0.001)
        self.dcp_t0 = config.get('dcp_t0', 0.1)
        
        # CLAHE parameters
        self.clahe_clip_limit = config.get('clahe_clip_limit', 2.0)
        self.clahe_tile_grid = tuple(config.get('clahe_tile_grid', [8, 8]))
    
    def compute_dark_channel(self, image: np.ndarray, patch_size: int = 15) -> np.ndarray:
        """
        Compute dark channel of an image.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            patch_size: Size of local patch (must be odd)
            
        Returns:
            Dark channel image (H, W)
        """
        if len(image.shape) == 3:
            # For multi-channel, take minimum across channels first
            min_channel = np.min(image, axis=2)
        else:
            min_channel = image
        
        # Compute minimum in local patch
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def estimate_atmospheric_light(self, image: np.ndarray, dark_channel: np.ndarray,
                                   percentile: float = 0.1) -> float:
        """
        Estimate atmospheric light from dark channel.
        
        Args:
            image: Original image (can be single or multi-channel)
            dark_channel: Dark channel of the image
            percentile: Percentage of brightest pixels to use (0.1 = top 0.1%)
            
        Returns:
            Estimated atmospheric light value
        """
        h, w = dark_channel.shape
        num_pixels = int(h * w * percentile / 100)
        num_pixels = max(num_pixels, 1)
        
        # Get indices of brightest pixels in dark channel
        flat_dark = dark_channel.flatten()
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        # Get corresponding pixel values from original image
        if len(image.shape) == 3:
            # For LAB, use L channel
            flat_image = image[:, :, 0].flatten()
        else:
            flat_image = image.flatten()
        
        atmospheric_light = np.mean(flat_image[indices])
        
        return atmospheric_light
    
    def compute_transmission(self, image: np.ndarray, atmospheric_light: float,
                            omega: float = 0.95, patch_size: int = 15) -> np.ndarray:
        """
        Compute transmission map.
        
        Args:
            image: Normalized image (0-1 range)
            atmospheric_light: Estimated atmospheric light
            omega: Dehazing strength parameter
            patch_size: Patch size for dark channel
            
        Returns:
            Transmission map
        """
        # Normalize by atmospheric light
        norm_image = image / (atmospheric_light + 1e-8)
        
        # Compute dark channel of normalized image
        dark_channel = self.compute_dark_channel(norm_image, patch_size)
        
        # Compute transmission
        transmission = 1.0 - omega * dark_channel
        
        return transmission
    
    def refine_transmission_guided_filter(self, transmission: np.ndarray, 
                                         guide: np.ndarray,
                                         radius: int = 60, 
                                         eps: float = 0.001) -> np.ndarray:
        """
        Refine transmission map using guided filter.
        
        Args:
            transmission: Raw transmission map
            guide: Guide image (typically the L channel)
            radius: Filter radius
            eps: Regularization parameter
            
        Returns:
            Refined transmission map
        """
        # Ensure images are float32
        transmission = transmission.astype(np.float32)
        guide = guide.astype(np.float32)
        
        # Use OpenCV's ximgproc guided filter if available, otherwise simple box filter
        try:
            refined = cv2.ximgproc.guidedFilter(guide, transmission, radius, eps)
        except AttributeError:
            # Fallback to simple box filter if ximgproc not available
            refined = cv2.boxFilter(transmission, -1, (radius, radius))
        
        return refined
    
    def recover_scene(self, image: np.ndarray, transmission: np.ndarray,
                     atmospheric_light: float, t0: float = 0.1) -> np.ndarray:
        """
        Recover dehazed scene using transmission map.
        
        Args:
            image: Hazy image
            transmission: Transmission map
            atmospheric_light: Atmospheric light value
            t0: Minimum transmission threshold
            
        Returns:
            Dehazed image
        """
        # Clip transmission to avoid division by very small values
        transmission_clip = np.maximum(transmission, t0)
        
        # Expand transmission to match image dimensions if needed
        if len(image.shape) == 3 and len(transmission_clip.shape) == 2:
            transmission_clip = transmission_clip[:, :, np.newaxis]
        
        # Recover scene: J(x) = (I(x) - A) / t(x) + A
        recovered = (image - atmospheric_light) / transmission_clip + atmospheric_light
        
        # Clip to valid range
        recovered = np.clip(recovered, 0, 255).astype(np.uint8)
        
        return recovered

    def apply_dcp_dehazing(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Apply Dark Channel Prior dehazing in LAB color space (for Zeiss images).

        This implements the full DCP pipeline as specified in Methodology Section 2.2:
        1. Convert BGR to LAB
        2. Compute dark channel of L channel
        3. Estimate atmospheric light
        4. Compute transmission map
        5. Refine transmission with guided filter
        6. Recover dehazed L channel
        7. Convert back to BGR

        Args:
            image_bgr: Input BGR image (Zeiss Visuscout)

        Returns:
            Dehazed BGR image
        """
        # Step 1: Convert to LAB color space
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # Work with L channel (luminance)
        L_normalized = L.astype(np.float32) / 255.0

        # Step 2: Compute dark channel
        dark_channel = self.compute_dark_channel(L_normalized, self.dcp_patch_size)

        # Step 3: Estimate atmospheric light
        atmospheric_light = self.estimate_atmospheric_light(
            lab, dark_channel, self.dcp_atm_percentile
        )

        # Step 4: Compute transmission map
        transmission = self.compute_transmission(
            L_normalized, atmospheric_light / 255.0, self.dcp_omega, self.dcp_patch_size
        )

        # Step 5: Refine transmission with guided filter
        transmission_refined = self.refine_transmission_guided_filter(
            transmission, L, self.dcp_guided_radius, self.dcp_guided_eps
        )

        # Step 6: Recover dehazed L channel
        L_dehazed = self.recover_scene(
            L, transmission_refined, atmospheric_light, self.dcp_t0
        )

        # Step 7: Merge back and convert to BGR
        lab_dehazed = cv2.merge([L_dehazed, A, B])
        bgr_dehazed = cv2.cvtColor(lab_dehazed, cv2.COLOR_LAB2BGR)

        return bgr_dehazed

    def apply_clahe(self, image_gray: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image_gray: Grayscale image (single channel)

        Returns:
            CLAHE-enhanced image
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid
        )
        enhanced = clahe.apply(image_gray)
        return enhanced

    def preprocess_image_for_detection(self, image_bgr: np.ndarray,
                                      apply_dehazing: bool = False) -> np.ndarray:
        """
        Preprocess image for feature detection.

        This implements the pipeline from Methodology Section 2:
        - Extract green channel (best contrast for retinal vessels)
        - Apply DCP dehazing (optional, for Zeiss images only)
        - Apply CLAHE enhancement

        Args:
            image_bgr: Input BGR image
            apply_dehazing: If True, apply DCP dehazing before other steps

        Returns:
            Enhanced grayscale image suitable for feature detection
        """
        # Optional: Apply DCP dehazing for Zeiss images
        if apply_dehazing:
            image_bgr = self.apply_dcp_dehazing(image_bgr)

        # Extract green channel (highest vessel contrast)
        green_channel = image_bgr[:, :, 1]

        # Apply CLAHE
        enhanced = self.apply_clahe(green_channel)

        return enhanced

    def create_robust_mask(self, image_gray: np.ndarray,
                          threshold: int = 10,
                          erosion_size: int = 20,
                          closing_kernel_size: int = 25) -> np.ndarray:
        """
        Create robust fundus mask to exclude black circular border.

        Implements the masking strategy from Methodology Section 3:
        1. Threshold to separate fundus from black border
        2. Find largest contour
        3. Morphological closing to fill gaps
        4. Erosion to exclude unreliable boundary region

        Args:
            image_gray: Grayscale image
            threshold: Binary threshold value
            erosion_size: Number of pixels to erode from boundary
            closing_kernel_size: Kernel size for morphological closing

        Returns:
            Binary fundus mask (255 = valid fundus, 0 = exclude)
        """
        # Step 1: Threshold
        _, mask = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY)

        # Step 2: Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return mask

        largest_contour = max(contours, key=cv2.contourArea)

        # Fill largest contour on blank mask
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Step 3: Morphological closing
        closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (closing_kernel_size, closing_kernel_size)
        )
        mask_closed = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, closing_kernel)

        # Step 4: Erosion
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * erosion_size + 1, 2 * erosion_size + 1)
        )
        mask_eroded = cv2.erode(mask_closed, erosion_kernel)

        return mask_eroded


def preprocess_zeiss_image(image_bgr: np.ndarray, config: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to preprocess Zeiss Visuscout image.

    Args:
        image_bgr: Input Zeiss BGR image
        config: Configuration dictionary

    Returns:
        Tuple of (enhanced_grayscale, fundus_mask)
    """
    preprocessor = ImagePreprocessor(config)

    # Apply dehazing + CLAHE
    enhanced = preprocessor.preprocess_image_for_detection(image_bgr, apply_dehazing=True)

    # Create mask
    mask = preprocessor.create_robust_mask(enhanced)

    return enhanced, mask


def preprocess_clarus_image(image_bgr: np.ndarray, config: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to preprocess Clarus image.

    Args:
        image_bgr: Input Clarus BGR image
        config: Configuration dictionary

    Returns:
        Tuple of (enhanced_grayscale, fundus_mask)
    """
    preprocessor = ImagePreprocessor(config)

    # No dehazing for Clarus images
    enhanced = preprocessor.preprocess_image_for_detection(image_bgr, apply_dehazing=False)

    # Create mask
    mask = preprocessor.create_robust_mask(enhanced)

    return enhanced, mask

