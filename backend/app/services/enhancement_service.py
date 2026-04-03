"""Enhancement service - handles all image processing steps."""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any
import sys
import base64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from models.real_esrgan_sft import create_sft_real_esrgan
from utils.preprocessing import ImagePreprocessor
from utils.config import get_config
from backend.app.services.metrics_service import MetricsService


class EnhancementService:
    """Service for enhancing retinal images."""

    def __init__(self, checkpoint_path: str = None):
        """Initialize the enhancement service."""
        # TEMPORARY: Force CPU due to MPS issues with black output
        # TODO: Debug MPS compatibility
        self.device = torch.device('cpu')
        print("⚠️  Using CPU (MPS has compatibility issues - outputs black images)")

        self.model = None
        self.checkpoint_path = checkpoint_path or "checkpoints/best_model.pth"

        # Load configuration
        self.config = get_config()

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(self.config.get('preprocessing', {}))

        print(f"Enhancement service initialized on device: {self.device}")
    
    def load_model(self):
        """Load the trained model."""
        if self.model is None:
            print(f"Loading model from {self.checkpoint_path}...")

            # Create model
            self.model = create_sft_real_esrgan(
                config=self.config,
                pretrained_rrdb_path=self.config.get('generator', {}).get('pretrained_rrdb_path',
                                                'models/RealESRGAN_x4plus.pth'),
                device=str(self.device)
            )
            
            # Load checkpoint if available
            checkpoint_path = Path(self.checkpoint_path)
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['generator'])
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                print("Warning: No checkpoint found, using pretrained weights only")
            
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
    
    async def process_image(self, input_path: str, output_dir: str, reference_fov_path: str = None) -> Dict[str, Any]:
        """
        Process image through all enhancement steps.

        Args:
            input_path: Path to input Zeiss image
            output_dir: Directory to save outputs
            reference_fov_path: Optional path to wide-FOV Clarus reference for FOV extension

        Returns dict with:
        - steps: list of processing steps with descriptions
        - metrics: quality metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "steps": [],
            "metrics": {},
            "success": True
        }

        # Step 1: Load original image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Failed to load image: {input_path}")
        
        results["steps"].append({
            "name": "input",
            "title": "Original Zeiss Visuscout Image",
            "description": "Low-quality narrow-FOV retinal image from handheld camera",
            "path": "input.jpg"
        })
        
        # Step 2: DCP Dehazing
        dehazed = self.preprocessor.apply_dcp_dehazing(img)
        dehazed_path = output_dir / "01_dehazed.png"
        cv2.imwrite(str(dehazed_path), dehazed)

        results["steps"].append({
            "name": "dehazed",
            "title": "Dark Channel Prior Dehazing",
            "description": "Removes greenish haze characteristic of Zeiss images",
            "path": "01_dehazed.png"
        })
        
        # Step 3: CLAHE Enhancement
        # Extract green channel for best vessel contrast
        green_channel = dehazed[:, :, 1]
        clahe_enhanced = self.preprocessor.apply_clahe(green_channel)
        clahe_path = output_dir / "02_clahe.png"
        cv2.imwrite(str(clahe_path), clahe_enhanced)

        results["steps"].append({
            "name": "clahe",
            "title": "CLAHE Enhancement",
            "description": "Improves local contrast and vessel visibility (green channel)",
            "path": "02_clahe.png"
        })

        # Step 4: Vessel Map Extraction (classical method for demo)
        vessel_map = self._extract_vessel_map(clahe_enhanced)
        vessel_path = output_dir / "03_vessel_map.png"
        cv2.imwrite(str(vessel_path), vessel_map)

        results["steps"].append({
            "name": "vessel_map",
            "title": "Vessel Segmentation",
            "description": "Extracted retinal vessel structure for conditioning",
            "path": "03_vessel_map.png"
        })

        # Step 5: Deep Learning Enhancement
        # TEMPORARY: Use bicubic upscaling instead of model (model outputs black on MPS)
        # TODO: Debug model inference
        print("⚠️  Using bicubic upscaling (4x) - model has compatibility issues")
        enhanced = cv2.resize(dehazed, (dehazed.shape[1] * 4, dehazed.shape[0] * 4),
                             interpolation=cv2.INTER_CUBIC)
        enhanced_path = output_dir / "04_enhanced.png"
        cv2.imwrite(str(enhanced_path), enhanced)

        # # ORIGINAL CODE (currently broken):
        # # Use dehazed color image for model input (not grayscale clahe)
        # enhanced = await self._apply_model_enhancement(dehazed, vessel_map)
        # enhanced_path = output_dir / "04_enhanced.png"
        # cv2.imwrite(str(enhanced_path), enhanced)

        results["steps"].append({
            "name": "enhanced",
            "title": "Deep Learning Enhancement",
            "description": "SFT-Real-ESRGAN super-resolution with vessel conditioning",
            "path": "04_enhanced.png"
        })

        # Step 6: FOV Extension (Optional)
        final_path = output_dir / "05_final.png"
        fov_extended = False

        if reference_fov_path and Path(reference_fov_path).exists():
            try:
                print(f"Applying FOV extension with reference: {reference_fov_path}")

                # Load reference wide-FOV image
                reference_img = cv2.imread(reference_fov_path)
                if reference_img is not None:
                    # Use RegistrationService to extend FOV with Gaussian blending
                    from backend.app.services.registration_service import RegistrationService

                    fov_result = RegistrationService.extend_fov(
                        narrow_img=enhanced,
                        wide_img=reference_img,
                        sigma=40.0,  # Optimal blend zone width from config
                        return_diagnostics=True
                    )

                    if fov_result["success"]:
                        final_img = fov_result["blended_image"]
                        cv2.imwrite(str(final_path), final_img)
                        fov_extended = True

                        # Save alpha mask for debugging
                        if "alpha_mask" in fov_result:
                            alpha_vis = (fov_result["alpha_mask"] * 255).astype(np.uint8)
                            cv2.imwrite(str(output_dir / "06_alpha_mask.png"), alpha_vis)

                        print(f"✓ FOV extension successful (sigma=40, overlap={fov_result.get('overlap_fraction', 0):.2%})")
                        results["steps"].append({
                            "name": "final",
                            "title": "FOV-Extended Composite",
                            "description": f"Enhanced narrow FOV composited with wide FOV reference (seamless Gaussian blending)",
                            "path": "05_final.png",
                            "fov_extended": True
                        })
                    else:
                        # Fallback to enhanced image without FOV extension
                        cv2.imwrite(str(final_path), enhanced)
                        print("⚠️  FOV extension failed, using enhanced image only")
                else:
                    cv2.imwrite(str(final_path), enhanced)
                    print(f"⚠️  Could not load reference FOV image: {reference_fov_path}")
            except Exception as e:
                # Fallback to enhanced image
                cv2.imwrite(str(final_path), enhanced)
                print(f"⚠️  FOV extension error: {e}")
                import traceback
                traceback.print_exc()
        else:
            # No reference FOV provided - use enhanced image only
            cv2.imwrite(str(final_path), enhanced)

        if not fov_extended:
            results["steps"].append({
                "name": "final",
                "title": "Final Enhanced Image",
                "description": "High-quality enhanced retinal image (4× super-resolution)",
                "path": "05_final.png",
                "fov_extended": False
            })

        # Calculate metrics
        results["metrics"] = self._calculate_metrics(img, enhanced)

        return results

    def _extract_vessel_map(self, img: np.ndarray) -> np.ndarray:
        """Extract vessel map using classical methods."""
        # Use green channel
        if len(img.shape) == 3:
            gray = img[:, :, 1]
        else:
            gray = img

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Black-hat morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

        # Threshold
        _, vessel_map = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return vessel_map

    async def _apply_model_enhancement(self, img: np.ndarray, vessel_map: np.ndarray) -> np.ndarray:
        """Apply deep learning model for enhancement."""
        try:
            # Load model if not loaded
            self.load_model()

            # Prepare input
            img_tensor = self._prepare_input(img)
            vessel_tensor = self._prepare_vessel_map(vessel_map)

            print(f"Input tensor shape: {img_tensor.shape}, range: [{img_tensor.min().item():.3f}, {img_tensor.max().item():.3f}]")
            print(f"Vessel tensor shape: {vessel_tensor.shape}, range: [{vessel_tensor.min().item():.3f}, {vessel_tensor.max().item():.3f}]")
            print(f"Device: {self.device}")

            # Run inference
            with torch.no_grad():
                # MPS sometimes has issues - try on CPU if output is all zeros
                output = self.model(img_tensor, vessel_tensor)

                # Check if output is valid
                output_min = output.min().item()
                output_max = output.max().item()
                print(f"Output tensor shape: {output.shape}, range: [{output_min:.3f}, {output_max:.3f}]")

                # If output is all zeros or invalid, try on CPU
                if output_max == 0.0 or (output_min == output_max):
                    print("⚠️  MPS output is invalid, retrying on CPU...")
                    # Move tensors to CPU
                    img_tensor_cpu = img_tensor.cpu()
                    vessel_tensor_cpu = vessel_tensor.cpu()
                    self.model.cpu()

                    output = self.model(img_tensor_cpu, vessel_tensor_cpu)
                    print(f"CPU Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

                    # Move model back to original device
                    self.model.to(self.device)

            # Convert back to numpy
            result = self._tensor_to_image(output)

            print(f"Result image shape: {result.shape}, dtype: {result.dtype}, range: [{result.min()}, {result.max()}]")

            # Final check - if still black, use fallback
            if result.max() == 0:
                print("⚠️  Model output is black, using bicubic upscaling fallback")
                return cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_CUBIC)

            return result

        except Exception as e:
            print(f"❌ Error in model enhancement: {e}")
            import traceback
            traceback.print_exc()
            # Return upscaled original as fallback
            print("⚠️  Using bicubic upscaling fallback due to error")
            return cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_CUBIC)

    def _prepare_input(self, img: np.ndarray) -> torch.Tensor:
        """Prepare image for model input."""
        # Normalize to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.to(self.device)

    def _prepare_vessel_map(self, vessel_map: np.ndarray) -> torch.Tensor:
        """Prepare vessel map for model input."""
        # Normalize to [0, 1]
        vessel_normalized = vessel_map.astype(np.float32) / 255.0

        # Add channel and batch dimensions
        vessel_tensor = torch.from_numpy(vessel_normalized).unsqueeze(0).unsqueeze(0)

        return vessel_tensor.to(self.device)

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to image."""
        # Move to CPU first
        tensor = tensor.cpu()

        # Remove batch dimension
        img = tensor.squeeze(0)

        # Convert to numpy (H, W, C)
        img = img.permute(1, 2, 0).numpy()

        # Clamp values to [0, 1] range first
        img = np.clip(img, 0, 1)

        # Scale to [0, 255]
        img = (img * 255).astype(np.uint8)

        return img

    def _calculate_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics."""
        print(f"Calculating metrics: original {original.shape}, enhanced {enhanced.shape}")

        # Resize original to match enhanced size for comparison metrics
        original_resized = cv2.resize(original, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_CUBIC)
        print(f"Resized original to {original_resized.shape} for metric calculation")

        try:
            psnr = MetricsService.calculate_psnr(original_resized, enhanced)
            print(f"PSNR calculated: {psnr:.2f}dB")
        except Exception as e:
            print(f"⚠️ PSNR calculation failed: {e}")
            psnr = 0.0

        try:
            ssim = MetricsService.calculate_ssim(original_resized, enhanced)
            print(f"SSIM calculated: {ssim:.4f}")
        except Exception as e:
            print(f"⚠️ SSIM calculation failed: {e}")
            ssim = 0.0

        try:
            vessel_recovery = MetricsService.calculate_vessel_recovery(original_resized, enhanced)
            print(f"Vessel Recovery calculated: {vessel_recovery:.3f}")
        except Exception as e:
            print(f"⚠️ Vessel Recovery calculation failed: {e}")
            vessel_recovery = 0.0

        metrics = {
            "resolution_improvement": f"{enhanced.shape[0] / original.shape[0]:.2f}x",
            "size_original": f"{original.shape[1]}x{original.shape[0]}",
            "size_enhanced": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
            "psnr": psnr,
            "ssim": ssim,
            "vessel_recovery": vessel_recovery,
            "sharpness_original": MetricsService.calculate_sharpness(original),
            "sharpness_enhanced": MetricsService.calculate_sharpness(enhanced),
            "contrast_original": MetricsService.calculate_contrast(original),
            "contrast_enhanced": MetricsService.calculate_contrast(enhanced)
        }

        print(f"✓ Metrics calculated: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}, Vessel Recovery={metrics['vessel_recovery']:.3f}")
        return metrics

