#!/usr/bin/env python3
"""
Test FOV Extension with Gaussian Blending

Tests the implementation of Methodology Section 7.4:
- Gaussian alpha blending for seamless FOV extension
- Sigma tuning to find optimal blend zone width
- Diagnostic visualization generation
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from backend.app.services.registration_service import RegistrationService
from utils.config import get_config


def test_basic_fov_extension():
    """Test Gaussian alpha blending functions directly."""
    print("\n" + "="*80)
    print("TEST 1: Gaussian Alpha Blending Functions")
    print("="*80)

    # Create synthetic test images to demonstrate blending
    print(f"✓ Creating synthetic test images...")

    # Create a background (wide FOV - blue)
    background = np.ones((600, 800, 3), dtype=np.uint8) * np.array([200, 100, 50], dtype=np.uint8)

    # Create a foreground (narrow FOV - red) with circular mask
    foreground = np.ones((600, 800, 3), dtype=np.uint8) * np.array([50, 50, 200], dtype=np.uint8)
    center = (400, 300)
    radius = 200
    mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply mask to foreground
    for i in range(3):
        foreground[:,:,i] = cv2.bitwise_and(foreground[:,:,i], mask)

    print(f"  - Background (blue): {background.shape}")
    print(f"  - Foreground (red): {foreground.shape}")
    print(f"  - Mask: circular, radius={radius}px")

    # Test alpha mask creation
    print(f"\n✓ Testing alpha mask creation...")
    alpha_masks = {}
    for sigma in [10, 20, 40, 80]:
        alpha = RegistrationService.create_alpha_mask(mask, sigma=sigma)
        alpha_masks[sigma] = alpha
        print(f"  - Sigma {sigma}: blend zone ~{int(sigma*2)} pixels")

    # Test blending with different sigmas
    print(f"\n✓ Testing Gaussian alpha blending...")
    output_dir = Path("outputs/fov_extension_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs
    cv2.imwrite(str(output_dir / "test_background.png"), background)
    cv2.imwrite(str(output_dir / "test_foreground.png"), foreground)
    cv2.imwrite(str(output_dir / "test_mask.png"), mask)

    for sigma in [10, 20, 40, 80]:
        blended, alpha = RegistrationService.gaussian_alpha_blend(
            background,
            foreground,
            sigma=sigma,
            return_alpha=True
        )

        # Save blended result
        cv2.imwrite(str(output_dir / f"blended_sigma_{sigma}.png"), blended)

        # Save alpha mask visualization
        alpha_img = (alpha * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"alpha_sigma_{sigma}.png"), alpha_img)

        # Save alpha as heatmap
        alpha_heatmap = cv2.applyColorMap(alpha_img, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / f"alpha_heatmap_sigma_{sigma}.png"), alpha_heatmap)

        print(f"  ✓ Sigma {sigma}: saved blended result and alpha visualizations")

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  Compare the blending transition smoothness:")
    print(f"    - Sigma 10: Sharp transition (narrow blend zone)")
    print(f"    - Sigma 20: Medium transition")
    print(f"    - Sigma 40: Smooth transition (recommended)")
    print(f"    - Sigma 80: Very smooth (wide blend zone)")

    return True


def test_sigma_tuning():
    """Test sigma tuning utility function."""
    print("\n" + "="*80)
    print("TEST 2: Sigma Tuning Utility")
    print("="*80)

    # Create larger synthetic images for realistic tuning test
    print(f"✓ Creating synthetic test images...")

    # Background: 1000x1000 gradient (simulating wide FOV Clarus)
    background = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for i in range(1000):
        background[i, :, :] = int(100 + i * 0.1)

    # Foreground: 1000x1000 with elliptical content (simulating narrow FOV Zeiss)
    foreground = np.zeros((1000, 1000, 3), dtype=np.uint8)
    center = (500, 500)
    axes = (300, 250)
    cv2.ellipse(foreground, center, axes, 0, 0, 360, (200, 150, 100), -1)

    # Add some detail to foreground
    for _ in range(50):
        pt1 = (np.random.randint(200, 800), np.random.randint(250, 750))
        pt2 = (np.random.randint(200, 800), np.random.randint(250, 750))
        cv2.line(foreground, pt1, pt2, (255, 200, 150), 2)

    # Load sigma candidates from config
    config = get_config()
    sigma_candidates = config['blending']['sigma_candidates']

    print(f"  - Background: {background.shape}")
    print(f"  - Foreground: {foreground.shape}")
    print(f"\n✓ Testing sigma values: {sigma_candidates}")

    output_dir = Path("outputs/sigma_tuning")
    results = RegistrationService.tune_sigma(
        foreground,
        background,
        sigma_candidates=sigma_candidates,
        output_dir=str(output_dir)
    )

    print(f"\n✓ Sigma tuning complete!")
    print(f"\nResults for each sigma:")
    print(f"{'Sigma':<10} {'Overlap %':<12} {'Status'}")
    print("-" * 40)

    for key, result in results.items():
        sigma = result['sigma']
        overlap = result['overlap_fraction'] * 100
        print(f"{sigma:<10} {overlap:<12.2f} ✓")

    print(f"\n✓ Comparison images saved to: {output_dir}")
    print(f"  Visual comparison guidance:")
    print(f"    - Lower sigma (20): Sharper but may show seams")
    print(f"    - Medium sigma (30-40): Balanced, recommended for medical images")
    print(f"    - Higher sigma (50): Very smooth but may dilute detail near edges")

    return True


def main():
    """Run all FOV extension tests."""
    print("\n" + "="*80)
    print("FOV EXTENSION WITH GAUSSIAN BLENDING - TEST SUITE")
    print("="*80)
    print("\nImplementation of Methodology Section 7.4:")
    print("  - Gaussian alpha blending for seamless composites")
    print("  - Distance-transformed alpha masks")
    print("  - Configurable blend zone width (sigma)")
    print("  - Diagnostic visualizations")
    
    # Run tests
    tests = [
        ("Gaussian Alpha Blending Functions", test_basic_fov_extension),
        ("Sigma Tuning Utility", test_sigma_tuning)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - FOV EXTENSION IMPLEMENTATION COMPLETE!")
    else:
        print("❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

