#!/usr/bin/env python3
"""
Create demonstration materials showing each processing stage.
Generates comparison images for presentations and documentation.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.preprocessing import apply_dcp_dehazing, apply_clahe


def create_processing_stages_demo(input_image_path: str, output_dir: str = "demo_outputs"):
    """Create a visual demonstration of all processing stages."""
    
    print(f"\n🎨 Creating demo materials from: {input_image_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load input image
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"❌ Failed to load image: {input_image_path}")
        return False
    
    print(f"✅ Loaded image: {img.shape}")
    
    stages = []
    titles = []
    
    # Stage 1: Original
    stages.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    titles.append("1. Original Zeiss Visuscout")
    
    # Stage 2: DCP Dehazing
    dehazed = apply_dcp_dehazing(img)
    stages.append(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB))
    titles.append("2. DCP Dehazing")
    cv2.imwrite(str(output_path / "01_dehazed.png"), dehazed)
    
    # Stage 3: CLAHE Enhancement
    clahe_img = apply_clahe(dehazed)
    stages.append(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
    titles.append("3. CLAHE Enhancement")
    cv2.imwrite(str(output_path / "02_clahe.png"), clahe_img)
    
    # Stage 4: Vessel Map (classical method)
    gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    _, vessel_map = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    stages.append(vessel_map)
    titles.append("4. Vessel Segmentation")
    cv2.imwrite(str(output_path / "03_vessel_map.png"), vessel_map)
    
    # Create composite figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Retinal Image Enhancement Pipeline', fontsize=20, fontweight='bold')
    
    for idx, (stage, title) in enumerate(zip(stages, titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if len(stage.shape) == 2:
            ax.imshow(stage, cmap='gray')
        else:
            ax.imshow(stage)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # Save composite
    composite_path = output_path / "processing_pipeline_demo.png"
    plt.tight_layout()
    plt.savefig(composite_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved composite demo: {composite_path}")
    print(f"✅ Saved individual stages: {output_path}")
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Before & After Comparison', fontsize=20, fontweight='bold')
    
    axes[0].imshow(stages[0])
    axes[0].set_title('Original Zeiss Visuscout', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(stages[2])  # CLAHE enhanced
    axes[1].set_title('Enhanced Result', fontsize=14)
    axes[1].axis('off')
    
    comparison_path = output_path / "before_after_comparison.png"
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison: {comparison_path}")
    
    # Create README for demo materials
    readme_content = f"""# Demo Materials

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files

1. **processing_pipeline_demo.png** - Complete 4-stage pipeline visualization
2. **before_after_comparison.png** - Side-by-side comparison
3. **01_dehazed.png** - DCP dehazing result
4. **02_clahe.png** - CLAHE enhancement result
5. **03_vessel_map.png** - Vessel segmentation result

## Processing Stages

1. **Original Zeiss Visuscout** - Input image from handheld camera
2. **DCP Dehazing** - Removes greenish haze using Dark Channel Prior
3. **CLAHE Enhancement** - Improves local contrast
4. **Vessel Segmentation** - Extracts retinal vessel structure

## Usage

These images can be used for:
- Project presentations
- Documentation
- Research papers
- Clinical demonstrations
- Training materials

---

*Retinal Image Enhancement System - FYP 2026*
"""
    
    readme_path = output_path / "README.md"
    readme_path.write_text(readme_content)
    
    print(f"✅ Saved README: {readme_path}")
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create demo materials for presentations')
    parser.add_argument('--input', type=str, help='Input Zeiss image path')
    parser.add_argument('--output', type=str, default='demo_outputs', help='Output directory')
    
    args = parser.parse_args()
    
    if args.input:
        success = create_processing_stages_demo(args.input, args.output)
        sys.exit(0 if success else 1)
    else:
        # Try to use a sample from the dataset
        sample_dirs = list(Path('patient_images').glob('*/'))
        if sample_dirs:
            for patient_dir in sample_dirs[:3]:  # Try first 3 patients
                zeiss_dir = patient_dir / 'zeiss-low quality'
                if not zeiss_dir.exists():
                    zeiss_dir = list(patient_dir.glob('*zeiss*'))
                    if zeiss_dir:
                        zeiss_dir = zeiss_dir[0]
                
                if zeiss_dir and zeiss_dir.exists():
                    images = list(zeiss_dir.glob('*.jpg')) + list(zeiss_dir.glob('*.png'))
                    if images:
                        print(f"Using sample image: {images[0]}")
                        success = create_processing_stages_demo(str(images[0]), args.output)
                        sys.exit(0 if success else 1)
        
        print("❌ No input image specified and no sample images found")
        print("Usage: python scripts/create_demo_materials.py --input <path_to_zeiss_image>")
        sys.exit(1)


if __name__ == "__main__":
    main()

