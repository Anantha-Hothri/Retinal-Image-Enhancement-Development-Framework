#!/usr/bin/env python3
"""
Test backend processing step by step to debug errors.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from backend.app.services.enhancement_service import EnhancementService

def test_processing():
    """Test the enhancement service step by step."""
    print("="*80)
    print("BACKEND PROCESSING TEST")
    print("="*80)
    
    # Load test image
    test_image_path = "patient_images/01/ZEISS - LOW QUALITY/LE DC.JPG"
    print(f"\n1. Loading test image: {test_image_path}")
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    img = cv2.imread(test_image_path)
    print(f"✓ Image loaded: {img.shape}")
    
    # Create enhancement service
    print(f"\n2. Creating enhancement service...")
    try:
        service = EnhancementService()
        print(f"✓ Service created")
    except Exception as e:
        print(f"❌ Failed to create service: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test preprocessing
    print(f"\n3. Testing DCP dehazing...")
    try:
        dehazed = service.preprocessor.apply_dcp_dehazing(img)
        print(f"✓ Dehazing successful: {dehazed.shape}")
    except Exception as e:
        print(f"❌ Dehazing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test CLAHE
    print(f"\n4. Testing CLAHE...")
    try:
        green_channel = dehazed[:, :, 1]
        clahe_enhanced = service.preprocessor.apply_clahe(green_channel)
        print(f"✓ CLAHE successful: {clahe_enhanced.shape}")
    except Exception as e:
        print(f"❌ CLAHE failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test vessel extraction
    print(f"\n5. Testing vessel extraction...")
    try:
        vessel_map = service._extract_vessel_map(clahe_enhanced)
        print(f"✓ Vessel extraction successful: {vessel_map.shape}")
    except Exception as e:
        print(f"❌ Vessel extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test model loading
    print(f"\n6. Testing model loading...")
    try:
        service.load_model()
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test tensor preparation
    print(f"\n7. Testing tensor preparation...")
    try:
        img_tensor = service._prepare_input(dehazed)
        vessel_tensor = service._prepare_vessel_map(vessel_map)
        print(f"✓ Input tensor: {img_tensor.shape}")
        print(f"✓ Vessel tensor: {vessel_tensor.shape}")
    except Exception as e:
        print(f"❌ Tensor preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test model inference
    print(f"\n8. Testing model inference...")
    try:
        import torch
        with torch.no_grad():
            output = service.model(img_tensor, vessel_tensor)
        print(f"✓ Model inference successful: {output.shape}")
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test tensor to image conversion
    print(f"\n9. Testing tensor to image conversion...")
    try:
        result = service._tensor_to_image(output)
        print(f"✓ Conversion successful: {result.shape}")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ ALL STEPS PASSED!")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_processing()
    sys.exit(0 if success else 1)

