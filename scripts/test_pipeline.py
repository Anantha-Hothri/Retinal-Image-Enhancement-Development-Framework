#!/usr/bin/env python3
"""
Test script for the complete retinal image enhancement pipeline.
Tests data loading, preprocessing, model inference, and API endpoints.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.preprocessing import apply_dcp_dehazing, apply_clahe
from utils.config import get_config
from data.retinal_dataset import RetinalDataset


def test_data_pipeline():
    """Test data loading and preprocessing."""
    print("\n" + "="*80)
    print("TEST 1: Data Pipeline")
    print("="*80)
    
    try:
        config = get_config()
        train_csv = config.get('dataset.train_csv', 'outputs/data/train_pairs.csv')
        
        if not Path(train_csv).exists():
            print(f"❌ FAILED: Train CSV not found at {train_csv}")
            return False
        
        # Load dataset
        dataset = RetinalDataset(
            csv_path=train_csv,
            mode='train',
            config=config
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} pairs")
        
        # Test loading one sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample loaded:")
            print(f"   - Zeiss shape: {sample['zeiss'].shape}")
            print(f"   - Clarus shape: {sample['clarus'].shape}")
            print(f"   - Vessel shape: {sample['vessel_map'].shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_preprocessing():
    """Test DCP dehazing and CLAHE preprocessing."""
    print("\n" + "="*80)
    print("TEST 2: Preprocessing Functions")
    print("="*80)
    
    try:
        # Create test image
        test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Test DCP dehazing
        dehazed = apply_dcp_dehazing(test_img)
        print(f"✅ DCP dehazing: {test_img.shape} -> {dehazed.shape}")
        
        # Test CLAHE
        clahe = apply_clahe(dehazed)
        print(f"✅ CLAHE enhancement: {dehazed.shape} -> {clahe.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_model_loading():
    """Test model initialization and loading."""
    print("\n" + "="*80)
    print("TEST 3: Model Loading")
    print("="*80)
    
    try:
        from models.real_esrgan_sft import create_sft_real_esrgan
        
        # Create model
        model = create_sft_real_esrgan(
            num_feat=64,
            num_block=23,
            scale=4,
            pretrained_path='models/RealESRGAN_x4plus.pth'
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        test_input = torch.randn(1, 3, 128, 128)
        test_vessel = torch.randn(1, 1, 128, 128)
        
        model.eval()
        with torch.no_grad():
            output = model(test_input, test_vessel)
        
        print(f"✅ Forward pass: {list(test_input.shape)} -> {list(output.shape)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_api_health():
    """Test backend API health endpoint."""
    print("\n" + "="*80)
    print("TEST 4: API Health Check")
    print("="*80)
    
    try:
        import requests
        
        response = requests.get('http://localhost:8000/api/health', timeout=5)
        
        if response.status_code == 200:
            print(f"✅ API is healthy: {response.json()}")
            return True
        else:
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  WARNING: Backend not running (start with: cd backend && python app/main.py)")
        return None
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RETINAL IMAGE ENHANCEMENT - PIPELINE TEST SUITE")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Data Pipeline", test_data_pipeline()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("API Health", test_api_health()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result is True else ("❌ FAILED" if result is False else "⚠️  SKIPPED")
        print(f"{name:.<50} {status}")
    
    print("="*80)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("="*80)
    
    if failed > 0:
        print("\n❌ Some tests failed. Please fix issues before deployment.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! System is ready.")
        sys.exit(0)


if __name__ == "__main__":
    main()

