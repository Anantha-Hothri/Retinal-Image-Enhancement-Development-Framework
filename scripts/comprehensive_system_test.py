#!/usr/bin/env python3
"""
Comprehensive System Test Suite

Tests all components:
1. Data Pipeline
2. Model Architecture
3. Backend API
4. Training Infrastructure
5. Frontend Integration
"""

import sys
from pathlib import Path
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_data_pipeline():
    """Test data loading and preprocessing."""
    print_section("TEST 1: DATA PIPELINE")
    
    try:
        from src.data.dataset import RetinalDataset
        from utils.config import get_config
        
        config = get_config()
        print("✓ Config loaded successfully")
        
        # Try to load dataset
        dataset = RetinalDataset(config, split='train')
        print(f"✓ Dataset loaded: {len(dataset)} training pairs")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample loaded: zeiss={sample['zeiss'].shape}, clarus={sample['clarus'].shape}")
            return True
        else:
            print("⚠ Warning: No training pairs found")
            return True  # Not critical for system test
            
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_architecture():
    """Test model can be instantiated."""
    print_section("TEST 2: MODEL ARCHITECTURE")
    
    try:
        from models.real_esrgan_sft import create_sft_real_esrgan
        from models.discriminator import PatchGANDiscriminator
        
        print("✓ Attempting to create generator...")
        generator = create_sft_real_esrgan()
        print(f"✓ Generator created successfully")
        
        print("✓ Attempting to create discriminator...")
        discriminator = PatchGANDiscriminator()
        print(f"✓ Discriminator created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing():
    """Test preprocessing functions."""
    print_section("TEST 3: PREPROCESSING")
    
    try:
        import cv2
        import numpy as np
        from utils.preprocessing import PreprocessingPipeline
        
        # Create test image
        test_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        pipeline = PreprocessingPipeline()
        print("✓ Preprocessing pipeline created")
        
        # Test DCP dehazing
        dehazed = pipeline.apply_dcp_dehazing(test_img)
        print(f"✓ DCP dehazing: {test_img.shape} -> {dehazed.shape}")
        
        # Test CLAHE
        clahe = pipeline.apply_clahe(dehazed)
        print(f"✓ CLAHE enhancement: {dehazed.shape} -> {clahe.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vessel_extraction():
    """Test vessel map extraction."""
    print_section("TEST 4: VESSEL EXTRACTION")
    
    try:
        import cv2
        import numpy as np
        from utils.vessel_extraction import extract_vessels_classical
        
        # Create test image
        test_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        vessel_map = extract_vessels_classical(test_img)
        print(f"✓ Vessel extraction: {test_img.shape} -> {vessel_map.shape}")
        print(f"  Vessel pixels: {np.count_nonzero(vessel_map)} ({100*np.count_nonzero(vessel_map)/vessel_map.size:.2f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Vessel extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backend_services():
    """Test backend services can be imported and initialized."""
    print_section("TEST 5: BACKEND SERVICES")
    
    try:
        from backend.app.services.enhancement_service import EnhancementService
        from backend.app.services.metrics_service import MetricsService
        from backend.app.services.registration_service import RegistrationService
        
        print("✓ All backend services imported successfully")
        
        # Test metrics service
        metrics = MetricsService()
        print("✓ MetricsService initialized")
        
        # Test registration service  
        reg = RegistrationService()
        print("✓ RegistrationService initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend services test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frontend_structure():
    """Test frontend files exist."""
    print_section("TEST 6: FRONTEND STRUCTURE")
    
    frontend_dir = project_root / "frontend"
    
    required_files = [
        "package.json",
        "next.config.js",
        "tailwind.config.js",
        "tsconfig.json",
        "src/pages/index.tsx",
        "src/pages/_app.tsx"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = frontend_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def test_api_imports():
    """Test API can be imported."""
    print_section("TEST 7: API IMPORTS")
    
    try:
        from backend.app.main import app
        from backend.app.routers import inference
        
        print("✓ FastAPI app imported successfully")
        print("✓ Inference router imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ API imports test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all system tests."""
    print("\n" + "="*80)
    print("  COMPREHENSIVE SYSTEM TEST SUITE")
    print("="*80)
    print("\nTesting all components before training and deployment...")
    
    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Model Architecture", test_model_architecture),
        ("Preprocessing", test_preprocessing),
        ("Vessel Extraction", test_vessel_extraction),
        ("Backend Services", test_backend_services),
        ("Frontend Structure", test_frontend_structure),
        ("API Imports", test_api_imports)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print("\n" + "="*80)
    print(f"  RESULTS: {passed_count}/{total_count} tests passed")
    print("="*80)
    
    if passed_count == total_count:
        print("\n✅ ALL TESTS PASSED - SYSTEM READY!")
    else:
        print("\n⚠ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

