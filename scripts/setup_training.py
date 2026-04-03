"""Setup script to prepare everything needed for training."""

import os
import sys
from pathlib import Path
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config


def check_python_version():
    """Check Python version."""
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.9+)")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("\n2. Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ CUDA version: {torch.version.cuda}")
            print(f"   ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("   ⚠ CUDA not available (training will be very slow on CPU)")
            return False
    except ImportError:
        print("   ✗ PyTorch not installed")
        return False


def check_dependencies():
    """Check required packages."""
    print("\n3. Checking dependencies...")
    required = [
        'torch', 'torchvision', 'cv2', 'numpy', 'pandas', 
        'albumentations', 'tqdm', 'tensorboard', 'PIL'
    ]
    
    missing = []
    for pkg in required:
        try:
            if pkg == 'cv2':
                __import__('cv2')
            elif pkg == 'PIL':
                __import__('PIL')
            else:
                __import__(pkg)
            print(f"   ✓ {pkg}")
        except ImportError:
            print(f"   ✗ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    return True


def check_pretrained_weights():
    """Check for pretrained model weights."""
    print("\n4. Checking pretrained weights...")
    
    weights = {
        'Real-ESRGAN': 'models/RealESRGAN_x4plus.pth',
        'U-Net DRIVE': 'models/drive_unet.pth'
    }
    
    all_found = True
    for name, path in weights.items():
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / 1e6
            print(f"   ✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ {name}: {path} (not found)")
            all_found = False
    
    if not all_found:
        print("\n   Download instructions:")
        print("   Real-ESRGAN:")
        print("     wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O models/RealESRGAN_x4plus.pth")
        print("   U-Net DRIVE:")
        print("     Find pretrained weights on GitHub or train from scratch")
    
    return all_found


def check_dataset():
    """Check dataset preparation."""
    print("\n5. Checking dataset preparation...")
    
    # Check patient images
    patient_dir = Path("patient_images")
    if not patient_dir.exists():
        print("   ✗ patient_images directory not found")
        return False
    
    patient_folders = [f for f in patient_dir.iterdir() if f.is_dir() and not f.name.startswith('.')]
    print(f"   ✓ Found {len(patient_folders)} patient folders")
    
    # Check CSVs
    train_csv = Path("outputs/data/train_pairs.csv")
    val_csv = Path("outputs/data/val_pairs.csv")
    
    if train_csv.exists() and val_csv.exists():
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        print(f"   ✓ train_pairs.csv: {len(train_df)} pairs")
        print(f"   ✓ val_pairs.csv: {len(val_df)} pairs")
        return True
    else:
        print("   ✗ Dataset CSVs not found")
        print("   Run: python src/data/prepare_dataset.py")
        return False


def check_vessel_maps():
    """Check vessel segmentation maps."""
    print("\n6. Checking vessel maps...")
    
    vessel_dir = Path("outputs/vessel_maps")
    if not vessel_dir.exists():
        print("   ✗ Vessel maps directory not found")
        print("   Run: python src/training/generate_vessel_maps.py")
        return False
    
    train_vessels = list((vessel_dir / "train").glob("*_vessel.png")) if (vessel_dir / "train").exists() else []
    val_vessels = list((vessel_dir / "val").glob("*_vessel.png")) if (vessel_dir / "val").exists() else []
    
    if train_vessels and val_vessels:
        print(f"   ✓ Train vessel maps: {len(train_vessels)}")
        print(f"   ✓ Val vessel maps: {len(val_vessels)}")
        return True
    else:
        print("   ✗ Vessel maps not generated")
        print("   Run: python src/training/generate_vessel_maps.py")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n7. Creating output directories...")
    
    dirs = [
        'checkpoints',
        'outputs/logs',
        'outputs/data',
        'outputs/vessel_maps/train',
        'outputs/vessel_maps/val',
        'outputs/enhanced',
        'models'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {dir_path}")
    
    return True


def print_summary(results):
    """Print setup summary."""
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    all_ready = all(results.values())
    
    for check, status in results.items():
        status_str = "✓ READY" if status else "✗ NOT READY"
        print(f"{status_str:12} - {check}")
    
    print("="*60)
    
    if all_ready:
        print("\n🎉 All prerequisites met! Ready to start training.")
        print("\nNext steps:")
        print("  1. python src/training/trainer.py --epochs 200")
        print("  2. tensorboard --logdir outputs/logs")
    else:
        print("\n⚠ Some prerequisites are missing. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  • Dependencies: pip install -r requirements.txt")
        print("  • Dataset: python src/data/prepare_dataset.py")
        print("  • Vessel maps: python src/training/generate_vessel_maps.py")
        print("  • Weights: Download from URLs shown above")
    
    return all_ready


def main():
    """Main setup check."""
    print("="*60)
    print("TRAINING SETUP CHECKER")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'CUDA': check_cuda(),
        'Dependencies': check_dependencies(),
        'Pretrained Weights': check_pretrained_weights(),
        'Dataset': check_dataset(),
        'Vessel Maps': check_vessel_maps(),
        'Directories': create_directories()
    }
    
    all_ready = print_summary(results)
    
    return 0 if all_ready else 1


if __name__ == '__main__':
    sys.exit(main())

