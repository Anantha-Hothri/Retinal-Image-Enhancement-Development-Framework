# Retinal Image Enhancement: Zeiss Visuscout to Clarus Transformation

A comprehensive deep learning system for transforming low-quality retinal images from Zeiss Visuscout handheld cameras into high-quality, wide field-of-view (FOV) images matching the Zeiss Clarus system standard.

## 🎯 Project Overview

**Problem**: Handheld Zeiss Visuscout cameras are widely available in hospitals but produce low-quality, narrow-FOV retinal images. High-quality Zeiss Clarus systems are expensive and less accessible.

**Solution**: AI-powered image enhancement that makes high-quality, wide-FOV retinal imaging feasible with readily available handheld cameras.

**Key Features**:
- Deep learning super-resolution with SFT-conditioned Real-ESRGAN
- Dark Channel Prior (DCP) dehazing for Zeiss images
- Automated image registration and FOV alignment
- Comprehensive quality metrics and comparison tools
- Interactive web-based visualization interface
- Clinical deployment-ready architecture

## 📁 Project Structure

```
augment/
├── src/
│   ├── models/              # Deep learning models
│   │   ├── unet_vessel.py   # U-Net for vessel segmentation
│   │   ├── sft_layer.py     # Spatial Feature Transform layers
│   │   ├── real_esrgan.py   # SFT-conditioned Real-ESRGAN (TODO)
│   │   └── discriminator.py # PatchGAN discriminator (TODO)
│   ├── data/                # Data loading and preprocessing
│   │   ├── retinal_dataset.py   # PyTorch dataset class
│   │   └── prepare_dataset.py   # Train/val split generation
│   ├── training/            # Training scripts (TODO)
│   │   ├── train.py
│   │   ├── losses.py
│   │   └── inference.py
│   └── utils/               # Utility functions
│       ├── config.py        # Configuration management
│       ├── preprocessing.py # DCP dehazing, CLAHE, masking
│       └── dataset_validator.py
├── backend/                 # FastAPI backend (TODO)
│   └── app/
│       ├── routers/
│       ├── services/
│       └── models/
├── frontend/                # Next.js/React frontend (TODO)
│   └── src/
│       ├── components/
│       ├── pages/
│       └── utils/
├── configs/
│   └── config.yaml          # Hyperparameters and settings
├── patient_images/          # Dataset (352 paired images)
├── outputs/                 # Generated outputs
├── checkpoints/             # Model checkpoints
├── fixed_trial4_updated.py  # Original registration script
├── Methodology.md           # Detailed methodology documentation
├── requirements.txt
└── setup.py
```

## 🚀 Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended: 12GB+ VRAM)
- Node.js 18+ (for frontend)

### Backend Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install opencv-contrib-python for guided filter
pip install opencv-contrib-python

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev  # Start development server at localhost:3000
```

## 📊 Dataset Structure

The dataset contains 352 paired retinal images organized as:

```
patient_images/
├── 01/
│   ├── CLARUS - HIGH QUALITY/
│   │   ├── LE DC.jpg  # Left Eye, Disc Centered
│   │   └── RE DC.jpg  # Right Eye, Disc Centered
│   └── ZEISS - LOW QUALITY/
│       ├── LE DC.JPG
│       ├── LE MC.JPG  # Macula Centered
│       ├── RE DC.JPG
│       └── RE MC.JPG
├── 02/
│   └── ...
└── patient_images_report.xlsx  # Pairing manifest
```

## 🔬 Methodology

The system follows a multi-stage pipeline as detailed in `Methodology.md`:

### Stage 1: Preprocessing
1. **Zeiss Images**: DCP dehazing in LAB color space → Green channel extraction → CLAHE
2. **Clarus Images**: Green channel extraction → CLAHE
3. **Fundus Masking**: Threshold → Largest contour → Morphological closing → Erosion

### Stage 2: Registration
1. Multi-method feature detection (SIFT, ORB, AKAZE)
2. Feature matching with ratio test
3. RANSAC homography estimation
4. Warp Zeiss image to Clarus coordinate space

### Stage 3: Super-Resolution Enhancement
1. Extract overlap region crops
2. U-Net vessel segmentation on Clarus images
3. SFT-conditioned Real-ESRGAN inference
4. Gaussian alpha blending for FOV extension

### Stage 4: Evaluation
- **Metrics**: PSNR, SSIM, FID, VesselRecoveryRatio, DiscOffset
- **Visualization**: Side-by-side, overlay, difference heatmaps
- **Reports**: Per-image and aggregate statistics

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

- **Training parameters**: batch size, learning rate, epochs
- **Model architecture**: SFT layers, RRDB blocks
- **Loss weights**: pixel, perceptual, adversarial, vessel
- **Data augmentation**: flip probabilities, color jitter
- **Preprocessing**: DCP parameters, CLAHE settings

## 🎓 Training

```bash
# Prepare dataset (create train/val split)
python src/data/prepare_dataset.py

# Generate vessel segmentation maps
python src/training/generate_vessel_maps.py

# Train SFT-Real-ESRGAN
python src/training/train.py --config configs/config.yaml

# Monitor with TensorBoard
tensorboard --logdir outputs/logs
```

## 🌐 Web Application

### Start Backend API
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Start Frontend
```bash
cd frontend
npm run dev
```

### Usage Workflow
1. **Upload** Zeiss Visuscout image
2. **View** preprocessing steps (dehazing, CLAHE)
3. **See** registration process (feature matching, homography)
4. **Watch** SR enhancement in real-time
5. **Upload** Clarus ground truth for comparison
6. **Compare** using side-by-side, overlay, heatmap modes
7. **Download** enhanced image and comparison report

## 📈 Expected Results

Based on methodology validation:

| Metric | Before Enhancement | Target After Enhancement |
|--------|-------------------|--------------------------|
| VesselRecoveryRatio | 0.3 - 0.5 | > 0.8 |
| SSIM | 0.85 - 0.92 | > 0.90 |
| NCC | 0.70 - 0.85 | > 0.85 |
| DiscOffset (px) | < 15 | Unchanged |

## 🚢 Deployment

### Vercel (Frontend)
```bash
cd frontend
vercel --prod
```

### Docker (Backend)
```bash
docker build -t retinal-enhancement-backend .
docker run -p 8000:8000 retinal-enhancement-backend
```

## 📝 TODO

- [ ] Complete Real-ESRGAN model implementation
- [ ] Implement PatchGAN discriminator
- [ ] Build multi-component loss function
- [ ] Create training loop with checkpointing
- [ ] Generate U-Net vessel maps for all Clarus images
- [ ] Train model for 200 epochs
- [ ] Implement FOV extension with Gaussian blending
- [ ] Build FastAPI backend endpoints
- [ ] Create Next.js frontend UI
- [ ] Implement comparison dashboard
- [ ] Add export functionality
- [ ] Write deployment scripts
- [ ] Create demo materials

## 📚 References

- Real-ESRGAN: [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- DRIVE Dataset: [https://drive.grand-challenge.org/](https://drive.grand-challenge.org/)
- Dark Channel Prior: He et al. (2011)
- Methodology: See `Methodology.md`

## 📧 Contact

For questions or collaboration, please open an issue on GitHub.

## 📄 License

MIT License - See LICENSE file for details.

