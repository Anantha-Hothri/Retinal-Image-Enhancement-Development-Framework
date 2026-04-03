"""Inference router for image enhancement."""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import uuid
import shutil
import cv2
import numpy as np
import torch
from typing import Dict, List
import base64
import sys

# Add src and project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from backend.app.services.enhancement_service import EnhancementService
from backend.app.services.metrics_service import MetricsService
from backend.app.services.registration_service import RegistrationService

router = APIRouter()

# Initialize services (singleton pattern)
enhancement_service = None
metrics_service = MetricsService()
registration_service = RegistrationService()


def get_enhancement_service():
    """Get or create enhancement service."""
    global enhancement_service
    if enhancement_service is None:
        enhancement_service = EnhancementService()
    return enhancement_service


@router.post("/process")
async def process_image(
    file: UploadFile = File(...),
    reference_fov: UploadFile = File(None)
):
    """
    Process uploaded Zeiss image through the enhancement pipeline.

    Args:
        file: Zeiss Visuscout image to enhance
        reference_fov: Optional wide-FOV Clarus reference for FOV extension (Step 6)

    Returns all intermediate steps and final result.
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Create temporary directory for this request
        request_id = str(uuid.uuid4())
        temp_dir = Path("backend/temp") / request_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        input_path = temp_dir / "input.jpg"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Save reference FOV if provided
        reference_fov_path = None
        if reference_fov is not None:
            if not reference_fov.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Reference FOV must be an image")
            reference_fov_path = temp_dir / "reference_fov.jpg"
            with open(reference_fov_path, "wb") as buffer:
                shutil.copyfileobj(reference_fov.file, buffer)
            print(f"Reference FOV provided for FOV extension: {reference_fov_path}")

        # Get enhancement service
        service = get_enhancement_service()

        # Process image with optional reference FOV
        results = await service.process_image(
            str(input_path),
            str(temp_dir),
            reference_fov_path=str(reference_fov_path) if reference_fov_path else None
        )

        # Add request_id to results
        results['request_id'] = request_id
        results['temp_dir'] = str(temp_dir)

        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/result/{request_id}/{step}")
async def get_step_image(request_id: str, step: str):
    """Get image for a specific processing step."""
    try:
        temp_dir = Path("backend/temp") / request_id
        
        # Map step names to file paths
        step_files = {
            "input": "input.jpg",
            "dehazed": "01_dehazed.png",
            "clahe": "02_clahe.png",
            "vessel_map": "03_vessel_map.png",
            "enhanced": "04_enhanced.png",
            "final": "05_final.png"
        }
        
        if step not in step_files:
            raise HTTPException(status_code=400, detail=f"Invalid step: {step}")
        
        image_path = temp_dir / step_files[step]
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found for step: {step}")
        
        return FileResponse(image_path, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image: {str(e)}")


@router.delete("/cleanup/{request_id}")
async def cleanup_request(request_id: str):
    """Clean up temporary files for a request."""
    try:
        temp_dir = Path("backend/temp") / request_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return {"status": "success", "message": f"Cleaned up request {request_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")


@router.post("/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Compare two images and calculate quality metrics.

    Typically used to compare:
    - Original Zeiss vs Enhanced result
    - Enhanced result vs Ground truth Clarus
    """
    try:
        # Validate file types
        if not file1.content_type.startswith('image/') or not file2.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Both files must be images")

        # Create temporary directory
        request_id = str(uuid.uuid4())
        temp_dir = Path("backend/temp") / request_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded files
        img1_path = temp_dir / "compare_img1.jpg"
        img2_path = temp_dir / "compare_img2.jpg"

        with open(img1_path, "wb") as buffer:
            shutil.copyfileobj(file1.file, buffer)

        with open(img2_path, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)

        # Calculate metrics
        metrics = metrics_service.compare_images(str(img1_path), str(img2_path))

        # Clean up
        shutil.rmtree(temp_dir)

        return JSONResponse(content={"metrics": metrics, "success": True})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")


@router.post("/register")
async def register_images_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Register two images using feature detection and matching.

    Returns homography matrix and blended result.
    """
    try:
        # Validate file types
        if not file1.content_type.startswith('image/') or not file2.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Both files must be images")

        # Create temporary directory
        request_id = str(uuid.uuid4())
        temp_dir = Path("backend/temp") / request_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded files
        img1_path = temp_dir / "img1.jpg"
        img2_path = temp_dir / "img2.jpg"

        with open(img1_path, "wb") as buffer:
            shutil.copyfileobj(file1.file, buffer)

        with open(img2_path, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)

        # Perform registration
        result = registration_service.register_images(str(img1_path), str(img2_path), str(temp_dir))

        # Add request_id for retrieving results
        result['request_id'] = request_id

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")


@router.get("/registration/{request_id}/{image_type}")
async def get_registration_result(request_id: str, image_type: str):
    """
    Get registration result images.

    image_type: 'warped' or 'blended'
    """
    try:
        temp_dir = Path("backend/temp") / request_id

        if image_type not in ["warped", "blended"]:
            raise HTTPException(status_code=400, detail="image_type must be 'warped' or 'blended'")

        image_path = temp_dir / f"{image_type}.png"

        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_type}")

        return FileResponse(image_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image: {str(e)}")


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload a Zeiss Visuscout image for processing.

    Validates file format and saves for processing.
    Returns upload ID for tracking.
    """
    try:
        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
            )

        # Check file size (max 50MB)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning

        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {max_size / (1024*1024):.0f}MB"
            )

        # Create upload directory
        upload_id = str(uuid.uuid4())
        upload_dir = Path("backend/temp") / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = upload_dir / f"upload{Path(file.filename).suffix}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Verify it's a valid image
        img = cv2.imread(str(file_path))
        if img is None:
            shutil.rmtree(upload_dir)
            raise HTTPException(status_code=400, detail="Invalid image file")

        return JSONResponse(content={
            "upload_id": upload_id,
            "filename": file.filename,
            "size": file_size,
            "dimensions": f"{img.shape[1]}x{img.shape[0]}",
            "message": "Upload successful. Use /api/process to enhance."
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.post("/compare")
async def compare_images(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    Compare two images and return quality metrics.

    Args:
        file1: First image (typically enhanced result)
        file2: Second image (typically ground truth)

    Returns:
        Comparison metrics including PSNR, SSIM, vessel recovery, etc.
    """
    try:
        # Validate file types
        if not file1.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File1 must be an image")
        if not file2.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File2 must be an image")

        # Create temporary directory for comparison
        compare_id = str(uuid.uuid4())
        temp_dir = Path("backend/temp") / f"compare_{compare_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save both images
        img1_path = temp_dir / "image1.jpg"
        img2_path = temp_dir / "image2.jpg"

        with open(img1_path, "wb") as buffer:
            shutil.copyfileobj(file1.file, buffer)

        with open(img2_path, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)

        # Load images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        if img1 is None or img2 is None:
            raise HTTPException(status_code=400, detail="Failed to load images")

        print(f"Comparing images: {img1.shape} vs {img2.shape}")

        # Resize to same size if needed (resize smaller to match larger)
        if img1.shape != img2.shape:
            if img1.shape[0] * img1.shape[1] < img2.shape[0] * img2.shape[1]:
                # Resize img1 to match img2
                img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                # Resize img2 to match img1
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
            print(f"Resized images to: {img1.shape}")

        # Calculate comparison metrics
        try:
            psnr = MetricsService.calculate_psnr(img1, img2)
        except Exception as e:
            print(f"⚠️ PSNR calculation failed: {e}")
            psnr = 0.0

        try:
            ssim = MetricsService.calculate_ssim(img1, img2)
        except Exception as e:
            print(f"⚠️ SSIM calculation failed: {e}")
            ssim = 0.0

        try:
            vessel_recovery = MetricsService.calculate_vessel_recovery(img1, img2)
        except Exception as e:
            print(f"⚠️ Vessel Recovery calculation failed: {e}")
            vessel_recovery = 0.0

        metrics = {
            "psnr": float(psnr),
            "ssim": float(ssim),
            "vessel_recovery": float(vessel_recovery),
            "sharpness_img1": float(MetricsService.calculate_sharpness(img1)),
            "sharpness_img2": float(MetricsService.calculate_sharpness(img2)),
            "contrast_img1": float(MetricsService.calculate_contrast(img1)),
            "contrast_img2": float(MetricsService.calculate_contrast(img2)),
            "size_img1": f"{img1.shape[1]}x{img1.shape[0]}",
            "size_img2": f"{img2.shape[1]}x{img2.shape[0]}"
        }

        print(f"✓ Comparison metrics: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}, Vessel={vessel_recovery:.3f}")

        # Clean up temp files
        shutil.rmtree(temp_dir)

        return JSONResponse(content={
            "success": True,
            "metrics": metrics,
            "message": "Comparison completed successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

