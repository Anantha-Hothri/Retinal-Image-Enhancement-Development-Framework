# Retinal Image Enhancement API Documentation

## Overview

FastAPI backend for transforming low-quality Zeiss Visuscout retinal images into high-quality Clarus-equivalent images.

**Base URL:** `http://localhost:8000`  
**API Docs:** `http://localhost:8000/docs` (Auto-generated Swagger UI)

---

## Authentication

Currently, the API does not require authentication. For production deployment, implement JWT or OAuth2.

---

## Endpoints

### 1. Health Check

**GET** `/api/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "retinal-enhancement-api"
}
```

---

### 2. Upload Image

**POST** `/api/upload`

Upload a Zeiss Visuscout image for validation.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** `file` (image file)

**Allowed Formats:** JPEG, PNG  
**Max Size:** 50MB

**Response:**
```json
{
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "zeiss_image.jpg",
  "size": 2048576,
  "dimensions": "1920x1080",
  "message": "Upload successful. Use /api/process to enhance."
}
```

**Errors:**
- `400`: Invalid file type or size
- `500`: Server error

---

### 3. Process Image

**POST** `/api/process`

Process uploaded image through the complete enhancement pipeline.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** `file` (image file)

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "temp_dir": "/path/to/temp/dir",
  "success": true,
  "steps": [
    {
      "name": "input",
      "title": "Original Zeiss Visuscout Image",
      "description": "Low-quality narrow-FOV retinal image",
      "path": "input.jpg"
    },
    {
      "name": "dehazed",
      "title": "Dark Channel Prior Dehazing",
      "description": "Removes greenish haze",
      "path": "01_dehazed.png"
    },
    {
      "name": "clahe",
      "title": "CLAHE Enhancement",
      "description": "Improves local contrast",
      "path": "02_clahe.png"
    },
    {
      "name": "vessel_map",
      "title": "Vessel Segmentation",
      "description": "Extracted retinal vessel structure",
      "path": "03_vessel_map.png"
    },
    {
      "name": "enhanced",
      "title": "Deep Learning Enhancement",
      "description": "SFT-Real-ESRGAN super-resolution",
      "path": "04_enhanced.png"
    },
    {
      "name": "final",
      "title": "Final Enhanced Image",
      "description": "High-quality Clarus-equivalent image",
      "path": "05_final.png"
    }
  ],
  "metrics": {
    "resolution_improvement": "4.00x",
    "size_original": "1920x1080",
    "size_enhanced": "7680x4320"
  }
}
```

---

### 4. Get Step Image

**GET** `/api/result/{request_id}/{step}`

Retrieve image from a specific processing step.

**Parameters:**
- `request_id`: UUID from `/api/process` response
- `step`: One of `input`, `dehazed`, `clahe`, `vessel_map`, `enhanced`, `final`

**Response:** Image file (PNG)

**Example:**
```
GET /api/result/550e8400-e29b-41d4-a716-446655440000/final
```

---

### 5. Compare Images

**POST** `/api/compare`

Calculate quality metrics between two images.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** 
  - `file1`: First image (e.g., original)
  - `file2`: Second image (e.g., enhanced)

**Response:**
```json
{
  "success": true,
  "metrics": {
    "psnr": 28.45,
    "ssim": 0.8523,
    "vessel_recovery": 0.7234,
    "sharpness_original": 125.34,
    "sharpness_enhanced": 456.78,
    "contrast_original": 45.2,
    "contrast_enhanced": 68.9,
    "resolution_improvement": "4.00x",
    "size_original": "1920x1080",
    "size_enhanced": "7680x4320"
  }
}
```

**Metrics Explained:**
- **PSNR** (20-50 dB): Peak Signal-to-Noise Ratio, higher is better
- **SSIM** (0-1): Structural Similarity, higher is better
- **Vessel Recovery** (0-1): Vessel structure preservation, higher is better
- **Sharpness**: Laplacian variance, higher is sharper
- **Contrast**: Standard deviation of intensities, higher is more contrast

---

### 6. Register Images

**POST** `/api/register`

Register and align two images using feature matching.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `file1`: First image
  - `file2`: Second image

**Response:**
```json
{
  "success": true,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
  "num_matches": 156,
  "warped_path": "/path/to/warped.png",
  "blended_path": "/path/to/blended.png"
}
```

---

### 7. Get Registration Result

**GET** `/api/registration/{request_id}/{image_type}`

Retrieve registration result images.

**Parameters:**
- `request_id`: UUID from `/api/register` response
- `image_type`: Either `warped` or `blended`

**Response:** Image file (PNG)

---

### 8. Cleanup

**DELETE** `/api/cleanup/{request_id}`

Delete temporary files for a request.

**Response:**
```json
{
  "status": "success",
  "message": "Cleaned up request 550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Error Responses

All endpoints follow this error format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Status Codes:**
- `200`: Success
- `400`: Bad request (invalid input)
- `404`: Resource not found
- `500`: Internal server error

---

## Usage Examples

### Python (requests)

```python
import requests

# Upload and process
with open('zeiss_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/process',
        files={'file': f}
    )

result = response.json()
request_id = result['request_id']

# Download final image
final_image = requests.get(
    f'http://localhost:8000/api/result/{request_id}/final'
)

with open('enhanced.png', 'wb') as f:
    f.write(final_image.content)
```

### cURL

```bash
# Process image
curl -X POST http://localhost:8000/api/process \
  -F "file=@zeiss_image.jpg"

# Get result
curl http://localhost:8000/api/result/<request_id>/final \
  -o enhanced.png
```

---

## Rate Limiting

No rate limiting is currently implemented. For production, consider adding rate limiting middleware.

---

## CORS

CORS is enabled for:
- `http://localhost:3000` (React dev server)
- `http://localhost:5173` (Vite dev server)

Update `backend/app/main.py` to add additional origins for production.

