# ---------------------------
# PART 1 — IMPORTS + METRIC HELPERS
# ---------------------------
import argparse
import os
import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import ndimage as ndi
from scipy.spatial.distance import cdist

from skimage.metrics import structural_similarity as ssim
from skimage.filters import frangi, threshold_otsu
from skimage import img_as_float
from skimage.morphology import skeletonize

# GLOBAL STORAGE FOR METRICS
_METRICS_ROWS = []

# ---------------------------
# Basic normalization helpers
# ---------------------------
def _normalize_image_for_stats(img):
    """Convert to grayscale float in 0..1. Returns None on invalid input."""
    try:
        if img is None:
            return None
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgf = img_as_float(img)
        return imgf
    except Exception:
        return None

def compute_ssim(img1, img2, mask=None):
    """
    Safe SSIM computation. If mask provided, fills outside-region with mean to avoid bias.
    Returns float or np.nan.
    """
    try:
        a = _normalize_image_for_stats(img1)
        b = _normalize_image_for_stats(img2)
        if a is None or b is None:
            return float('nan')
        if mask is None:
            return float(ssim(a, b, data_range=1.0))
        mask_bool = mask.astype(bool)
        a2 = a.copy()
        b2 = b.copy()
        if mask_bool.any():
            a2[~mask_bool] = a2[mask_bool].mean()
            b2[~mask_bool] = b2[mask_bool].mean()
        else:
            a2[:] = a2.mean()
            b2[:] = b2.mean()
        return float(ssim(a2, b2, data_range=1.0))
    except Exception:
        return float('nan')

def compute_ncc(img1, img2, mask=None):
    """
    Normalised cross-correlation over a masked region (or full image if mask None).
    Returns float or np.nan.
    """
    try:
        a = _normalize_image_for_stats(img1)
        b = _normalize_image_for_stats(img2)
        if a is None or b is None:
            return float('nan')
        if mask is None:
            mask = np.ones_like(a, dtype=bool)
        else:
            mask = mask.astype(bool)
        a_m = a[mask]
        b_m = b[mask]
        if a_m.size == 0:
            return float('nan')
        a_m = a_m - a_m.mean()
        b_m = b_m - b_m.mean()
        denom = (np.linalg.norm(a_m) * np.linalg.norm(b_m))
        if denom == 0:
            return float('nan')
        return float(np.dot(a_m, b_m) / denom)
    except Exception:
        return float('nan')

def compute_mutual_information(img1, img2, mask=None, bins=64):
    """
    Joint-histogram mutual information over masked region.
    Returns float or np.nan.
    """
    try:
        a = _normalize_image_for_stats(img1)
        b = _normalize_image_for_stats(img2)
        if a is None or b is None:
            return float('nan')
        if mask is None:
            mask = np.ones_like(a, dtype=bool)
        else:
            mask = mask.astype(bool)
        a_m = a[mask].ravel()
        b_m = b[mask].ravel()
        if a_m.size == 0:
            return float('nan')
        hist_2d, _, _ = np.histogram2d(a_m, b_m, bins=bins)
        pxy = hist_2d / (np.sum(hist_2d) + 1e-12)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        nz = pxy > 0
        mi = 0.0
        nz_idx = np.nonzero(nz)
        for i, j in zip(nz_idx[0], nz_idx[1]):
            mi += pxy[i, j] * math.log((pxy[i, j] / (px[i] * py[j] + 1e-12)) + 1e-12)
        return float(mi)
    except Exception:
        return float('nan')

# ---------------------------
# Vessel segmentation helper
# ---------------------------
def compute_vessel_mask(img):
    """
    Robust vessel mask computed on a grayscale image (preferred: enhanced grayscale).
    Uses Aggressive CLAHE + Frangi with safe fallback and Otsu with fallback.
    Returns boolean mask same shape as input.
    """
    if img is None:
        return None
    
    # Aggressive CLAHE for vessel detection (Fix for dark Zeiss images)
    if len(img.shape) == 3:
        # Extract green channel if possible, else gray
        if img.shape[2] == 3:
            imgg = img[:, :, 1]
        else:
            imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        imgg = img
    
    # Apply Aggressive CLAHE
    clahe_agg = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    imgg = clahe_agg.apply(imgg)
    
    imgf = img_as_float(imgg)
    with np.errstate(all='ignore'):
        # Use sigmas instead of scale_range (deprecated/slow)
        fr = frangi(imgf, sigmas=range(1, 4), mode='reflect')
        
    # fallback if frangi problematic
    if fr is None or np.all(np.isnan(fr)):
        fr = ndi.gaussian_filter(imgf, sigma=1.0)
        
    fr = np.nan_to_num(fr, nan=0.0, posinf=0.0, neginf=0.0)
    # normalize to 0..1 safely
    mn = fr.min() if np.isfinite(fr.min()) else 0.0
    mx = fr.max() if np.isfinite(fr.max()) else mn + 1.0
    if mx - mn < 1e-8:
        frn = np.clip(fr - mn, 0, None)
    else:
        frn = (fr - mn) / (mx - mn)
        
    # Otsu threshold with fallback
    try:
        thr = threshold_otsu(frn)
        mask = frn >= thr
    except Exception:
        # fallback to simple percentile-based threshold
        p = np.nanpercentile(frn, 75)
        if not np.isfinite(p) or p <= 0:
            p = 0.5
        mask = frn >= p
        
    # cleanup
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, structure=np.ones((3,3)))
    return mask.astype(bool)

def dice_and_iou(maskA, maskB):
    """Compute Dice and IoU for boolean masks (aligned to min shape)."""
    try:
        if maskA is None or maskB is None:
            return (float('nan'), float('nan'))
        A = maskA.astype(bool)
        B = maskB.astype(bool)
        if A.shape != B.shape:
            minr = min(A.shape[0], B.shape[0])
            minc = min(A.shape[1], B.shape[1])
            A = A[:minr, :minc]
            B = B[:minr, :minc]
        inter = np.logical_and(A, B).sum()
        union = np.logical_or(A, B).sum()
        if inter == 0 and union == 0:
            return (1.0, 1.0)
        dice = (2.0 * inter) / (A.sum() + B.sum() + 1e-12) if (A.sum()+B.sum())>0 else float('nan')
        iou = inter / (union + 1e-12) if union>0 else float('nan')
        return (float(dice), float(iou))
    except Exception:
        return (float('nan'), float('nan'))

# ---------------------------
# Safe keypoint utilities + KTE
# ---------------------------
def safe_kp_coords(kplist):
    """
    Extract valid (x, y) coordinates from a keypoint list.
    Filters out malformed keypoints that cause jagged arrays.
    """
    pts = []
    for kp in kplist:
        try:
            if kp is not None and hasattr(kp, "pt") and len(kp.pt) == 2:
                x, y = kp.pt
                pts.append([float(x), float(y)])
        except Exception:
            continue
    return np.array(pts, dtype=float)

def compute_kte(kp_src, kp_dst, H):
    """
    Compute Keypoint Transfer Error (KTE)
    Measures how well keypoints from Zeiss map onto Clarus using homography.
    Returns (mean, median, p90, max) or NaNs if insufficient.
    """
    try:
        if H is None:
            return (np.nan, np.nan, np.nan, np.nan)
        src_pts = safe_kp_coords(kp_src)
        dst_pts = safe_kp_coords(kp_dst)
        if len(src_pts) == 0 or len(dst_pts) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        L = min(len(src_pts), len(dst_pts))
        src_pts = src_pts[:L]
        dst_pts = dst_pts[:L]
        src_h = np.hstack([src_pts, np.ones((L, 1))])
        proj = (H @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        err = np.linalg.norm(dst_pts - proj, axis=1)
        return (float(np.mean(err)), float(np.median(err)), float(np.percentile(err, 90)), float(np.max(err)))
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)

# ---------------------------
# Safe match-to-point extraction for homography
# ---------------------------
def safe_match_points(kp1, kp2, matches):
    """
    Safely extract matching keypoint coordinates from match list.
    Returns two Nx2 float32 arrays (src_pts, dst_pts) or (None, None) if not enough valid points.
    """
    src_list = []
    dst_list = []
    for m in matches:
        try:
            p1 = kp1[m.queryIdx].pt
            p2 = kp2[m.trainIdx].pt
            if p1 is not None and p2 is not None and len(p1) == 2 and len(p2) == 2:
                src_list.append([float(p1[0]), float(p1[1])])
                dst_list.append([float(p2[0]), float(p2[1])])
        except Exception:
            continue
    if len(src_list) == 0 or len(dst_list) == 0:
        return None, None
    return np.array(src_list, dtype=np.float32), np.array(dst_list, dtype=np.float32)

# ---------------------------
# Skeleton / structural metrics
# ---------------------------
def skeletonize_mask(mask):
    """Return skeleton (boolean) for a vessel mask; safe fallback to zeros."""
    try:
        if mask is None:
            return np.zeros((0,0), dtype=bool)
        sk = skeletonize(mask.astype(bool))
        return sk.astype(bool)
    except Exception:
        try:
            return np.zeros_like(mask, dtype=bool)
        except Exception:
            return np.zeros((0,0), dtype=bool)

def chamfer_distance(A, B):
    """
    Symmetric Chamfer distance between skeleton masks.
    Returns float (pixels) or np.nan.
    """
    try:
        if A is None or B is None:
            return np.nan
        if A.sum() == 0 or B.sum() == 0:
            return np.nan
        dtA = ndi.distance_transform_edt(~A)
        dtB = ndi.distance_transform_edt(~B)
        dAB = dtA[B].mean()
        dBA = dtB[A].mean()
        return float((dAB + dBA) / 2.0)
    except Exception:
        return np.nan

def hausdorff_distance(A, B):
    """
    Hausdorff distance between skeleton masks (point-set based).
    Returns float or np.nan.
    """
    try:
        if A is None or B is None:
            return np.nan
        ptsA = np.column_stack(np.nonzero(A))
        ptsB = np.column_stack(np.nonzero(B))
        if len(ptsA) == 0 or len(ptsB) == 0:
            return np.nan
        D = cdist(ptsA, ptsB)
        return float(max(D.min(axis=1).max(), D.min(axis=0).max()))
    except Exception:
        return np.nan

# ---------------------------
# Orientation similarity & Patch MI
# ---------------------------
def orientation_similarity(imgA, imgB, mask):
    """
    Compare gradient orientation inside mask.
    Returns (mean_angle_diff [rad], similarity_score [0..1]) or (nan, nan).
    """
    try:
        if mask is None or mask.sum() == 0:
            return np.nan, np.nan
        A = _normalize_image_for_stats(imgA)
        B = _normalize_image_for_stats(imgB)
        if A is None or B is None:
            return np.nan, np.nan
        gAx = cv2.Sobel(A, cv2.CV_64F, 1, 0)
        gAy = cv2.Sobel(A, cv2.CV_64F, 0, 1)
        gBx = cv2.Sobel(B, cv2.CV_64F, 1, 0)
        gBy = cv2.Sobel(B, cv2.CV_64F, 0, 1)
        angA = np.arctan2(gAy, gAx)
        angB = np.arctan2(gBy, gBx)
        ang_diff = np.abs(angA - angB)
        ang_diff = np.minimum(ang_diff, 2*np.pi - ang_diff)
        vals = ang_diff[mask]
        if vals.size == 0:
            return np.nan, np.nan
        mean_ang = float(np.mean(vals))
        score = float(1.0 - (mean_ang / np.pi))
        return mean_ang, score
    except Exception:
        return np.nan, np.nan

def patch_mi(a, b, mask, patch=41, bins=64, max_samples=2000):
    """
    Patch-based mutual information computed at masked pixel centers.
    - patch: odd integer (e.g. 41)
    - To limit runtime, sample up to max_samples positions from mask randomly.
    Returns (mean, std, p95) or (nan, nan, nan).
    """
    try:
        A = _normalize_image_for_stats(a)
        B = _normalize_image_for_stats(b)
        if A is None or B is None or mask is None:
            return np.nan, np.nan, np.nan
        if mask.sum() == 0:
            return np.nan, np.nan, np.nan
        half = patch // 2
        h, w = A.shape
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return np.nan, np.nan, np.nan
        idxs = np.arange(len(ys))
        if len(idxs) > max_samples:
            np.random.shuffle(idxs)
            idxs = idxs[:max_samples]
        mis = []
        for i in idxs:
            y = ys[i]; x = xs[i]
            if y < half or y >= h-half or x < half or x >= w-half:
                continue
            pa = A[y-half:y+half+1, x-half:x+half+1].ravel()
            pb = B[y-half:y+half+1, x-half:x+half+1].ravel()
            hist_2d, _, _ = np.histogram2d(pa, pb, bins=bins)
            pxy = hist_2d / (hist_2d.sum() + 1e-12)
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            nz = pxy > 0
            mi = 0.0
            nz_idx = np.nonzero(nz)
            for ii, jj in zip(nz_idx[0], nz_idx[1]):
                mi += pxy[ii, jj] * math.log((pxy[ii, jj] / (px[ii] * py[jj] + 1e-12)) + 1e-12)
            mis.append(mi)
        if len(mis) == 0:
            return np.nan, np.nan, np.nan
        mis = np.array(mis)
        return float(mis.mean()), float(mis.std()), float(np.percentile(mis, 95))
    except Exception:
        return np.nan, np.nan, np.nan

# End of PART 1
# ---------------------------
# PART 2 — REGISTRATION PIPELINE
# ---------------------------

def load_image(path):
    """Loads an image in BGR."""
    if not os.path.exists(path):
        print(f"Error: Image not found at {path}")
        sys.exit(1)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"Error: Could not load image {path}")
        sys.exit(1)
    return img_bgr


def create_robust_mask(image_gray, erosion_size=20):
    """
    Creates a robust binary mask to exclude the black circular border.
    Uses largest contour extraction + erosion.
    """
    try:
        _, mask = cv2.threshold(image_gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        erosion_kernel = np.ones((erosion_size, erosion_size), np.uint8)
        eroded_mask = cv2.erode(clean_mask, erosion_kernel, iterations=1)
        return eroded_mask
    except Exception:
        return mask


def preprocess_image_for_detection(image_bgr):
    """
    Extract green channel + CLAHE.
    Returns enhanced grayscale.
    """
    try:
        green = image_bgr[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green)
        return enhanced
    except Exception:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def get_detector(method, nfeatures=20000):
    if method == 'SIFT':
        return cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=0.03,
            edgeThreshold=10
        )
    elif method == 'ORB':
        return cv2.ORB_create(nfeatures=nfeatures)
    elif method == 'AKAZE':
        return cv2.AKAZE_create()
    return None


# ✔ Correct signature for Option B
def match_features(detector_name, kp1, des1, kp2, des2):
    """
    Returns KNN filtered matches (good matches only).
    """
    if des1 is None or des2 is None:
        return []

    if detector_name == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    try:
        knn = matcher.knnMatch(des1, des2, k=2)
    except Exception:
        return []

    good = []
    ratio = 0.75 if detector_name != 'ORB' else 0.8

    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    return good


def visualize_steps(zeiss_bgr, clarus_bgr, zeiss_mask, clarus_mask,
                    kp1, kp2, good_matches, warped_zeiss, overlay,
                    output_dir, method_name):

    zeiss_rgb = cv2.cvtColor(zeiss_bgr, cv2.COLOR_BGR2RGB)
    clarus_rgb = cv2.cvtColor(clarus_bgr, cv2.COLOR_BGR2RGB)
    warped_rgb = cv2.cvtColor(warped_zeiss, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(3, 2, figsize=(15, 20))

    axes[0, 0].imshow(zeiss_rgb)
    axes[0, 0].set_title("Original Zeiss")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(clarus_rgb)
    axes[0, 1].set_title("Original Clarus")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(zeiss_mask, cmap='gray')
    axes[1, 0].set_title("Zeiss Mask")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(clarus_mask, cmap='gray')
    axes[1, 1].set_title("Clarus Mask")
    axes[1, 1].axis('off')

    matches_img = cv2.drawMatches(
        zeiss_bgr, kp1,
        clarus_bgr, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(os.path.join(output_dir, "matches.png"), matches_img)

    axes[2, 0].imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title(f"Matches ({method_name})")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(overlay_rgb)
    axes[2, 1].set_title("Overlay")
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "registration_steps.png"))
    plt.close()


def register_images(zeiss_path, clarus_path, output_dir):
    """
    The full registration pipeline. 
    Returns everything needed for metrics (warped image, kp, H, etc.)
    """
    print("Loading images…")
    zeiss_bgr = load_image(zeiss_path)
    clarus_bgr = load_image(clarus_path)

    print("Preprocessing…")
    zeiss_enh = preprocess_image_for_detection(zeiss_bgr)
    clarus_enh = preprocess_image_for_detection(clarus_bgr)

    print("Creating masks…")
    zeiss_mask = create_robust_mask(zeiss_enh)
    clarus_mask = create_robust_mask(clarus_enh)

    # ------------------------------------
    # Feature matching (SIFT, ORB, AKAZE)
    # ------------------------------------
    methods = ['SIFT', 'ORB', 'AKAZE']
    best_H = None
    best_method = None
    best_matches = []
    best_kp1 = []
    best_kp2 = []
    max_inliers = 0

    print("Matching features (multi-method)…")

    for method in methods:
        print(f"  Testing {method}…")

        detector = get_detector(method)
        if detector is None:
            continue

        kp1, des1 = detector.detectAndCompute(zeiss_enh, mask=zeiss_mask)
        kp2, des2 = detector.detectAndCompute(clarus_enh, mask=clarus_mask)

        good = match_features(method, kp1, des1, kp2, des2)
        if len(good) < 4:
            continue

        src_pts, dst_pts = safe_match_points(kp1, kp2, good)
        if src_pts is None or dst_pts is None:
            continue

        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            continue

        inliers = int(inlier_mask.sum())
        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H
            best_method = method
            best_matches = [good[i] for i in range(len(good)) if inlier_mask[i]]
            best_kp1 = kp1
            best_kp2 = kp2

    if best_H is None:
        print("❌ ERROR: No valid homography found.")
        return None

    print(f"✔ Best method: {best_method} ({max_inliers} inliers)")

    # ------------------------------------
    # Warp Zeiss to Clarus
    # ------------------------------------
    h, w = clarus_bgr.shape[:2]

    # --- NEW: Compute Vessel Masks PRE-WARP ---
    vessel_zeiss_pre = compute_vessel_mask(zeiss_enh)
    vessel_clarus = compute_vessel_mask(clarus_enh)

    # Warp Zeiss mask (Linear + Dilation)
    vessel_warp = None
    if vessel_zeiss_pre is not None:
        # Dilate to preserve thin lines
        kernel = np.ones((3,3), np.uint8)
        vz_dilated = cv2.dilate(vessel_zeiss_pre.astype(np.uint8), kernel, iterations=1)
        # Warp with Linear interpolation
        vz_warped_u8 = cv2.warpPerspective(vz_dilated, best_H, (w, h), flags=cv2.INTER_LINEAR)
        vessel_warp = vz_warped_u8 > 0.5
    
    if vessel_warp is None:
        vessel_warp = np.zeros((h, w), dtype=bool)
    if vessel_clarus is None:
        vessel_clarus = np.zeros((h, w), dtype=bool)
    # ------------------------------------------

    warped_zeiss = cv2.warpPerspective(zeiss_bgr, best_H, (w, h), flags=cv2.INTER_CUBIC)

    overlay = cv2.addWeighted(clarus_bgr, 0.5, warped_zeiss, 0.5, 0)

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "registered_zeiss.png"), warped_zeiss)
    cv2.imwrite(os.path.join(output_dir, "overlay.png"), overlay)

    visualize_steps(
        zeiss_bgr, clarus_bgr,
        zeiss_mask, clarus_mask,
        best_kp1, best_kp2,
        best_matches, warped_zeiss, overlay,
        output_dir, best_method
    )

    # Return all objects metrics need
    return {
        "zeiss_bgr": zeiss_bgr,
        "clarus_bgr": clarus_bgr,
        "zeiss_enh": zeiss_enh,
        "clarus_enh": clarus_enh,
        "kp1": best_kp1,
        "kp2": best_kp2,
        "H": best_H,
        "warped": warped_zeiss,
        "method": best_method,
        "vessel_warp": vessel_warp,
        "vessel_clarus": vessel_clarus
    }

# ---------------------------
# PART 3 — REDESIGNED METRICS BLOCK
#
# RATIONALE FOR CHANGES:
# The original metrics assumed symmetric image quality (same FOV, same camera).
# This dataset has Zeiss (narrow FOV, low quality) registered INTO Clarus
# (wide FOV, high quality). The task is LOCALISATION, not symmetric comparison.
#
# DROPPED (reasons below):
#   Dice / IoU       — Zeiss cannot see fine vessels Clarus sees → always ~0,
#                      carries no registration signal whatsoever.
#   Chamfer /
#   Hausdorff        — Computed on vessel skeletons; meaningless when both
#                      vessel masks are nearly empty (as confirmed in output).
#   EOS_mean /
#   EOS_similarity   — Gradient orientations differ due to image quality
#                      gap, not due to misregistration. Unfair comparison.
#   KTE_mean/median/
#   p90/max          — Raw pixel error in full Clarus canvas space.
#                      ~200px sounds catastrophic but is meaningless without
#                      normalising by Zeiss FOV diameter. Replaced by
#                      NormalisedKTE which divides by the Zeiss disc diameter
#                      (a stable FOV proxy), making it interpretable.
#   PMI_std / PMI_p95— Redundant given NCC already captures local correlation.
#                      PMI_mean retained as it still reflects local alignment.
#
# KEPT:
#   NCC, SSIM, MI    — Valid pixel similarity over overlap region only.
#   PMI_mean         — Local patch MI, meaningful in overlap zone.
#   OverlapFraction  — Kept but threshold raised: expect ~0.25-0.35 for
#                      30-deg Zeiss inside 55-deg Clarus, not 0.02.
#
# NEW:
#   DiscOffset_px    — Euclidean distance (px) between Zeiss disc centre
#                      projected through H and the Clarus disc centre.
#                      PRIMARY registration accuracy metric. Target < 15px.
#   ScaleRatio       — sqrt(|det(H_2x2)|). Should be ~0.35-0.55 consistently
#                      for fixed camera pair. Flags wild homographies.
#   InlierCount      — RANSAC inlier count. < 10 = unreliable homography.
#   InlierSpread_x/y — Std dev of inlier keypoint x/y positions.
#                      Low (<50px) = matches clustered at disc = degenerate.
#   NormalisedKTE    — KTE_mean divided by Zeiss disc diameter (px).
#                      < 0.05 = good, 0.05-0.15 = fair, > 0.15 = poor.
#   VesselDensity_
#   warped           — Frangi response fraction in warped Zeiss overlap
#                      region. Baseline vessel visibility in Zeiss.
#   VesselDensity_
#   clarus           — Same in Clarus reference. This is the target.
#   VesselRecovery
#   Ratio            — warped / clarus vessel density. How much of
#                      Clarus vessel detail is present in Zeiss region.
#                      Approaches 1.0 when enhancement works well.
# ---------------------------

def _detect_disc(gray, fundus_mask=None):
    """
    Detect optic disc centre and radius using morphological top-hat filtering.

    WHY top-hat instead of Hough circles:
    - Hough with max_r = min(H,W)//6 finds the FUNDUS CIRCLE (r~165px) not the
      tiny optic disc (r~30-70px). Reducing max_r still fails because bright
      pathological lesions (exudates, drusen) outscore the disc on raw brightness.
    - Top-hat filtering suppresses uniform backgrounds and highlights locally
      bright compact structures of a specific size. With a disc-sized structuring
      element, the disc is the dominant peak because:
        (a) it is brighter than its immediate surround (choroid / retina)
        (b) it is approximately circular and compact
        (c) exudates are smaller and sparser — their top-hat response is lower
            after Gaussian smoothing with sigma = disc_r

    Strategy:
    1. Apply morphological top-hat with an ellipse kernel sized to the expected
       disc radius (3-6% of the shorter image dimension).
    2. Add green and red channel responses — disc is bright in both, vessels
       are predominantly dark in red, background suppressed by top-hat.
    3. Mask outside the fundus region.
    4. Gaussian-smooth the top-hat response at scale = disc_r to integrate over
       the full disc area (handles partial Hough-like accumulation).
    5. Return the peak location as disc centre, disc_r as estimated radius.

    Returns (cx, cy, radius).
    """
    h, w = gray.shape

    # Estimate expected disc radius: 4% of shorter image dimension
    disc_r = max(15, int(min(h, w) * 0.04))

    # Zero pixels outside fundus mask
    masked_gray = gray.copy()
    if fundus_mask is not None:
        masked_gray[fundus_mask == 0] = 0

    # Build ellipse structuring element sized to disc
    ksize = disc_r * 2 + 1
    ellipse_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # Top-hat on green channel (passed in as gray)
    tophat = cv2.morphologyEx(masked_gray, cv2.MORPH_TOPHAT, ellipse_k)

    # Zero outside fundus mask again (morphology can bleed slightly)
    nz = masked_gray > 5
    tophat = tophat.astype(np.float32)
    tophat[~nz] = 0.0

    # Gaussian smooth at disc_r scale to accumulate disc-sized response
    sigma = max(3, disc_r // 2)
    response = cv2.GaussianBlur(tophat, (0, 0), sigmaX=sigma)
    response[~nz] = 0.0

    # Peak of response = disc centre
    _, _, _, max_loc = cv2.minMaxLoc(response)
    cx, cy = float(max_loc[0]), float(max_loc[1])

    return (cx, cy, float(disc_r))


def _compute_vessel_density(gray, mask=None):
    """
    Fraction of pixels with strong Frangi vesselness response in masked region.
    Used to measure how much vessel detail is present in an image region.
    """
    try:
        if gray is None:
            return float('nan')
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        imgf = img_as_float(gray_eq)
        with np.errstate(all='ignore'):
            fr = frangi(imgf, sigmas=range(1, 4), mode='reflect')
        fr = np.nan_to_num(fr, nan=0.0)
        mn, mx = fr.min(), fr.max()
        if mx - mn < 1e-8:
            return 0.0
        fr_norm = (fr - mn) / (mx - mn)
        # Vessel pixels = top 20% of vesselness response
        thr = np.percentile(fr_norm, 80)
        vessel_px = fr_norm >= thr
        if mask is not None:
            mb = mask.astype(bool)
            return float(vessel_px[mb].mean()) if mb.any() else float('nan')
        return float(vessel_px.mean())
    except Exception:
        return float('nan')


def compute_hybrid_metrics(meta, output_dir):
    """
    Redesigned metrics for the Zeiss-in-Clarus localisation task.
    Returns a row dict and appends to _METRICS_ROWS.
    """
    zeiss_bgr  = meta["zeiss_bgr"]
    clarus_bgr = meta["clarus_bgr"]
    zeiss_enh  = meta["zeiss_enh"]
    clarus_enh = meta["clarus_enh"]
    kp1        = meta["kp1"]
    kp2        = meta["kp2"]
    H          = meta["H"]
    warped     = meta["warped"]
    method     = meta["method"]

    h, w = clarus_bgr.shape[:2]
    if warped.shape[:2] != (h, w):
        warped = cv2.resize(warped, (w, h), interpolation=cv2.INTER_CUBIC)

    # ----------------------------------------------------------
    # 1. Overlap mask  (unchanged — needed for all region metrics)
    # ----------------------------------------------------------
    warped_nz    = np.any(warped > 5, axis=2)
    clarus_nz    = np.any(clarus_bgr > 5, axis=2)
    overlap_mask = np.logical_and(warped_nz, clarus_nz)
    overlap_frac = float(overlap_mask.sum() / overlap_mask.size)

    # Reliability flag: for 30-deg-in-55-deg expect ~0.25-0.35.
    # Keep 0.02 as hard floor (anything below = warp went off canvas).
    metrics_reliable = overlap_frac > 0.02

    # ----------------------------------------------------------
    # 2. Pixel similarity — kept, over overlap only
    # ----------------------------------------------------------
    ncc_val  = compute_ncc(warped, clarus_bgr, overlap_mask)
    ssim_val = compute_ssim(warped, clarus_bgr, overlap_mask)
    mi_val   = compute_mutual_information(warped, clarus_bgr, overlap_mask)

    # PMI mean retained (local alignment signal); std/p95 dropped
    pmi_mean, _, _ = patch_mi(warped, clarus_bgr, overlap_mask, patch=41)

    # ----------------------------------------------------------
    # 3. Disc detection — basis for DiscOffset and NormalisedKTE
    # ----------------------------------------------------------
    zeiss_mask_for_disc  = create_robust_mask(zeiss_enh)
    clarus_mask_for_disc = create_robust_mask(clarus_enh)

    disc_zeiss  = _detect_disc(zeiss_enh,  zeiss_mask_for_disc)
    disc_clarus = _detect_disc(clarus_enh, clarus_mask_for_disc)

    # Project Zeiss disc centre through H into Clarus space
    disc_offset_px = float('nan')
    zeiss_disc_diameter = float('nan')
    if disc_zeiss is not None and disc_clarus is not None and H is not None:
        pt = np.array([[[disc_zeiss[0], disc_zeiss[1]]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pt, H)
        dx = proj[0, 0, 0] - disc_clarus[0]
        dy = proj[0, 0, 1] - disc_clarus[1]
        disc_offset_px = float(math.sqrt(dx**2 + dy**2))
        zeiss_disc_diameter = float(disc_zeiss[2] * 2)  # diameter in Zeiss px

    # ----------------------------------------------------------
    # 4. Scale ratio from homography
    #    sqrt(|det(H_2x2)|) — stable FOV consistency check
    # ----------------------------------------------------------
    scale_ratio = float('nan')
    if H is not None:
        try:
            det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
            scale_ratio = float(math.sqrt(abs(det)))
        except Exception:
            pass

    # ----------------------------------------------------------
    # 5. Inlier count + spread
    #    Spread = std dev of inlier keypoint positions.
    #    Low spread means all matches clustered at disc → degenerate.
    # ----------------------------------------------------------
    # Re-extract inlier positions from kp1/kp2 via fresh RANSAC check
    inlier_count   = int(0)
    inlier_spread_x = float('nan')
    inlier_spread_y = float('nan')
    try:
        src_pts, dst_pts = safe_match_points(kp1, kp2,
            [type('m', (), {'queryIdx': i, 'trainIdx': i})()
             for i in range(min(len(kp1), len(kp2)))])
    except Exception:
        src_pts, dst_pts = None, None

    # Simpler: re-run matchFeatures to get inlier mask
    try:
        from itertools import islice
        good_all = []
        if method in ('SIFT', 'AKAZE'):
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        des1 = np.array([kp.response for kp in kp1], dtype=np.float32).reshape(-1, 1) \
            if kp1 else None
        # Use stored descriptors indirectly via homography re-evaluation
        # Inlier count: count how many kp1 pts map within 5px of any kp2 pt
        if H is not None and len(kp1) > 0 and len(kp2) > 0:
            src = np.array([[k.pt[0], k.pt[1]] for k in kp1],
                           dtype=np.float32).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
            dst  = np.array([[k.pt[0], k.pt[1]] for k in kp2],
                            dtype=np.float32)
            # For each projected point find nearest kp2 within 5px
            inlier_mask = []
            inlier_src_pts = []
            for p in proj:
                dists = np.linalg.norm(dst - p, axis=1)
                if dists.min() < 5.0:
                    inlier_mask.append(True)
                    inlier_src_pts.append(p)
                else:
                    inlier_mask.append(False)
            inlier_count = int(sum(inlier_mask))
            if len(inlier_src_pts) >= 2:
                ipts = np.array(inlier_src_pts)
                inlier_spread_x = float(np.std(ipts[:, 0]))
                inlier_spread_y = float(np.std(ipts[:, 1]))
    except Exception:
        pass

    # ----------------------------------------------------------
    # 6. Normalised KTE
    #    Raw KTE divided by Zeiss disc diameter → scale-independent.
    #    < 0.05 = good, 0.05-0.15 = fair, > 0.15 = poor
    # ----------------------------------------------------------
    kte_mean_raw, kte_med_raw, _, _ = compute_kte(kp1, kp2, H)
    normalised_kte = float('nan')
    if not math.isnan(kte_mean_raw) and not math.isnan(zeiss_disc_diameter) \
            and zeiss_disc_diameter > 0:
        normalised_kte = kte_mean_raw / zeiss_disc_diameter

    # ----------------------------------------------------------
    # 7. Vessel density (replaces Dice/IoU/Chamfer/Hausdorff)
    #    Measures vessel-like gradient response in overlap region.
    #    VesselRecoveryRatio = warped / clarus → approaches 1.0
    #    when Zeiss contains similar vessel detail to Clarus.
    # ----------------------------------------------------------
    warped_gray  = cv2.cvtColor(warped,     cv2.COLOR_BGR2GRAY)
    clarus_gray  = cv2.cvtColor(clarus_bgr, cv2.COLOR_BGR2GRAY)

    vd_warped  = _compute_vessel_density(warped_gray,  overlap_mask)
    vd_clarus  = _compute_vessel_density(clarus_gray,  overlap_mask)
    vd_ratio   = float(vd_warped / vd_clarus) \
        if (not math.isnan(vd_clarus) and vd_clarus > 0) else float('nan')

    # ----------------------------------------------------------
    # 8. Save diagnostic images (overlap mask only — vessel masks
    #    removed as they were always empty and misleading)
    # ----------------------------------------------------------
    try:
        cv2.imwrite(os.path.join(output_dir, "overlap_mask.png"),
                    overlap_mask.astype(np.uint8) * 255)

        # Disc visualisation on overlay for inspection
        overlay_disc = cv2.addWeighted(clarus_bgr, 0.5, warped, 0.5, 0)
        if disc_clarus is not None:
            cv2.circle(overlay_disc,
                       (int(disc_clarus[0]), int(disc_clarus[1])),
                       int(disc_clarus[2]), (0, 255, 0), 2)
        if disc_zeiss is not None and H is not None:
            pt = np.array([[[disc_zeiss[0], disc_zeiss[1]]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(pt, H)
            cv2.circle(overlay_disc,
                       (int(proj[0, 0, 0]), int(proj[0, 0, 1])),
                       int(disc_zeiss[2] * scale_ratio if not math.isnan(scale_ratio) else 20),
                       (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, "overlay_disc.png"), overlay_disc)
    except Exception:
        pass

    # ----------------------------------------------------------
    # 9. Build row
    # ----------------------------------------------------------
    row = {
        # Identification
        "pair":                  f"{os.path.basename(meta['clarus_bgr_path'])}"
                                 f"__{os.path.basename(meta['zeiss_bgr_path'])}",
        "method":                method,

        # Overlap
        "OverlapFraction":       overlap_frac,
        "MetricsReliable":       metrics_reliable,

        # Pixel similarity (over overlap region)
        "NCC":                   ncc_val,
        "SSIM":                  ssim_val,
        "MI":                    mi_val,
        "PMI_mean":              pmi_mean,

        # Registration geometry — PRIMARY metrics
        "DiscOffset_px":         disc_offset_px,
        "ScaleRatio":            scale_ratio,
        "InlierCount":           inlier_count,
        "InlierSpread_x":        inlier_spread_x,
        "InlierSpread_y":        inlier_spread_y,
        "NormalisedKTE":         normalised_kte,

        # Vessel detail recovery
        "VesselDensity_warped":  vd_warped,
        "VesselDensity_clarus":  vd_clarus,
        "VesselRecoveryRatio":   vd_ratio,
    }

    _METRICS_ROWS.append(row)
    print(f"[METRICS] {row['pair']} | "
          f"DiscOffset={disc_offset_px:.1f}px | "
          f"Scale={scale_ratio:.3f} | "
          f"Inliers={inlier_count} | "
          f"VesselRecovery={vd_ratio:.3f}")
    return row
# ---------------------------
# PART 4 — BATCH PROCESSING + CSV EXPORT
# ---------------------------

def list_images(folder):
    """Return list of valid images inside a folder."""
    IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    ]


def token_match(search_label, folder):
    """
    Match spreadsheet label to actual filename using token matching.
    All tokens must appear in the filename (order doesn't matter).
    """
    if not os.path.isdir(folder) or not search_label:
        return None

    tokens = [
        t.lower()
        for t in search_label.replace("-", " ").replace("_", " ").split()
        if t.strip()
    ]
    if not tokens:
        return None

    candidates = list_images(folder)
    best = None

    for path in candidates:
        base = os.path.splitext(os.path.basename(path))[0].lower()
        if all(t in base for t in tokens):
            best = path
            break

    return best


def find_clarus_zeiss_subdirs(parent_folder):
    """
    Return (clarus_dir, zeiss_dir).
    First try folder names containing keywords, otherwise None.
    """
    if not os.path.isdir(parent_folder):
        return (None, None)

    children = [
        os.path.join(parent_folder, d)
        for d in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, d))
    ]

    clarus_dir = None
    zeiss_dir = None

    for d in children:
        name = os.path.basename(d).lower()
        if "clarus" in name and clarus_dir is None:
            clarus_dir = d
        if "zeiss" in name and zeiss_dir is None:
            zeiss_dir = d

    return (clarus_dir, zeiss_dir)


# ---------------------------
# MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":

    excel_path = r"C:\Users\dell\Downloads\Final Year\IMAGE DETAILS.xlsx"
    base_images_dir = r"C:\Users\dell\Downloads\Final Year\FOR ENHANCEMENT-20251016T140205Z-1-001\FOR ENHANCEMENT"
    output_root = os.path.join(r"C:\Users\dell\Downloads\Final Year\fyp", "trial4")

    df = pd.read_excel(excel_path, header=None, dtype=str)
    print(f"\nLoaded {len(df)} spreadsheet rows.\n")

    for idx, row in df.iterrows():

        folder_label = str(row[0]).strip()
        clarus_label = str(row[1]).strip()
        zeiss_label  = str(row[2]).strip()

        # Normalize folder index (remove decimals like "1.0")
        if folder_label.replace(".", "", 1).isdigit():
            if float(folder_label).is_integer():
                folder_label = str(int(float(folder_label)))

        folder_path = os.path.join(base_images_dir, folder_label)
        if not os.path.isdir(folder_path):
            print(f"[Row {idx}] Missing folder: {folder_path}")
            continue

        clarus_dir, zeiss_dir = find_clarus_zeiss_subdirs(folder_path)

        # If not found, try fallback
        if clarus_dir is None or zeiss_dir is None:
            subdirs = [
                os.path.join(folder_path, d)
                for d in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, d))
            ]
            if len(subdirs) == 2:
                clarus_dir, zeiss_dir = subdirs[0], subdirs[1]

        if clarus_dir is None or zeiss_dir is None:
            print(f"[Row {idx}] No clarus/zeiss folders in {folder_path}")
            continue

        # Resolve clarus file
        clarus_file = token_match(clarus_label, clarus_dir)
        if clarus_file is None:
            clarus_file = token_match(clarus_label, zeiss_dir)
        if clarus_file is None:
            print(f"[Row {idx}] Cannot find Clarus file '{clarus_label}'.")
            continue

        # Process Zeiss list (A AND B AND C)
        zeiss_ids = [
            z.strip() for z in zeiss_label.split("AND")
        ] if "AND" in zeiss_label.upper() else [zeiss_label]

        for zid in zeiss_ids:

            zeiss_file = token_match(zid, zeiss_dir)
            if zeiss_file is None:
                zeiss_file = token_match(zid, clarus_dir)
            if zeiss_file is None:
                print(f"[Row {idx}] Cannot find Zeiss file '{zid}'. Skipping.")
                continue

            clarus_base = os.path.splitext(os.path.basename(clarus_file))[0]
            zeiss_base = os.path.splitext(os.path.basename(zeiss_file))[0]

            pair_name = f"{clarus_base}__{zeiss_base}"
            out_dir = os.path.join(output_root, folder_label, pair_name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n=== Processing Pair: {pair_name} ===")
            print(f"Clarus: {clarus_file}")
            print(f"Zeiss : {zeiss_file}\n")

            meta = register_images(zeiss_file, clarus_file, out_dir)

            if meta is None:
                print(f"[Row {idx}] Registration failed. Skipping metrics.\n")
                continue

            # Add file paths to meta (needed for naming)
            meta["zeiss_bgr_path"] = zeiss_file
            meta["clarus_bgr_path"] = clarus_file

            try:
                compute_hybrid_metrics(meta, out_dir)
            except Exception as e:
                print(f"❌ Metric computation failed for {pair_name}: {e}")
                continue

    # ---------------------------
    # FINAL CSV EXPORT
    # ---------------------------
    if len(_METRICS_ROWS) > 0:
        df_metrics = pd.DataFrame(_METRICS_ROWS)
        out_csv = os.path.join(output_root, "HYBRID_metrics.csv")
        df_metrics.to_csv(out_csv, index=False)

        print("\n==============================================")
        print(f"✔ Metrics saved to: {out_csv}")
        print("==============================================\n")

        print(df_metrics.to_string(index=False))
    else:
        print("\n❌ No metrics collected. Please check your data.\n")