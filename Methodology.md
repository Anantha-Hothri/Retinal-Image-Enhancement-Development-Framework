# Methodology: Multi-FOV Retinal Image Registration and Enhancement Pipeline

---

## 1. Dataset Organisation and Preparation

### 1.1 Dataset Structure

The dataset is organised in a hierarchical folder structure rooted at a single parent directory referred to here as `patient_images/`. Inside this directory, every patient has a dedicated folder named by a unique patient identifier (e.g., `patient_images/01/`, `patient_images/02/`, and so on). Within each patient folder, two subfolders exist.

The first subfolder is named `zeiss-low quality/` (or contains the keyword "zeiss" or "visuscout" in its name). This folder holds the narrow-FOV Zeiss VisuScout fundus photographs for that patient. There are typically two images inside: one for the left eye and one for the right eye. File naming conventions vary, but the filenames generally contain laterality indicators such as "LE = left eye, RE = right eye, MC = Macula centred, DC = Disc centred".

The second subfolder is named `clarus-high quality/` (or contains the keyword "clarus" in its name). This folder holds the wide-FOV Zeiss Clarus fundus photographs for the same patient. Again, two images are expected: one for the left eye and one for the right eye, following similar naming conventions.

The pairing logic works as follows: for every patient folder, each Zeiss image is paired with the Clarus image of the **same eye** from the same patient. A left-eye Zeiss image is paired only with the left-eye Clarus image, and the same applies for the right eye. Across the entire dataset, this yields 352 valid Zeiss–Clarus pairs in total.

A spreadsheet file named `patient_images_report.xlsx` serves as the pairing manifest. It contains columns for the folder label (patient identifier), the Clarus filename, and the Zeiss filename. Some rows use an "AND" syntax in the Zeiss column to indicate that multiple Zeiss images correspond to a single Clarus image (for instance, if a patient had two captures of the same eye). The batch processing loop reads this spreadsheet to drive the pairing.

### 1.2 Token-Based File Matching

Because the filenames in the spreadsheet may not exactly match the filenames on disk (due to spaces, dashes, underscores, or slight naming differences), the pipeline uses a token-matching strategy. The spreadsheet label is split on spaces, dashes, and underscores into individual tokens. A file on disk is considered a match if **all** tokens from the spreadsheet label appear somewhere in the filename. This handles minor inconsistencies without manual renaming.

### 1.3 Automatic Subfolder Detection

The code scans each patient folder for subfolders. It identifies the Clarus subfolder by checking whether the folder name contains the substring "clarus" (case-insensitive) and the Zeiss subfolder by checking for "zeiss" or "visuscout". If neither keyword is found, a fallback heuristic is applied (e.g., alphabetical ordering or the first two subfolders), but this is fragile and the recommended practice is to ensure the folder names contain these keywords.

### 1.4 Output Directory Structure

For every valid pair, the pipeline creates a dedicated output directory named after the pair identifier (derived from the patient ID and eye laterality). All outputs for that pair — registered images, overlays, diagnostic panels, metric values — are written into this directory. The final aggregated metrics across all 352 pairs are exported to a single CSV file named `HYBRID_metrics.csv` at the root of the output directory.

---

## 2. Preprocessing

### 2.1 Clarus Image Preprocessing

The Clarus image is loaded in BGR colour space using OpenCV. The green channel is extracted because it provides the highest contrast between retinal vessels and the surrounding fundus tissue; the red channel is too bright and saturated, and the blue channel is too noisy in retinal photographs. This single-channel extraction is the first operation.

CLAHE (Contrast Limited Adaptive Histogram Equalisation) is then applied to the extracted green channel. The parameters used are `clipLimit=2.0` and `tileGridSize=(8, 8)`. CLAHE divides the image into 8×8 tiles and equalises the histogram within each tile independently, with the clip limit preventing over-amplification of noise in homogeneous regions. This enhances local vessel contrast while preserving global intensity relationships, making fine secondary, tertiary, and quaternary vessels more detectable by keypoint extractors.

The function responsible is `preprocess_image_for_detection()`. It accepts a BGR image and returns a single-channel CLAHE-enhanced green-channel image.

### 2.2 Zeiss Image Preprocessing

The Zeiss VisuScout images suffer from a characteristic greenish haze caused by the lower-quality optics and narrower field of view. Currently, the same `preprocess_image_for_detection()` function is applied: green channel extraction followed by CLAHE with identical parameters.

**What remains to be implemented:** Before the CLAHE step, a Dark Channel Prior (DCP) dehazing pass must be added specifically for Zeiss images. The implementation should proceed as follows:

1. Convert the Zeiss BGR image to LAB colour space using `cv2.cvtColor(img, cv2.COLOR_BGR2LAB)`. Working in LAB separates luminance from colour, allowing dehazing to operate on the lightness channel without introducing colour artefacts.

2. Compute the dark channel of the L channel. For every pixel, take the minimum value in a local patch (recommended patch size: 15×15 pixels). This produces a single-channel dark channel map where hazy regions have high values (because haze raises the minimum intensity).

3. Estimate the atmospheric light `A` as the mean intensity of the top 0.1% brightest pixels in the dark channel.

4. Compute the transmission map `t(x) = 1 - omega * (dark_channel(x) / A)` where `omega = 0.95` (a standard value that preserves a small amount of haze for natural appearance).

5. Refine the transmission map using guided filtering (OpenCV's `cv2.ximgproc.guidedFilter`) with the original L channel as the guide image, radius 60, and epsilon 1e-3. This preserves edges while smoothing the transmission estimate.

6. Recover the dehazed L channel: `J(x) = (L(x) - A) / max(t(x), t0) + A` where `t0 = 0.1` is a lower bound to prevent division by near-zero values.

7. Replace the L channel in the LAB image with the dehazed version, convert back to BGR, and then proceed with the existing green channel extraction and CLAHE.

This dehazing step should be gated by a flag or by checking which camera produced the image, so that it is applied only to Zeiss inputs and not to Clarus inputs.

---

## 3. Fundus Masking 

### 3.1 Purpose

Both Clarus and Zeiss images contain a circular or near-circular fundus region surrounded by a black (or near-black) border. Keypoints detected on the border or in the transition zone between the fundus and the border are unreliable and will corrupt the homography estimation. The mask confines all subsequent operations — keypoint detection, vessel segmentation, metric computation — to the valid fundus interior.

### 3.2 Implementation — `create_robust_mask()`

The function proceeds in four sequential steps:

1. **Thresholding:** Convert the input image to grayscale (if not already) and apply a binary threshold at intensity value 10. Every pixel with intensity above 10 is set to 255 (white); everything else is set to 0 (black). This separates the lit fundus from the black surround.

2. **Largest contour extraction:** Find all contours in the binary image using `cv2.findContours` with `RETR_EXTERNAL` mode. Select the contour with the largest area. Fill this contour on a blank mask to produce a solid white region corresponding to the fundus. This step discards small specular reflections or sensor noise outside the fundus that might have survived the threshold.

3. **Morphological closing:** Apply `cv2.morphologyEx` with `MORPH_CLOSE` using a large elliptical kernel (recommended 25×25 pixels). This fills small internal gaps or holes in the fundus mask that might arise from very dark retinal regions (e.g., haemorrhages near the border).

4. **Erosion:** Erode the mask by 20 pixels using `cv2.erode` with a circular kernel of diameter 41 (i.e., 2×20+1). This pulls the mask boundary inward by 20 pixels, ensuring that the transition zone between the fundus edge and the black border — where pixel values are unreliable — is excluded. Keypoints and metrics will only operate inside this eroded boundary.

### 3.3 Known Limitation — Clarus Notched Corners

The Clarus imaging optics produce an image with notched (cut-off) corners rather than a perfectly circular boundary. These notches can survive the 20-pixel erosion and intrude into the valid mask region. This was observed to cause problems specifically in disc detection (see Section 5.4), where the bright edge of a notch was occasionally misidentified as the optic disc. The disc detection algorithm was redesigned to handle this (see Section 5.4), but if future processing steps are sensitive to corner artefacts, a larger erosion radius or a convex hull operation on the mask may be necessary.

---

## 4. Feature Detection and Matching 

### 4.1 Multi-Method Detection Strategy

Rather than committing to a single keypoint detector, the pipeline runs three detectors independently and selects the best result. The three detectors are:

**SIFT** (Scale-Invariant Feature Transform): Configured with `nfeatures=20000` and `contrastThreshold=0.03`. The high feature count and low contrast threshold are intentional — Zeiss images have low contrast, so a permissive threshold is needed to detect anything at all. SIFT produces 128-dimensional floating-point descriptors.

**ORB** (Oriented FAST and Rotated BRIEF): Used with default parameters. ORB produces 32-dimensional binary descriptors and is significantly faster than SIFT but less discriminative.

**AKAZE**: Used with default parameters. AKAZE operates in nonlinear scale space and produces binary descriptors. It often performs well on images with non-uniform illumination.

For each detector, keypoints are detected **only inside the fundus mask** by passing the mask to the `detect` or `detectAndCompute` call. This ensures no border keypoints contaminate the result.

### 4.2 Feature Matching — `match_features()`

Matching is performed using `cv2.BFMatcher` (brute-force matcher). For SIFT and AKAZE (floating-point descriptors), `NORM_L2` is used. For ORB (binary descriptors), `NORM_HAMMING` is used.

KNN matching with `k=2` is applied, producing two nearest neighbours for each descriptor. Lowe's ratio test is then applied: a match is kept only if the distance to the nearest neighbour is less than `ratio × distance to the second-nearest neighbour`. The ratio threshold is set to **0.75** for SIFT and AKAZE, and **0.80** for ORB (ORB descriptors are less discriminative, so a slightly more permissive threshold is used to retain enough matches).

### 4.3 Homography Estimation and Method Selection

For each of the three detector methods, the surviving matches after ratio filtering are passed to `cv2.findHomography` with the RANSAC method. The RANSAC reprojection threshold is set to the default (5.0 pixels). The function returns a 3×3 homography matrix `H` and a mask indicating which matches are inliers.

The **number of RANSAC inliers** is counted for each method. The method with the highest inlier count is selected as the best method for that pair. The corresponding homography matrix, inlier matches, and method name are stored in the output meta dictionary.

If no method produces at least 4 inliers (the minimum required to estimate a homography), the pair is flagged as a registration failure and skipped.

### 4.4 Known Limitation — Sparse, Spatially Clustered Inliers

In practice, the vast majority of matches are found at or near the optic disc because it is the only landmark in the Zeiss image with sufficient contrast and texture for keypoint detection. The macula is often too dark and featureless, and vessels in the Zeiss image lack the fine detail needed for distinctive descriptors. This means the homography is estimated from a small cluster of spatially concentrated points. The transform is therefore most accurate near the disc and may drift significantly toward the image periphery. The metrics `InlierSpread_x` and `InlierSpread_y` (standard deviation of inlier positions in Clarus space) quantify this: values below 50 pixels indicate a degenerate, disc-only homography. If this is observed consistently across the dataset, a fallback to a simpler affine transform (6 parameters instead of 8) anchored at the disc should be considered.

---

## 5. Registration and Warping 

### 5.1 Orchestration — `register_images()`

This is the main function that calls all of the above steps in sequence and produces the registered output. The execution order is:

1. Load the Clarus and Zeiss BGR images from disk.
2. Call `preprocess_image_for_detection()` on both images to obtain CLAHE-enhanced green channel versions.
3. Call `create_robust_mask()` on both images to obtain binary fundus masks.
4. For each of the three detector methods (SIFT, ORB, AKAZE): call `get_detector()` to instantiate the detector, call `detectAndCompute` on both preprocessed images with their masks, call `match_features()` to obtain filtered matches, and call `cv2.findHomography` with RANSAC.
5. Select the method with the most RANSAC inliers.
6. Store the selected homography matrix `H`, the inlier count, the method name, and the match coordinates.

### 5.2 Warping the Zeiss Image

Using the selected homography `H`, the original full-colour Zeiss BGR image is warped into the Clarus coordinate space using:

```
cv2.warpPerspective(zeiss_bgr, H, (clarus_w, clarus_h), flags=cv2.INTER_CUBIC)
```

`INTER_CUBIC` interpolation is used for smooth subpixel resampling. The output image has the same dimensions as the Clarus image, with the Zeiss content placed at the location dictated by the homography and all other pixels set to black.

### 5.3 Vessel Mask Warping

Before warping, the Frangi vessel mask of the Zeiss image is computed in its original coordinate space. This mask is then **dilated** by a small kernel (5×5, 2 iterations) before being warped. Dilation is necessary because thin single-pixel vessel lines can be broken or lost during perspective warping due to interpolation. The dilated mask is warped using `cv2.warpPerspective` with `INTER_NEAREST` to preserve binary values.

### 5.4 Optic Disc Detection — `_detect_disc()` (after three iterations)

This function locates the optic disc centre in a given fundus image. It is used to compute the `DiscOffset_px` metric (the primary registration accuracy metric). The function went through three design iterations.

**Iteration 1 (failed):** Used Hough circle detection with `max_r = min(H, W) // 6`, which equals approximately 165 pixels for a typical Clarus image. This radius corresponds to the entire fundus circle, not the optic disc (which is roughly 4% of the image dimension in radius). Hough consistently found the fundus boundary instead of the disc.

**Iteration 2 (failed):** Corrected the Hough radius range to the expected disc size, but on CLAHE-enhanced images, bright pathological lesions (particularly yellow exudates common in diabetic retinopathy patients) have higher peak pixel intensity than the disc. Hough detected the brightest circular region, which was often an exudate cluster rather than the disc.

**Iteration 3 (final, working):** Uses morphological top-hat filtering, which explicitly detects structures that are **brighter than their local neighbourhood at a specific spatial scale**. Implementation:

1. Compute the expected disc radius: `disc_r = max(15, int(min(h, w) * 0.04))`. For a 4000-pixel-wide Clarus image this gives approximately 160 pixels; for a smaller Zeiss image, proportionally smaller.
2. Compute the kernel size: `ksize = 2 * disc_r + 1` (ensuring an odd-sized kernel).
3. Create an elliptical structuring element: `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))`.
4. Apply morphological top-hat: `tophat = cv2.morphologyEx(masked_gray, cv2.MORPH_TOPHAT, ellipse_k)`. This subtracts the morphological opening from the original, leaving only bright features smaller than the kernel.
5. Apply Gaussian smoothing: `response = cv2.GaussianBlur(tophat, (0, 0), sigmaX=sigma)` where `sigma = disc_r * 0.75`. This aggregates the top-hat response over the disc-sized area, creating a smooth peak at the disc centre.
6. Find the pixel location of the maximum value in the smoothed response map. This is the detected disc centre.

**Why top-hat succeeds where Hough failed:** The disc is a compact, bright, roughly elliptical structure at a specific spatial scale. Exudates are smaller and sparser; their top-hat response is diluted after Gaussian smoothing at the disc scale. The fundus boundary is not a bright blob (it is a dark-to-bright transition) and is completely suppressed by the top-hat. Notched corners from the Clarus optics are suppressed because they are not bright relative to their neighbourhood — they are boundary artefacts adjacent to black regions.

Validation result: On the test pair, predicted disc centre was (229, 290) versus the manually annotated ground truth of (226, 287), giving a 4-pixel error.

### 5.5 Output Files per Pair

For each registered pair, four output files are saved into the pair's output directory:

1. **`registered_zeiss.png`** — The warped Zeiss image in Clarus coordinate space. Black pixels indicate regions outside the Zeiss footprint.
2. **`overlay.png`** — A 50/50 alpha blend of the Clarus image and the warped Zeiss image, useful for quick visual assessment of alignment.
3. **`registration_steps.png`** — A 3×2 diagnostic panel showing: (row 1) original Clarus and Zeiss images, (row 2) their binary fundus masks, (row 3) the match visualisation and the overlay.
4. **`matches.png`** — Lines drawn between matched keypoints for the winning method, with inliers and outliers distinguished by colour.
5. **`overlay_disc.png`** — Disc detection diagnostic overlay: the Clarus disc centre is marked with a green circle, and the Zeiss disc centre (projected through the homography) is marked with a red circle. The distance between them is `DiscOffset_px`.

---

## 6. Metrics Computation 

### 6.1 Design Philosophy

The original metric suite was designed for comparing two images of similar FOV and similar quality. This is fundamentally inappropriate for the Zeiss–Clarus task because the two cameras produce images with vastly different vessel visibility, resolution, and field of view. Twelve metrics that were misleading or uninformative in this asymmetric scenario were dropped, and nine new metrics designed specifically for narrow-to-wide FOV registration were introduced.

### 6.2 Metrics That Were Dropped and Justification

**Dice coefficient and IoU (Intersection over Union):** These compare binary vessel masks between the two images. Since the Zeiss image only shows primary and secondary vessels while the Clarus shows secondary through quaternary, the Clarus vessel mask will always contain vastly more vessel pixels. The overlap between the two masks is near zero regardless of registration quality. These metrics carry no registration signal.

**Chamfer distance and Hausdorff distance:** These operate on vessel skeletons (single-pixel-wide centrelines). When both vessel masks are nearly empty or very different in density, these distance metrics produce meaningless values dominated by noise.

**EOS_mean and EOS_similarity (Edge Orientation Similarity):** These compare image gradient orientations. The quality gap between cameras means gradients differ in magnitude and direction due to the imaging physics, not due to misregistration. High EOS does not imply good registration and low EOS does not imply bad registration.

**KTE_mean, KTE_median, KTE_p90, KTE_max (Keypoint Transfer Error — raw):** These report the pixel error when projecting keypoints through the homography, in absolute Clarus-canvas pixels. Without normalisation, these values are uninterpretable: a 200-pixel error in a 3000-pixel canvas may or may not be acceptable depending on the scale ratio and the expected Zeiss footprint size.

**PMI_std and PMI_p95 (Patch Mutual Information variability):** Redundant with NCC, which already captures local correlation. PMI_mean alone was retained as a single local alignment indicator.

### 6.3 Metrics Retained from Original Suite

**NCC (Normalised Cross-Correlation):** Computed over the overlap region only (where both the warped Zeiss and the Clarus have valid pixels). The overlap mask is the intersection of the warped Zeiss non-zero mask and the Clarus fundus mask. NCC is robust to linear intensity differences between the two cameras. Values range from -1 to 1; values above 0.7 indicate good alignment.

Implementation: Convert both images to grayscale float [0,1] using `_normalize_image_for_stats()`. Extract pixels within the overlap mask. Subtract the mean of each set. Compute `sum(a*b) / sqrt(sum(a²) * sum(b²))`.

**SSIM (Structural Similarity Index):** Computed over the overlap region using `skimage.metrics.structural_similarity` (or equivalent). Observed values around 0.89, which indicates strong structural agreement in the overlap zone. Values above 0.85 are considered good for cross-camera comparison.

**MI (Mutual Information):** Computed from the joint histogram of the two images over the overlap region. Both images are quantised to 256 bins. The joint histogram is normalised to a joint probability distribution, and MI is computed as `H(X) + H(Y) - H(X,Y)` where H denotes entropy. MI captures nonlinear intensity relationships and is camera-invariant.

**PMI_mean (Patch Mutual Information — mean):** The overlap region is divided into non-overlapping local patches (e.g., 64×64 pixels). MI is computed within each patch independently. The mean across all patches is reported. This captures local alignment quality and can reveal regions where the homography is drifting.

**OverlapFraction:** The fraction of the Clarus canvas that is covered by valid (non-zero) warped Zeiss pixels. Expected range is 0.25–0.35 for a 30-degree FOV inside a 55-degree FOV. Values below 0.02 indicate the warp has placed the Zeiss footprint mostly off-canvas (registration failure). The currently observed value of approximately 0.07 is lower than expected and is flagged as a known issue (see Section 8).

### 6.4 New Metrics Introduced

**DiscOffset_px:** The Euclidean distance (in pixels) between the detected Clarus disc centre and the Zeiss disc centre projected through the homography H. This is the **primary registration accuracy metric**. To compute it: detect the disc centre in the original Zeiss image using `_detect_disc()`, project that point through H using `cv2.perspectiveTransform`, detect the disc centre in the Clarus image independently, and compute the Euclidean distance. Target: below 15 pixels. Below 30 pixels is acceptable. Above 50 pixels indicates a likely registration failure.

**ScaleRatio:** Computed from the homography as `sqrt(|det(H[0:2, 0:2])|)`. This extracts the area scaling factor implied by the 2×2 linear submatrix of H. For a 30-degree FOV mapped onto a 55-degree canvas, the expected scale ratio is approximately 0.35–0.55. Values outside this range indicate a degenerate homography (e.g., extreme shear or flip). This metric is a quick sanity check before trusting other metric values.

**InlierCount:** The number of RANSAC inliers from the winning homography estimation. Below 10 inliers, the homography is unreliable. Below 6, it is likely degenerate (a homography requires a minimum of 4 point correspondences, so 6 provides minimal redundancy). Typical values with disc-only keypoints are 5–12.

**InlierSpread_x and InlierSpread_y:** The standard deviation of the x-coordinates and y-coordinates, respectively, of the inlier keypoints in Clarus canvas space. If all inliers are clustered at the optic disc, spread will be low (below 50 pixels). If inliers span the image (disc and vessels), spread will be higher. Low spread means the homography is anchored at only one point and may be inaccurate elsewhere.

**NormalisedKTE:** The mean keypoint transfer error divided by the detected Zeiss optic disc diameter (in pixels). This makes KTE scale-independent. Interpretation: below 0.05 is good, 0.05–0.15 is fair, above 0.15 is poor.

**VesselDensity_warped:** The fraction of pixels in the overlap region that are classified as vessel by the Frangi filter applied to the warped Zeiss image. This is a baseline measurement of how much vessel detail the Zeiss image contributes before enhancement.

**VesselDensity_clarus:** The same fraction computed on the Clarus image over the same overlap region. This represents the target vessel density — the amount of detail available in the ground truth.

**VesselRecoveryRatio:** Defined as `VesselDensity_warped / VesselDensity_clarus`. Before enhancement, this ratio is expected to be 0.3–0.5 (the Zeiss captures roughly one-third to one-half of the vessels visible in the Clarus). After super-resolution enhancement, the target is above 0.8. This is the **key enhancement evaluation metric** and will be tracked across all 352 pairs before and after the SR stage.

### 6.5 Vessel Mask Computation — `compute_vessel_mask()`

Currently implemented using Frangi filtering: the image is converted to grayscale float, CLAHE is applied, and `skimage.filters.frangi` is run with scale range [1, 6] and scale step 1. The Frangi response is thresholded using Otsu's method (`skimage.filters.threshold_otsu`) to produce a binary vessel mask.

**What will change (TODO, see Section 7.2):** For the Clarus images, the Frangi filter will be replaced by a DRIVE-pretrained U-Net that produces a richer, more accurate vessel segmentation. The Frangi filter will remain as the method for Zeiss images (where the U-Net is not expected to perform well due to the lower image quality).

### 6.6 CSV Output

All metrics for all pairs are aggregated into a single CSV file named `HYBRID_metrics.csv`. The file contains 17 columns:

`pair, method, OverlapFraction, MetricsReliable, NCC, SSIM, MI, PMI_mean, DiscOffset_px, ScaleRatio, InlierCount, InlierSpread_x, InlierSpread_y, NormalisedKTE, VesselDensity_warped, VesselDensity_clarus, VesselRecoveryRatio`

The `MetricsReliable` column is a boolean flag set to `False` when OverlapFraction is below 0.02 or InlierCount is below 4, indicating that the metric values for that pair should not be trusted.

---

## 7. Stages To Be Implemented

### 7.1 DCP Dehazing for Zeiss Preprocessing (TODO — Prerequisite for Stages 5 and 6)

As described in Section 2.2, implement the Dark Channel Prior dehazing in LAB colour space. After implementation, re-run the full pipeline on all 352 pairs and compare the following metrics before and after DCP:

- NCC (expect increase due to better intensity normalisation)
- SSIM (expect marginal increase)
- InlierCount (expect increase due to more detectable keypoints in dehazed Zeiss images)

Save the comparison as two columns in a summary table. If InlierCount does not improve, the DCP parameters (patch size, omega, guided filter radius) should be tuned.

### 7.2 U-Net Vessel Segmentation for Clarus (TODO — Prerequisite for Stage 5A)

The Frangi filter misses fine tertiary and quaternary vessels that a learned segmentation model can detect. A U-Net pretrained on the DRIVE retinal vessel segmentation dataset must be integrated.

**Implementation steps:**

1. Download a publicly available DRIVE-pretrained U-Net from GitHub (multiple repositories exist; select one with published Dice score above 0.78 on DRIVE test set).
2. The U-Net expects input images of a fixed size (typically 584×565 or 512×512 depending on the repository). Resize each Clarus image to the expected input size, preserving aspect ratio with zero-padding if necessary.
3. Run inference on all 352 Clarus images. The output is a single-channel probability map where each pixel value represents the probability of being a vessel.
4. Threshold the probability map at 0.5 to obtain a binary vessel mask.
5. Resize the binary vessel mask back to the original Clarus image dimensions.
6. Save the vessel map alongside each Clarus image as `<pair>/clarus_unet_vessel.png`.
7. Update `VesselDensity_clarus` in the metrics block to use the U-Net vessel map instead of the Frangi map. This provides a more accurate target vessel density for computing VesselRecoveryRatio.

### 7.3 Stage 5A — Super-Resolution Enhancement with SFT-Conditioned Real-ESRGAN (TODO — Main Enhancement Component)

This is the core remaining component. The goal is to take the warped Zeiss patch (which shows only primary and secondary vessels) and produce a version with Clarus-level vessel detail, using the Clarus vessel map as a structural prior.

#### 7.3.1 Why This Approach Works Without Synthetic Data

Traditional super-resolution requires paired low-resolution / high-resolution training data, which is typically generated synthetically by downsampling high-resolution images. In this project, the registration itself provides the pairing naturally. For each of the 352 pairs:

- The **warped Zeiss patch** (from `registered_zeiss.png`) is the low-quality input.
- The **Clarus image cropped to the overlap region** is the high-quality ground truth target.
- The **U-Net vessel map of the Clarus** provides a spatial conditioning signal that tells the SR model exactly where vessels should anatomically appear.

No synthetic degradation is needed. The real degradation comes from the physical difference between the two cameras.

#### 7.3.2 Architecture Details

**Base model:** Real-ESRGAN. Download the official pretrained weights from the Real-ESRGAN GitHub repository (`RealESRGAN_x4plus.pth` or equivalent). This model uses an RRDB (Residual-in-Residual Dense Block) backbone, which is a deep residual network with dense connections within each block.

**Modification — Spatial Feature Transform (SFT) layers:** SFT layers are inserted into each RRDB block. An SFT layer applies an affine transformation to the feature maps at each spatial location, conditioned on an external input (the vessel map). Specifically, for feature map `F` and conditioning map `C`:

```
F_out = gamma(C) * F + beta(C)
```

where `gamma` and `beta` are learned convolutional networks that take the conditioning map `C` and output per-pixel scale and shift parameters. This allows the model to amplify features along vessel paths (where gamma is high) and suppress features elsewhere.

**Implementation of SFT insertion:**

1. For each RRDB block in the backbone, add two small convolutional networks (2–3 layers each, 64 channels) that process the conditioning map (Clarus vessel map downsampled to the feature resolution of that block) and output gamma and beta tensors of the same spatial size as the feature map.
2. After each RRDB block's residual output, apply the SFT modulation: multiply element-wise by gamma, add beta.
3. The conditioning map must be downsampled to match the spatial resolution at each RRDB block level. Use bilinear interpolation for this downsampling.

**Conditioning input:** The Clarus U-Net vessel segmentation map (binary, single-channel) cropped to the overlap region and resized to match the input patch dimensions. During training, this is the ground-truth vessel map. During inference on new images, this same map is available because the Clarus image is always present as the reference canvas.

#### 7.3.3 Training Dataset Construction

From the 352 registered pairs, construct the training dataset as follows:

1. **Compute the overlap mask** for each pair. This is the intersection of the warped Zeiss non-zero mask and the Clarus fundus mask. This defines the region where both images have valid content.

2. **Extract the bounding box** of the overlap mask. Crop both the warped Zeiss image and the Clarus image to this bounding box. This removes the large black margins and focuses the model on the content region.

3. **Resize both crops** to a fixed training resolution. A reasonable choice is 256×256 pixels for initial training, scaling to 512×512 once the model converges at the lower resolution. Use `cv2.INTER_CUBIC` for the Clarus crop (preserving detail) and `cv2.INTER_AREA` for the Zeiss crop (reducing aliasing during downscale).

4. **Crop the Clarus U-Net vessel map** to the same bounding box and resize to the same training resolution using `cv2.INTER_NEAREST` (to preserve binary values).

5. For each pair, the training sample is a triplet:
   - **Input:** Warped Zeiss crop (3-channel BGR, normalised to [0, 1])
   - **Conditioning:** Clarus U-Net vessel map crop (1-channel binary, values 0 or 1)
   - **Target:** Clarus crop (3-channel BGR, normalised to [0, 1])

6. **Data augmentation:** Apply random horizontal flips, random vertical flips, random 90-degree rotations, and random brightness/contrast jitter (±10%) to the input and target simultaneously (with identical spatial transforms applied to both). The conditioning map receives the same spatial transforms but not the intensity augmentation.

7. **Train/validation split:** Use 300 pairs for training and 52 pairs for validation. The split should be at the patient level — all images from a given patient must be in the same split to prevent data leakage. Shuffle patients randomly with a fixed seed (e.g., seed=42) before splitting.

#### 7.3.4 Training Loss Function

The total loss is a weighted sum of four components:

**Component 1 — L1 Pixel Loss:**
```
L_pixel = mean(|SR_output - Clarus_target|)
```
This enforces pixel-level reconstruction fidelity. It is the primary driver of convergence in early training. Weight: `lambda_pixel = 1.0`.

**Component 2 — VGG Perceptual Loss:**
```
L_perceptual = sum over layers l of (mean(|VGG_l(SR_output) - VGG_l(Clarus_target)|))
```
Use a pretrained VGG-19 network (frozen weights, downloaded from torchvision). Extract feature maps from layers `conv1_2`, `conv2_2`, `conv3_4`, `conv4_4`, and `conv5_4`. Compute the L1 distance between the feature maps of the SR output and the Clarus target at each layer, then sum. This loss encourages the SR output to have similar high-level structural features (edges, textures, vessel patterns) to the Clarus target, even when pixel-level alignment is slightly imperfect. Weight: `lambda_perceptual = 0.1`.

**Component 3 — Adversarial Loss:**
```
L_adversarial = -mean(log(D(SR_output)))
```
A PatchGAN discriminator `D` is trained alongside the generator (the SFT-conditioned Real-ESRGAN). The discriminator receives either the SR output or the real Clarus target and must classify each overlapping patch as real or fake. The generator loss encourages the discriminator to classify SR outputs as real, pushing the SR model toward photorealistic vessel textures rather than blurry averages. The discriminator is trained with the standard binary cross-entropy loss on real vs fake patches. Weight for the generator adversarial loss: `lambda_adversarial = 0.01`. The discriminator uses a learning rate of `1e-4`, and the generator uses `1e-4` for SFT layers and `0` for frozen RRDB backbone weights.

**Component 4 — Vessel Segmentation Loss:**
```
L_vessel = mean(|UNet(SR_output) - UNet(Clarus_target)|)
```
Run the same DRIVE-pretrained U-Net (frozen weights) on both the SR output and the Clarus target. Compute the L1 distance between their vessel probability maps. This loss directly penalises the SR model if its output does not produce the same vessel segmentation as the Clarus target when passed through the U-Net. It forces the model to hallucinate anatomically plausible vessels in the correct locations rather than generic texture. Weight: `lambda_vessel = 0.5`.

**Total loss:**
```
L_total = lambda_pixel * L_pixel + lambda_perceptual * L_perceptual + lambda_adversarial * L_adversarial + lambda_vessel * L_vessel
```

#### 7.3.5 Training Procedure

1. **Freeze the entire RRDB backbone.** Load the pretrained Real-ESRGAN weights into the RRDB blocks and set `requires_grad = False` for all parameters in these blocks. This preserves the powerful feature extraction capability learned from millions of natural images.

2. **Initialise the SFT layers** (gamma and beta convolutional networks) with small random weights. Specifically, initialise gamma's final layer bias to 1.0 and beta's final layer bias to 0.0 so that the initial SFT modulation is an identity transform and the model starts from the pretrained Real-ESRGAN behaviour.

3. **Initialise the upsampler head** (the final convolutional layers after the RRDB backbone that produce the output image) with the pretrained weights but set `requires_grad = True` so they can adapt to retinal image statistics.

4. **Optimiser:** Adam with learning rate `1e-4`, betas `(0.9, 0.999)`, weight decay `1e-5`. Apply a cosine annealing learning rate schedule with warm restarts (period = 50 epochs).

5. **Batch size:** 8 (adjust based on available GPU memory; with 256×256 patches and a single GPU with 12GB VRAM, batch size 8 should be feasible).

6. **Training epochs:** 200 epochs. Monitor validation VesselRecoveryRatio every 5 epochs. Save a checkpoint whenever validation VesselRecoveryRatio improves.

7. **Validation monitoring:** After each validation epoch, compute VesselRecoveryRatio on the 52 held-out pairs. Specifically, run the SR model on each validation Zeiss crop, compute the Frangi vessel density of the SR output, divide by the U-Net vessel density of the corresponding Clarus target, and report the mean across all 52 pairs. Plot this over epochs to confirm convergence.

8. **Early stopping:** If validation VesselRecoveryRatio does not improve for 30 consecutive epochs, stop training and use the best checkpoint.

#### 7.3.6 Inference Procedure

Once trained, the inference pipeline for a single pair is:

1. Load the warped Zeiss image (`registered_zeiss.png`) and the Clarus image.
2. Compute the overlap mask and bounding box.
3. Crop the warped Zeiss to the overlap bounding box. Resize to the training resolution.
4. Load (or compute) the Clarus U-Net vessel map. Crop and resize to the same dimensions.
5. Pass the Zeiss crop as input and the vessel map as conditioning to the SFT-conditioned Real-ESRGAN. The output is the SR-enhanced Zeiss crop at the training resolution.
6. Resize the SR output back to the original overlap bounding box dimensions using `cv2.INTER_CUBIC`.
7. Place the SR output back into the full Clarus-canvas-sized image at the overlap bounding box location.
8. This produces the `sr_zeiss.png` image: the enhanced Zeiss content in Clarus coordinate space.

---

### 7.4 Stage 5B — FOV Extension and Gaussian Alpha Blending (TODO)

This stage composites the SR-enhanced Zeiss region onto the full Clarus canvas so that the final output image shows enhanced detail in the Zeiss footprint region and original Clarus detail everywhere else, with a visually seamless boundary between the two.

#### 7.4.1 Why This Is Needed

The SR-enhanced Zeiss patch occupies only a portion of the Clarus canvas (approximately 25–35% by area). Outside this region, the Clarus image provides the only available information. At the boundary of the Zeiss footprint, a hard cut between the SR output and the Clarus image would create a visible seam due to slight intensity and colour differences. Gaussian alpha blending creates a smooth transition zone that eliminates this artefact.

#### 7.4.2 Implementation Steps

1. **Compute the Zeiss footprint mask.** This is already available from the registration stage as the non-zero mask of the warped Zeiss image. Specifically:
   ```python
   warped_gray = cv2.cvtColor(sr_zeiss_full, cv2.COLOR_BGR2GRAY)
   footprint_mask = (warped_gray > 0).astype(np.uint8) * 255
   ```

2. **Create the alpha blending mask.** Apply Gaussian blur to the binary footprint mask to create a smooth gradient at the boundary:
   ```python
   alpha = cv2.GaussianBlur(footprint_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=sigma)
   ```
   The `sigma` parameter controls the width of the blend zone. A sigma of 30–50 pixels is recommended as the starting point. Larger sigma produces a wider, smoother transition but may dilute the SR enhancement near the edges. Smaller sigma keeps the enhancement sharp but risks visible seams.

3. **Expand alpha to 3 channels** for element-wise multiplication with BGR images:
   ```python
   alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
   ```

4. **Composite the final image:**
   ```python
   composite = alpha_3ch * sr_zeiss_full + (1.0 - alpha_3ch) * clarus_bgr
   composite = np.clip(composite, 0, 255).astype(np.uint8)
   ```
   Inside the Zeiss footprint (where alpha approaches 1.0), the SR-enhanced Zeiss content dominates. Outside the footprint (where alpha is 0.0), the Clarus content is used unchanged. In the transition zone, the two blend smoothly.

5. **Verify the boundary visually.** Save a diagnostic image showing the alpha mask as a heatmap overlay on the composite. Check that there are no visible intensity discontinuities across the blend zone. Specifically, sample pixel intensities along horizontal and vertical lines crossing the blend boundary and plot them to confirm a smooth gradient.

6. **Tune sigma.** Run the blending on 3–5 representative pairs with sigma values of 20, 30, 40, and 50 pixels. Select the sigma that produces the most visually seamless boundary across all test pairs. Once selected, apply the same sigma to all 352 pairs in the batch run.

#### 7.4.3 Output

The composite image is saved as `<pair>/composite_enhanced.png`. This is the final deliverable image for each pair: a full Clarus-FOV image where the Zeiss footprint region has been enhanced to show finer vessel detail.

---

### 7.5 Stage 6 — Full Evaluation and Final Output (TODO)

#### 7.5.1 Before-and-After Metric Comparison

After the SR enhancement and FOV extension stages are complete, re-run the metrics block on all 352 pairs, but this time computing metrics on **both** the original warped Zeiss and the SR-enhanced Zeiss. The following table summarises the expected values:

| Metric | Before Enhancement (Baseline) | After Enhancement (Target) |
|---|---|---|
| VesselRecoveryRatio | 0.3–0.5 | > 0.8 |
| SSIM in overlap region | 0.85–0.92 | > 0.90 |
| NCC in overlap region | 0.70–0.85 | > 0.85 |
| DiscOffset_px | < 15 px | Unchanged (registration not re-run) |
| VesselDensity_warped | Low (Zeiss captures few vessels) | Approaching VesselDensity_clarus |
| Visual vessel order | Primary and secondary only | Secondary, tertiary, quaternary visible |

The `DiscOffset_px` metric should remain unchanged because the registration homography is not re-estimated after enhancement — it only verifies that the original registration remains the foundation.

#### 7.5.2 VesselRecoveryRatio Tracking

VesselRecoveryRatio is the single most important metric for evaluating whether the SR enhancement is working. It should be tracked at three stages:

1. **Baseline (pre-enhancement):** Compute on all 352 pairs using the warped Zeiss images as-is. Report mean and standard deviation.
2. **Per-epoch during training:** Compute on the 52 validation pairs at each validation epoch. Plot as a learning curve.
3. **Final (post-enhancement):** Compute on all 352 pairs using the SR-enhanced Zeiss images. Report mean and standard deviation.

The improvement from baseline to final should be statistically significant. A paired t-test or Wilcoxon signed-rank test on the 352 VesselRecoveryRatio values (before vs after) should yield p < 0.001 for the enhancement to be considered successful.

#### 7.5.3 Failure Case Analysis

After the full batch run, identify pairs where the enhancement did not achieve VesselRecoveryRatio > 0.7. For each such failure case, inspect:

- **DiscOffset_px:** If greater than 30 pixels, the registration was poor and the SR model received a misaligned input. The fix is to improve registration for that pair, not to retrain the SR model.
- **InlierCount:** If below 6, the homography was degenerate. Same conclusion.
- **OverlapFraction:** If below 0.05, the warped Zeiss barely overlaps the Clarus canvas. The SR model had very little valid content to enhance.
- **Image pathology:** If the Zeiss image contains large haemorrhages, exudates, or laser scars, these can confuse the SR model. Such pairs may need to be excluded from the quantitative evaluation and reported separately.

#### 7.5.4 Final Output Files

For each of the 352 pairs, the complete set of output files after all stages will be:

| File | Contents |
|---|---|
| `registered_zeiss.png` | Warped Zeiss in Clarus canvas space (from Stage 3) |
| `overlay.png` | 50/50 blend of Clarus and warped Zeiss (from Stage 3) |
| `overlay_disc.png` | Disc detection diagnostic (from Stage 4) |
| `registration_steps.png` | 3×2 diagnostic panel (from Stage 3) |
| `matches.png` | Match line visualisation (from Stage 3) |
| `clarus_unet_vessel.png` | U-Net vessel map of Clarus (from Stage 7.2) |
| `sr_zeiss.png` | SR-enhanced Zeiss in Clarus canvas space (from Stage 5A) |
| `composite_enhanced.png` | Final blended composite with FOV extension (from Stage 5B) |

At the dataset level, the following aggregate files will be produced:

| File | Contents |
|---|---|
| `HYBRID_metrics.csv` | Per-pair metrics, 17 columns, 352 rows (from Stage 4) |
| `HYBRID_metrics_enhanced.csv` | Per-pair metrics after enhancement, same columns, 352 rows (from Stage 6) |
| `summary_statistics.txt` | Mean and std of all key metrics before and after, plus statistical test results |
| `vessel_recovery_curve.png` | Plot of VesselRecoveryRatio over training epochs for the validation set |

---

## 8. Known Issues and Limitations

### 8.1 Sparse, Spatially Clustered Inliers

As described in Section 4.4, matches are predominantly at the optic disc. The homography is therefore accurate at the disc but may drift at the image periphery. The InlierSpread metrics will quantify this. If InlierSpread_x and InlierSpread_y are consistently below 50 pixels across the dataset, future work should explore replacing the full 8-parameter homography with a 6-parameter affine transform anchored at the disc, or a 4-parameter similarity transform (rotation, scale, translation) if the imaging geometry permits it. Fewer parameters mean less overfitting to the clustered inlier set.

### 8.2 OverlapFraction Lower Than Expected

The observed overlap fraction of approximately 7% is substantially lower than the theoretical 25–35%. Two possible causes have been identified:

1. **Non-central gaze position during Zeiss acquisition.** If the patient was looking off-centre during the Zeiss capture, the Zeiss FOV may not be centred on the same anatomical region as the Clarus FOV. The overlap between the two would then be smaller than the geometric maximum.

2. **Homography placing Zeiss footprint partially off-canvas.** If the homography overestimates the scale or introduces excessive translation, the warped Zeiss content may extend beyond the Clarus image boundaries and be clipped.

**Recommended diagnostic:** Before proceeding to SR training, visually inspect `overlay.png` for at least 10 representative pairs spanning the range of OverlapFraction values in the CSV. Determine whether the Zeiss content is correctly localised (visible on the Clarus canvas) or partially off-screen. If the latter, the homography estimation parameters (RANSAC threshold, minimum inlier count) may need adjustment.

### 8.3 DCP Dehazing Not Yet Implemented

As detailed in Section 7.1, the Dark Channel Prior dehazing for Zeiss images is described in the pipeline design but not yet present in the code. This should be implemented before the SR training dataset is constructed, because the dehazed Zeiss images will be the inputs to the SR model. Training the SR model on hazed images and then switching to dehazed images at inference (or vice versa) would create a domain mismatch.

### 8.4 U-Net Vessel Segmentation Not Yet Integrated

As detailed in Section 7.2, the Frangi filter currently used for vessel density computation is a reasonable proxy but is known to miss fine tertiary and quaternary vessels that a learned U-Net model would detect. Since VesselRecoveryRatio is the primary enhancement metric, and it depends on VesselDensity_clarus (which depends on the vessel segmentation method), the U-Net integration should be completed before any VesselRecoveryRatio values are treated as final baselines.

### 8.5 352-Pair Dataset Size for SR Training

The training dataset consists of only 300 pairs (after the 300/52 train/validation split). This is a small dataset for training a super-resolution model, even with frozen backbone weights and only SFT layers being learned. Data augmentation (flips, rotations, colour jitter) is essential to prevent overfitting. Additionally, extracting multiple overlapping crops from each pair (rather than a single centre crop) can multiply the effective training set size. For instance, extracting 4 overlapping 256×256 crops per pair would yield 1200 training samples. Monitor the gap between training loss and validation loss to detect overfitting; if the gap grows consistently, increase augmentation intensity or reduce model capacity (fewer channels in SFT layers).

---

## 9. Recommended Execution Order — Summary

The following is the exact sequence of implementation and execution steps, ordered by dependency:

**Step 1:** Run `fixed_trial4_updated.py` on all 352 pairs in its current state. Review `HYBRID_metrics.csv`. Check that DiscOffset_px is below 30 pixels for the majority of pairs. Flag pairs where InlierCount is below 6 or InlierSpread is below 30 pixels. Visually inspect `overlay.png` for approximately 10 representative pairs to confirm the registration is placing the Zeiss content in the correct anatomical region on the Clarus canvas. This step validates the existing completed stages before building on top of them.

**Step 2:** Implement DCP dehazing in `preprocess_image_for_detection()` for Zeiss images. Apply Dark Channel Prior in LAB colour space as described in Section 2.2, followed by the existing CLAHE step. Re-run the full pipeline on all 352 pairs. Compare NCC, SSIM, and InlierCount before and after DCP to confirm improvement.

**Step 3:** Download and integrate a DRIVE-pretrained U-Net for Clarus vessel segmentation. Run inference on all 352 Clarus images. Save vessel maps. Update VesselDensity_clarus in the metrics block to use U-Net output. Re-run metrics and record the updated baseline VesselRecoveryRatio values.

**Step 4:** Construct the SR training dataset from the 352 registered pairs. For each pair, extract the warped Zeiss crop, the Clarus crop, and the Clarus U-Net vessel map crop over the overlap region. Resize to the training resolution. Apply augmentation. Split into 300 training and 52 validation pairs at the patient level.

**Step 5:** Implement the SFT-conditioned Real-ESRGAN architecture. Load pretrained RRDB weights and freeze them. Initialise SFT layers. Implement the four-component loss function (L1, VGG perceptual, adversarial, vessel segmentation). Train for up to 200 epochs, monitoring validation VesselRecoveryRatio.

**Step 6:** Implement FOV extension with Gaussian alpha blending. Tune sigma on representative pairs. Run on all 352 pairs to produce composite images.

**Step 7:** Run the full evaluation. Compute all metrics on both the original and enhanced images. Produce summary statistics, before-and-after comparisons, statistical significance tests, and failure case analysis. Generate the final output files and aggregate reports.