"""PyTorch Dataset for paired Zeiss-Clarus retinal images."""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RetinalImageDataset(Dataset):
    """
    Dataset for paired Zeiss Visuscout and Clarus fundus images.
    
    Loads paired images with their overlap masks and vessel segmentation maps.
    Applies preprocessing (DCP dehazing for Zeiss) and data augmentation.
    """
    
    def __init__(
        self,
        pairs_file: str,
        clarus_vessel_maps_dir: Optional[str] = None,
        patch_size: int = 256,
        transform: Optional[A.Compose] = None,
        apply_dehazing: bool = True,
        mode: str = 'train'
    ):
        """
        Initialize dataset.
        
        Args:
            pairs_file: CSV file with columns: zeiss_path, clarus_path, overlap_bbox
            clarus_vessel_maps_dir: Directory containing U-Net vessel maps for Clarus images
            patch_size: Target size for image patches
            transform: Albumentations transform pipeline
            apply_dehazing: Whether to apply DCP dehazing to Zeiss images
            mode: 'train' or 'val'
        """
        self.pairs_df = pd.read_csv(pairs_file)
        self.clarus_vessel_maps_dir = Path(clarus_vessel_maps_dir) if clarus_vessel_maps_dir else None
        self.patch_size = patch_size
        self.transform = transform
        self.apply_dehazing = apply_dehazing
        self.mode = mode
        
        print(f"Loaded {len(self.pairs_df)} pairs for {mode} mode")
    
    def __len__(self) -> int:
        return len(self.pairs_df)
    
    def load_image(self, path: str) -> np.ndarray:
        """Load image in BGR format."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img
    
    def compute_overlap_mask(self, warped_zeiss: np.ndarray, clarus_mask: np.ndarray) -> np.ndarray:
        """
        Compute overlap region between warped Zeiss and Clarus.
        
        Args:
            warped_zeiss: Warped Zeiss image in Clarus coordinate space
            clarus_mask: Clarus fundus mask
            
        Returns:
            Binary overlap mask
        """
        # Create mask of non-zero regions in warped Zeiss
        zeiss_mask = cv2.cvtColor(warped_zeiss, cv2.COLOR_BGR2GRAY) > 0
        
        # Intersection of both masks
        overlap_mask = (zeiss_mask & (clarus_mask > 0)).astype(np.uint8) * 255
        
        return overlap_mask
    
    def extract_overlap_crop(
        self, 
        warped_zeiss: np.ndarray, 
        clarus: np.ndarray,
        vessel_map: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract and resize crops from overlap region.
        
        Args:
            warped_zeiss: Warped Zeiss image
            clarus: Clarus image
            vessel_map: Clarus vessel segmentation map (optional)
            
        Returns:
            Tuple of (zeiss_crop, clarus_crop, overlap_mask_crop, vessel_map_crop)
        """
        # Create overlap mask
        gray_warped = cv2.cvtColor(warped_zeiss, cv2.COLOR_BGR2GRAY)
        overlap_mask = (gray_warped > 0).astype(np.uint8) * 255
        
        # Find bounding box of overlap region
        contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No overlap found, return full images
            h, w = clarus.shape[:2]
            bbox = (0, 0, w, h)
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox = (x, y, x + w, y + h)
        
        x1, y1, x2, y2 = bbox
        
        # Crop all images to bounding box
        zeiss_crop = warped_zeiss[y1:y2, x1:x2]
        clarus_crop = clarus[y1:y2, x1:x2]
        overlap_crop = overlap_mask[y1:y2, x1:x2]
        
        vessel_crop = None
        if vessel_map is not None:
            vessel_crop = vessel_map[y1:y2, x1:x2]
        
        # Resize to target patch size
        zeiss_crop = cv2.resize(zeiss_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        clarus_crop = cv2.resize(clarus_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        overlap_crop = cv2.resize(overlap_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        
        if vessel_crop is not None:
            vessel_crop = cv2.resize(vessel_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        
        return zeiss_crop, clarus_crop, overlap_crop, vessel_crop
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns dictionary with:
            - input: Zeiss crop (3, H, W) - float32, normalized to [0, 1]
            - target: Clarus crop (3, H, W) - float32, normalized to [0, 1]
            - condition: Vessel map (1, H, W) - binary 0/1
            - overlap_mask: Overlap mask (1, H, W) - binary 0/1
        """
        row = self.pairs_df.iloc[idx]
        
        # Load images
        warped_zeiss = self.load_image(row['zeiss_warped_path'])
        clarus = self.load_image(row['clarus_path'])
        
        # Load vessel map if available
        vessel_map = None
        if self.clarus_vessel_maps_dir is not None:
            vessel_map_path = self.clarus_vessel_maps_dir / f"{row['pair_id']}_vessel.png"
            if vessel_map_path.exists():
                vessel_map = cv2.imread(str(vessel_map_path), cv2.IMREAD_GRAYSCALE)
        
        # Extract overlap crops
        zeiss_crop, clarus_crop, overlap_crop, vessel_crop = self.extract_overlap_crop(
            warped_zeiss, clarus, vessel_map
        )
        
        # Apply augmentation if in train mode
        if self.transform is not None and self.mode == 'train':
            # Albumentations expects numpy arrays in HWC format
            transformed = self.transform(
                image=clarus_crop,
                image0=zeiss_crop,
                mask=vessel_crop if vessel_crop is not None else np.zeros_like(overlap_crop)
            )
            clarus_crop = transformed['image']
            zeiss_crop = transformed['image0']
            if vessel_crop is not None:
                vessel_crop = transformed['mask']

        # Convert to torch tensors
        # Normalize images to [0, 1]
        zeiss_tensor = torch.from_numpy(zeiss_crop).permute(2, 0, 1).float() / 255.0
        clarus_tensor = torch.from_numpy(clarus_crop).permute(2, 0, 1).float() / 255.0

        # Vessel map as binary mask
        if vessel_crop is not None:
            vessel_tensor = torch.from_numpy(vessel_crop).unsqueeze(0).float() / 255.0
        else:
            vessel_tensor = torch.zeros(1, self.patch_size, self.patch_size)

        # Overlap mask
        overlap_tensor = torch.from_numpy(overlap_crop).unsqueeze(0).float() / 255.0

        return {
            'input': zeiss_tensor,          # (3, H, W) - Zeiss input
            'target': clarus_tensor,        # (3, H, W) - Clarus target
            'condition': vessel_tensor,     # (1, H, W) - Vessel map for SFT conditioning
            'overlap_mask': overlap_tensor, # (1, H, W) - Valid region mask
            'pair_id': row['pair_id'],
        }


def get_train_transform(patch_size: int = 256) -> A.Compose:
    """
    Get training data augmentation pipeline.

    Args:
        patch_size: Target patch size

    Returns:
        Albumentations transform
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.0,
            hue=0.0,
            p=0.3
        ),
        # Already resized, no need for resize here
    ], additional_targets={'image0': 'image'})


def get_val_transform() -> A.Compose:
    """Get validation transform (no augmentation)."""
    return A.Compose([], additional_targets={'image0': 'image'})


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    vessel_maps_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    patch_size: int = 256
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_csv: Path to training pairs CSV
        val_csv: Path to validation pairs CSV
        vessel_maps_dir: Directory with vessel segmentation maps
        batch_size: Batch size
        num_workers: Number of data loading workers
        patch_size: Target patch size

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = RetinalImageDataset(
        pairs_file=train_csv,
        clarus_vessel_maps_dir=vessel_maps_dir,
        patch_size=patch_size,
        transform=get_train_transform(patch_size),
        mode='train'
    )

    val_dataset = RetinalImageDataset(
        pairs_file=val_csv,
        clarus_vessel_maps_dir=vessel_maps_dir,
        patch_size=patch_size,
        transform=get_val_transform(),
        mode='val'
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

