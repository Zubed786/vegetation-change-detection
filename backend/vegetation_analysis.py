"""
Vegetation Analysis Module.
Combines ChangeFormer generic change detections with T1 and T2 NDVI masks to classify:
- VEGETATION LOSS (T1 vegetation -> T2 non-vegetation within detected change)
- VEGETATION GAIN (T1 non-vegetation -> T2 vegetation within detected change)
- STABLE VEGETATION (T1 vegetation and T2 vegetation)
- OTHER CHANGE (Change detected by ChangeFormer that is non-vegetation)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import ndimage


@dataclass
class VegetationMasks:
    t1_veg_mask: np.ndarray       # boolean (True = vegetation at T1)
    t2_veg_mask: np.ndarray       # boolean (True = vegetation at T2)
    veg_loss_mask: np.ndarray     # boolean (T1 veg & not T2 veg & change)
    veg_gain_mask: np.ndarray     # boolean (not T1 veg & T2 veg & change)
    stable_veg_mask: np.ndarray   # boolean (T1 veg & T2 veg & not change)
    other_change_mask: np.ndarray # boolean (change & not (veg_loss or veg_gain))
    ndvi_threshold: float
    min_region_size: int
    t1_veg_pixels: int
    t2_veg_pixels: int
    loss_pixels: int
    gain_pixels: int
    stable_pixels: int


def classify_vegetation_changes(
    t1_ndvi: np.ndarray,
    t2_ndvi: np.ndarray,
    change_mask: np.ndarray,
    ndvi_threshold: float = 0.30,
    min_region_size: int = 16,
) -> VegetationMasks:
    """
    Classifies vegetation states and transitions using true NDVI and ChangeFormer change signals.
    """
    # 1. Vegetation thresholding
    t1_veg = t1_ndvi >= ndvi_threshold
    t2_veg = t2_ndvi >= ndvi_threshold

    # Binary change mask from ChangeFormer
    is_change = change_mask > 0

    # 2. Conjunction logic
    # Vegetation Loss: was vegetation at T1, lost at T2
    raw_loss = t1_veg & (~t2_veg)

    # Vegetation Gain: was not vegetation at T1, became vegetation at T2
    raw_gain = (~t1_veg) & t2_veg

    # Stable vegetation: vegetation at both time points
    stable_veg = t1_veg & t2_veg

    # Other change: detected change that does not involve vegetation gain or loss (e.g. soil/pavement/building)
    other_change = is_change & (~raw_loss) & (~raw_gain)

    # 3. Morphological filtering to clean isolated speckle noise
    def filter_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
        if min_size <= 1:
            return mask
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask
        sizes = ndimage.sum(mask, labeled, range(num_features + 1))
        mask_sizes = sizes < min_size
        remove_pixel = mask_sizes[labeled]
        cleaned = mask.copy()
        cleaned[remove_pixel] = False
        return cleaned

    veg_loss = filter_small_components(raw_loss, min_region_size)
    veg_gain = filter_small_components(raw_gain, min_region_size)
    other_change_clean = filter_small_components(other_change, min_region_size)

    return VegetationMasks(
        t1_veg_mask=t1_veg,
        t2_veg_mask=t2_veg,
        veg_loss_mask=veg_loss,
        veg_gain_mask=veg_gain,
        stable_veg_mask=stable_veg,
        other_change_mask=other_change_clean,
        ndvi_threshold=ndvi_threshold,
        min_region_size=min_region_size,
        t1_veg_pixels=int(np.count_nonzero(t1_veg)),
        t2_veg_pixels=int(np.count_nonzero(t2_veg)),
        loss_pixels=int(np.count_nonzero(veg_loss)),
        gain_pixels=int(np.count_nonzero(veg_gain)),
        stable_pixels=int(np.count_nonzero(stable_veg)),
    )


def compute_visible_vegetation_mask(rgb_img: np.ndarray) -> np.ndarray:
    """
    Computes a boolean vegetation mask from 3-channel RGB imagery using
    visible spectrum indices (VARI, ExG, and GLI) combined with spectral constraints.
    Accurately identifies trees, green canopy, lawns, shrubs, and agricultural crops
    while rejecting roofs (orange/brown/red), concrete/asphalt (neutral gray), water, and shadows.
    """
    arr = rgb_img.astype(np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total = R + G + B + 1e-6
    r, g, b = R / total, G / total, B / total

    # 1. Visible Atmospherically Resistant Index (VARI) = (G - R) / (G + R - B)
    denom = G + R - B
    denom = np.where(np.abs(denom) < 1.0, np.sign(denom) * 1.0 + 1e-5, denom)
    vari = np.clip((G - R) / denom, -1.0, 1.0)

    # 2. Green Leaf Index (GLI) = (2*G - R - B) / (2*G + R + B)
    gli = (2.0 * G - R - B) / (2.0 * G + R + B + 1e-6)

    # 3. Excess Green (ExG) = 2*g - r - b
    exg = 2.0 * g - r - b

    brightness = total / 3.0

    # Vegetation classification rules:
    is_green = (
        ((G > R + 2) & (G >= B))
        | (gli > 0.04)
        | ((vari > 0.05) & (G >= R))
        | ((exg > 0.03) & (G > R))
    )
    not_roof = R <= G + 5
    not_water = B <= G + 15
    not_concrete = ~((np.abs(R - G) <= 4) & (np.abs(G - B) <= 4) & (brightness > 110))
    not_shadow = brightness > 15

    return is_green & not_roof & not_water & not_concrete & not_shadow


def classify_visible_vegetation_changes(
    t1_rgb: np.ndarray,
    t2_rgb: np.ndarray,
    change_mask: np.ndarray,
    min_region_size: int = 16,
) -> VegetationMasks:
    """
    Classifies vegetation states and transitions on 3-channel RGB imagery
    using visible spectrum vegetation indices (VARI/ExG/GLI) within ChangeFormer change signals.
    """
    t1_veg = compute_visible_vegetation_mask(t1_rgb)
    t2_veg = compute_visible_vegetation_mask(t2_rgb)

    is_change = change_mask > 0

    # Conjunction logic:
    # Vegetation Loss: was vegetation at T1, lost at T2
    raw_loss = t1_veg & (~t2_veg)

    # Vegetation Gain: was not vegetation at T1, became vegetation at T2
    raw_gain = (~t1_veg) & t2_veg

    # Stable vegetation: vegetation at both time points
    stable_veg = t1_veg & t2_veg

    # Other change: detected surface change by ChangeFormer that does not involve vegetation gain or loss
    other_change = is_change & (~raw_loss) & (~raw_gain)

    def filter_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
        if min_size <= 1:
            return mask
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask
        sizes = ndimage.sum(mask, labeled, range(num_features + 1))
        mask_sizes = sizes < min_size
        remove_pixel = mask_sizes[labeled]
        cleaned = mask.copy()
        cleaned[remove_pixel] = False
        return cleaned

    veg_loss = filter_small_components(raw_loss, min_region_size)
    veg_gain = filter_small_components(raw_gain, min_region_size)
    other_change_clean = filter_small_components(other_change, min_region_size)

    return VegetationMasks(
        t1_veg_mask=t1_veg,
        t2_veg_mask=t2_veg,
        veg_loss_mask=veg_loss,
        veg_gain_mask=veg_gain,
        stable_veg_mask=stable_veg,
        other_change_mask=other_change_clean,
        ndvi_threshold=0.0,
        min_region_size=min_region_size,
        t1_veg_pixels=int(np.count_nonzero(t1_veg)),
        t2_veg_pixels=int(np.count_nonzero(t2_veg)),
        loss_pixels=int(np.count_nonzero(veg_loss)),
        gain_pixels=int(np.count_nonzero(veg_gain)),
        stable_pixels=int(np.count_nonzero(stable_veg)),
    )

