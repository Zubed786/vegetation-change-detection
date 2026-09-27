"""
NDVI (Normalized Difference Vegetation Index) Analytical Module.
Computes true physical NDVI for Sentinel-2 multispectral imagery:
    NDVI = (B8 - B4) / (B8 + B4)
Strictly handles nodata, NaN, infinite values, zero denominators, and reflectance scaling.
Does NOT fabricate pseudo-NDVI on RGB imagery.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


class MultispectralBandsRequiredError(Exception):
    """Raised when an attempt is made to compute NDVI without valid Red (B4) and NIR (B8) spectral bands."""
    pass


@dataclass
class NDVIResult:
    ndvi: np.ndarray             # 2D float32 array in [-1.0, 1.0]
    valid_mask: np.ndarray       # 2D boolean array of valid computed pixels
    mean_ndvi: float             # Mean of valid pixels
    min_ndvi: float
    max_ndvi: float
    scaling_detected: str        # e.g., "0-10000_surface_reflectance", "0-1_normalized", "0-255_scaled"


def detect_and_normalize_reflectance(band_array: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Detects reflectance scale without blindly assuming a fixed factor.
    Sentinel-2 Level-1C/Level-2A products typically use 10,000 scale (BOA / BOA reflectance).
    Standard 8-bit rasters use 0-255.
    Normalized floating point rasters use 0.0-1.0.
    """
    finite_vals = band_array[np.isfinite(band_array) & (band_array > 0)]
    if finite_vals.size == 0:
        return band_array.astype(np.float32), "empty_or_zero"

    max_val = float(np.percentile(finite_vals, 99))

    if max_val > 1000.0:
        # Sentinel-2 standard DN scale (10,000 = 1.0 reflectance)
        normalized = band_array.astype(np.float32) / 10000.0
        return normalized, "0-10000_surface_reflectance"
    elif max_val > 1.5:
        # 8-bit or integer scaled raster
        normalized = band_array.astype(np.float32) / 255.0
        return normalized, "0-255_scaled"
    else:
        # Already in 0.0 - 1.0 range
        return band_array.astype(np.float32), "0-1_normalized"


def calculate_ndvi(
    b4_red: Optional[np.ndarray],
    b8_nir: Optional[np.ndarray],
    nodata_val: Optional[float] = None,
) -> NDVIResult:
    """
    Computes Normalized Difference Vegetation Index (NDVI).
    NDVI = (B8 - B4) / (B8 + B4)
    Requires both Red (B4) and Near-Infrared (B8) spectral bands.
    """
    if b4_red is None or b8_nir is None:
        raise MultispectralBandsRequiredError(
            "NDVI calculation requires both Red (B4) and Near-Infrared (B8) spectral bands. "
            "Optical RGB imagery does not possess NIR, so actual NDVI cannot be calculated."
        )

    if b4_red.shape != b8_nir.shape:
        raise ValueError(
            f"Shape mismatch between Red band {b4_red.shape} and NIR band {b8_nir.shape}."
        )

    # Detect scaling on both bands
    norm_red, scale_red = detect_and_normalize_reflectance(b4_red)
    norm_nir, scale_nir = detect_and_normalize_reflectance(b8_nir)

    # Mask invalid values (NaN, Inf, nodata, negative reflectance)
    invalid_mask = (
        np.isnan(norm_red)
        | np.isnan(norm_nir)
        | np.isinf(norm_red)
        | np.isinf(norm_nir)
        | (norm_red < 0)
        | (norm_nir < 0)
    )

    if nodata_val is not None:
        invalid_mask |= (b4_red == nodata_val) | (b8_nir == nodata_val)

    # Denominator calculation
    denom = norm_nir + norm_red
    zero_denom_mask = denom <= 1e-6

    valid_mask = (~invalid_mask) & (~zero_denom_mask)

    # Safe division
    diff = norm_nir - norm_red
    ndvi = np.zeros_like(diff, dtype=np.float32)
    ndvi[valid_mask] = diff[valid_mask] / denom[valid_mask]

    # Clip to theoretical physical limits [-1.0, 1.0]
    ndvi = np.clip(ndvi, -1.0, 1.0)
    # Re-zero invalid pixels
    ndvi[~valid_mask] = 0.0

    valid_pixels = ndvi[valid_mask]
    if valid_pixels.size > 0:
        mean_ndvi = float(np.mean(valid_pixels))
        min_ndvi = float(np.min(valid_pixels))
        max_ndvi = float(np.max(valid_pixels))
    else:
        mean_ndvi = 0.0
        min_ndvi = 0.0
        max_ndvi = 0.0

    return NDVIResult(
        ndvi=ndvi,
        valid_mask=valid_mask,
        mean_ndvi=round(mean_ndvi, 4),
        min_ndvi=round(min_ndvi, 4),
        max_ndvi=round(max_ndvi, 4),
        scaling_detected=f"Red: {scale_red}, NIR: {scale_nir}",
    )
