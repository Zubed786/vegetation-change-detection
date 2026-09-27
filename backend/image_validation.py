"""
Image Validation Module for Bi-Temporal Remote Sensing Analysis.
Validates formats, readability, spatial dimensions, band consistency,
CRS/geospatial metadata compatibility, and valid-pixel coverage.
Rejects incompatible pairs without arbitrary resizing, cropping, or warping.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class ImageValidationError(Exception):
    """Raised when an image or bi-temporal image pair fails validation."""
    pass


@dataclass
class ImageMetadata:
    path: str
    format: str
    width: int
    height: int
    num_bands: int
    dtype: str
    has_geospatial: bool
    crs_epsg: Optional[int] = None
    crs_wkt: Optional[str] = None
    transform: Optional[Tuple[float, ...]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    valid_pixel_percentage: float = 100.0
    detected_mode: str = "optical_rgb"  # "sentinel2_multispectral" or "optical_rgb"
    b4_red_index: Optional[int] = None   # 0-indexed
    b8_nir_index: Optional[int] = None   # 0-indexed


@dataclass
class PairValidationResult:
    is_compatible: bool
    mode: str
    width: int
    height: int
    t1_meta: ImageMetadata
    t2_meta: ImageMetadata
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def inspect_single_image(file_path: str) -> ImageMetadata:
    """
    Inspects a single image file using Rasterio if applicable, falling back to PIL.
    Determines dimensions, bands, valid pixel ratio, geospatial metadata,
    and whether Sentinel-2 multispectral bands (Red/NIR) are present.
    """
    p = Path(file_path)
    if not p.exists():
        raise ImageValidationError(f"File not found: {file_path}")

    # Check file extension
    ext = p.suffix.lower()
    valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    if ext not in valid_exts:
        raise ImageValidationError(f"Unsupported file format '{ext}'. Supported formats: {valid_exts}")

    # Try opening with rasterio first
    if HAS_RASTERIO and ext in {".tif", ".tiff"}:
        try:
            with rasterio.open(file_path) as src:
                width = src.width
                height = src.height
                num_bands = src.count
                dtype = str(src.dtypes[0])
                crs = src.crs
                transform = tuple(src.transform) if src.transform else None
                bounds = tuple(src.bounds) if src.bounds else None

                crs_epsg = crs.to_epsg() if crs else None
                crs_wkt = crs.to_wkt() if crs else None
                has_geospatial = bool(crs and transform)

                # Check valid pixels (sample or full array for manageable size)
                # Read first band to estimate valid/nodata ratio
                arr = src.read(1)
                nodata = src.nodata
                if nodata is not None:
                    valid_mask = (arr != nodata) & (~np.isnan(arr))
                else:
                    valid_mask = ~np.isnan(arr)
                valid_pixel_pct = float(np.count_nonzero(valid_mask) / arr.size * 100.0)

                # Check band descriptions or band count to identify Sentinel-2 bands
                b4_idx = None
                b8_idx = None
                descriptions = [d.upper() if d else "" for d in (src.descriptions or [])]

                for i, desc in enumerate(descriptions):
                    if "B4" in desc or "RED" in desc:
                        b4_idx = i
                    elif "B8" in desc or "NIR" in desc:
                        b8_idx = i

                # Standard Sentinel-2 conventions if not explicitly described:
                # If 4 bands: [B2(Blue), B3(Green), B4(Red), B8(NIR)] -> b4=2, b8=3
                # If 12/13 bands: B4 is band 4 (index 3), B8 is band 8 (index 7)
                # If 2 bands: assumed [Red, NIR] -> b4=0, b8=1
                if b4_idx is None or b8_idx is None:
                    if num_bands == 4:
                        b4_idx = 2
                        b8_idx = 3
                    elif num_bands in {12, 13}:
                        b4_idx = 3
                        b8_idx = 7
                    elif num_bands == 2:
                        b4_idx = 0
                        b8_idx = 1

                mode = "sentinel2_multispectral" if (b4_idx is not None and b8_idx is not None) else "optical_rgb"

                return ImageMetadata(
                    path=file_path,
                    format=ext.lstrip(".").upper(),
                    width=width,
                    height=height,
                    num_bands=num_bands,
                    dtype=dtype,
                    has_geospatial=has_geospatial,
                    crs_epsg=crs_epsg,
                    crs_wkt=crs_wkt,
                    transform=transform,
                    bounds=bounds,
                    valid_pixel_percentage=valid_pixel_pct,
                    detected_mode=mode,
                    b4_red_index=b4_idx,
                    b8_nir_index=b8_idx,
                )
        except Exception as e:
            # Fall back to PIL if rasterio read fails
            pass

    # PIL fallback for standard images (PNG, JPG, or unreferenced TIFF)
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            mode_str = img.mode
            num_bands = len(mode_str) if mode_str not in {"L", "1"} else 1
            arr = np.array(img)
            # Calculate non-zero / non-blank valid pixels
            if arr.ndim == 2:
                valid_mask = arr > 0
            else:
                valid_mask = np.any(arr > 0, axis=-1)
            valid_pixel_pct = float(np.count_nonzero(valid_mask) / valid_mask.size * 100.0)

            return ImageMetadata(
                path=file_path,
                format=ext.lstrip(".").upper(),
                width=width,
                height=height,
                num_bands=num_bands,
                dtype=str(arr.dtype),
                has_geospatial=False,
                valid_pixel_percentage=valid_pixel_pct,
                detected_mode="optical_rgb",
            )
    except Exception as e:
        raise ImageValidationError(f"Could not read image '{file_path}': {str(e)}")


def validate_bitemporal_pair(t1_path: str, t2_path: str) -> PairValidationResult:
    """
    Validates a bi-temporal pair (T1 and T2).
    Ensures:
    1. Both images are readable and valid.
    2. Dimensions match exactly. Arbitrary resizing or cropping is strictly prohibited.
    3. Valid-pixel percentage >= 40% (avoids processing all-black or nodata rasters).
    4. Band compatibility: both are either Sentinel-2 multispectral or both are optical RGB.
    5. CRS & spatial extent compatibility if geospatial metadata is present.
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        t1_meta = inspect_single_image(t1_path)
    except ImageValidationError as e:
        errors.append(f"T1 Validation Failed: {str(e)}")
        t1_meta = None

    try:
        t2_meta = inspect_single_image(t2_path)
    except ImageValidationError as e:
        errors.append(f"T2 Validation Failed: {str(e)}")
        t2_meta = None

    if errors or t1_meta is None or t2_meta is None:
        return PairValidationResult(
            is_compatible=False,
            mode="invalid",
            width=0,
            height=0,
            t1_meta=t1_meta,
            t2_meta=t2_meta,
            errors=errors,
            warnings=warnings,
        )

    # 1. Dimension check
    if t1_meta.width != t2_meta.width or t1_meta.height != t2_meta.height:
        errors.append(
            f"Dimension mismatch between T1 ({t1_meta.width}x{t1_meta.height}) and T2 ({t2_meta.width}x{t2_meta.height}). "
            "Bi-temporal pairs must represent the exact same pixel dimensions. Arbitrary resizing or cropping is rejected."
        )

    # 2. Valid pixel percentage check
    if t1_meta.valid_pixel_percentage < 40.0:
        errors.append(
            f"T1 image contains too few valid pixels ({t1_meta.valid_pixel_percentage:.1f}% valid). Minimum required is 40%."
        )
    if t2_meta.valid_pixel_percentage < 40.0:
        errors.append(
            f"T2 image contains too few valid pixels ({t2_meta.valid_pixel_percentage:.1f}% valid). Minimum required is 40%."
        )

    # 3. Geospatial metadata compatibility check
    if t1_meta.has_geospatial and t2_meta.has_geospatial:
        if t1_meta.crs_epsg and t2_meta.crs_epsg and t1_meta.crs_epsg != t2_meta.crs_epsg:
            errors.append(
                f"CRS mismatch: T1 has EPSG:{t1_meta.crs_epsg} while T2 has EPSG:{t2_meta.crs_epsg}."
            )
        # Check extent overlap if bounds exist
        if t1_meta.bounds and t2_meta.bounds:
            b1, b2 = t1_meta.bounds, t2_meta.bounds
            # Approximate overlap check
            if abs(b1[0] - b2[0]) > 1e-4 or abs(b1[1] - b2[1]) > 1e-4:
                warnings.append("Slight bounding box offset detected between T1 and T2 geospatial metadata.")
    elif t1_meta.has_geospatial != t2_meta.has_geospatial:
        warnings.append(
            "One image has geospatial metadata while the other does not. Image-relative statistics will be used."
        )

    # 4. Mode determination
    if t1_meta.detected_mode == "sentinel2_multispectral" and t2_meta.detected_mode == "sentinel2_multispectral":
        final_mode = "sentinel2_multispectral"
    elif t1_meta.detected_mode == "optical_rgb" and t2_meta.detected_mode == "optical_rgb":
        final_mode = "optical_rgb"
    else:
        # Mismatched modes
        if t1_meta.detected_mode == "sentinel2_multispectral" and t2_meta.detected_mode != "sentinel2_multispectral":
            errors.append("T1 is multispectral with Red/NIR bands, but T2 is standard RGB without NIR.")
        elif t2_meta.detected_mode == "sentinel2_multispectral" and t1_meta.detected_mode != "sentinel2_multispectral":
            errors.append("T2 is multispectral with Red/NIR bands, but T1 is standard RGB without NIR.")
        final_mode = "incompatible"

    is_compatible = len(errors) == 0

    return PairValidationResult(
        is_compatible=is_compatible,
        mode=final_mode,
        width=t1_meta.width,
        height=t1_meta.height,
        t1_meta=t1_meta,
        t2_meta=t2_meta,
        errors=errors,
        warnings=warnings,
    )
